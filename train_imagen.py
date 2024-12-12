import os, sys
from rich.progress import track
import random
import numpy as np

import torch
from torch import nn
from datetime import datetime
from tqdm import tqdm

from torchvision.utils import save_image
from data import get_ds
from utils import compute_mse, save_json
from imagen_pytorch import Unet, Imagen, ImagenTrainer
import wandb
import logging

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_imagen(args):
    # Seed setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=args.idx)  
    print("CUDA available? ", torch.cuda.is_available())
    print("Device being used: ", device)

    # Dataset setup
    data, args = get_ds(args)
    _, _, _, train_dl, valid_dl, test_dl = data

    print(f"#TRAIN batch: {len(train_dl)}")
    print(f"#VAL batch: {len(valid_dl)}")
    print(f"#TEST batch: {len(test_dl)}")

    now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    if args.log:
        run = wandb.init(
        project='MultiphaseBioImgGen',
        # entity='truelove',
        config=args,
        name=now,
        force=True
        )

    run_dir = os.getcwd() + '/runs'
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    
    sv_dir = run_dir + f"/{now}"
    if not os.path.exists(sv_dir):
        os.mkdir(sv_dir)

    best_model_path = sv_dir + f'/best.pt'
    last_model_path = sv_dir + f'/last.pt'

    unet1 = Unet(
        dim=32,
        cond_dim=512,
        dim_mults=(1, 2, 4, 8),
        num_resnet_blocks=3,
        layer_attns=(False, True, True, True),
    )

    unet2 = Unet(
        dim=32,
        cond_dim=512,
        dim_mults=(1, 2, 4, 8),
        num_resnet_blocks=(2, 4, 8, 8),
        layer_attns=(False, False, False, True),
        layer_cross_attns=(False, False, False, True)
    )

    imagen = Imagen(
        unets=(unet1, unet2),
        text_encoder_name='t5-large',  
        image_sizes=(64, 256),         
        timesteps=1000,                
        cond_drop_prob=0.1             
    ).to(device)

    trainer = ImagenTrainer(imagen, cosine_decay_max_steps=args.cosine_decay_max_steps).to(device)
    
    print(f'Number of parameters: {count_parameters(unet1)}')
    logging.info(f'Number of parameters: {count_parameters(unet1)}')

    train_loss, valid_loss = [], []
    mse_scores = {channel: [] for channel in range(args.num_channels)}
    best_loss = float('inf')
    best_mse = np.full(args.num_channels, float('inf'))
    
    for epoch in range(1, args.epochs + 1):
        trainer.train()
        train_loss_epoch = 0

        # Training loop
        for _, (images, text_embeds) in enumerate(tqdm(train_dl, desc=f"Training Epoch {epoch}")):
            # Forward pass and loss calculation
            loss = trainer(
                images,
                text_embeds=text_embeds,
                unet_number=args.train_unet
            )

            trainer.update(unet_number = args.train_unet)

            train_loss_epoch += loss * args.bs

        avg_train_loss = train_loss_epoch / len(train_dl.dataset)
        train_loss.append(avg_train_loss)

        print(f"Epoch {epoch}: Training Loss = {avg_train_loss:.4f}")
        logging.info(f"Epoch {epoch}: Training Loss = {avg_train_loss:.4f}")

        # Validation loop
        if epoch % args.valid_per_step_num == 0 or epoch == args.epochs:
            trainer.eval()
            valid_loss_epoch = 0
            valid_mse_epoch = np.zeros(args.num_channels)
            valid_nsamples = 0

            with torch.no_grad():
                for images, text_embeds in tqdm(valid_dl, desc=f"Validation Epoch {epoch}"):
                    # Forward pass
                    loss = trainer(
                        images,
                        text_embeds=text_embeds,
                        unet_number=args.train_unet
                    )

                    # Compute MSE for each channel
                    for channel in range(args.num_channels):
                        valid_mse_epoch[channel] += compute_mse(
                            images[:, channel, ...].unsqueeze(1), 
                            trainer.imagen.unets[args.train_unet - 1], 
                            device=device
                        )

                    valid_loss_epoch += loss * args.bs
                    valid_nsamples += args.bs

            avg_valid_loss = valid_loss_epoch / valid_nsamples
            avg_valid_mse = valid_mse_epoch / valid_nsamples

            valid_loss.append(avg_valid_loss)
            for channel in range(args.num_channels):
                mse_scores[channel].append(avg_valid_mse[channel])

            print(f"Validation Loss: {avg_valid_loss:.4f}")
            logging.info(f"Validation Loss: {avg_valid_loss:.4f}")
            print(f"Validation MSE Scores: {avg_valid_mse}")
            logging.info(f"Validation MSE Scores: {avg_valid_mse}")

            # Sampling and saving generated images
            sample_dir = os.path.join(sv_dir, f"epoch_{epoch}_samples")
            os.makedirs(sample_dir, exist_ok=True)
            num_samples = 5

            sample_counter = 0
            for batch_idx, (images, text_embeds) in enumerate(tqdm(valid_dl, desc="Sampling Images")):
                if sample_counter >= num_samples:
                    break

                samples = trainer.sample(
                    cond_images=images.to(device),
                    text_embeds=text_embeds.to(device),
                    batch_size=args.bs
                )

                for i in range(args.bs):
                    if sample_counter >= num_samples:
                        break
                    
                    save_image(images[i], os.path.join(sample_dir, f"original_{sample_counter}.png"))
                    save_image(samples[i], os.path.join(sample_dir, f"generated_{sample_counter}.png"))

                    sample_counter += 1

            # Save best model
            if avg_valid_loss < best_loss:
                best_loss = avg_valid_loss
                trainer.save(imagen.state_dict(), best_model_path)

            # Save best MSE for each channel
            for channel in range(args.num_channels):
                if avg_valid_mse[channel] < best_mse[channel]:
                    best_mse[channel] = avg_valid_mse[channel]

            # Save last model
            trainer.save(imagen.state_dict(), last_model_path)

    # Save training and validation results
    save_json({"train_loss": train_loss, "valid_loss": valid_loss}, os.path.join(sv_dir, "losses.json"))
    for channel in range(args.num_channels):
        save_json({"mse_scores": mse_scores[channel]}, os.path.join(sv_dir, f"mse_scores_channel_{channel}.json"))

    if args.log:
        wandb.finish()

    print("Training completed.")