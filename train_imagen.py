import os, sys
import re
import torch
import wandb
import random
import numpy as np

from rich.progress import track

from model import get_model
from dataset import get_ds
from utils import folder_setup, save_cfg, Logging, EarlyStopping

def train_imagen(config, args):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    # Seed setup
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True

    # Device setup
    idx = config["trainer"]["idx"]
    torch.cuda.set_device(idx)
    device = torch.device(f"cuda:{idx}" if torch.cuda.is_available() else "cpu")

    # Folder setup and save setting
    args.exp_dir = folder_setup(args)
    save_cfg(args, args.exp_dir)
    
    model_checkpoint_dir = os.path.join(args.exp_dir, "model")
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    
    sample_dir = os.path.join(args.exp_dir, "sample_images")
    os.makedirs(sample_dir, exist_ok=True)
    
    # Logging setup
    log_interface = Logging(args)

    # Model setup
    model_type = config["model"]["model_type"]
    model = get_model(config=config, model_type=model_type, device=device)
    unet_number = config["trainer"]["unet_number"]
    
    # Load checkpoint
    if unet_number == 2:
        unet1_path = os.path.join(model_checkpoint_dir, f"{model_type}_unet1.pt")
        
        if os.path.exists(unet1_path):
            print(f"Loading Unet 1 model from: {unet1_path}")
            model.trainer.load(unet1_path)
    
    model_save_path = os.path.join(model_checkpoint_dir, f"{model_type}_unet{unet_number}.pt")
    if os.path.exists(model_save_path):
        print(f"Loading checkpoint: {model_save_path}")
        model.trainer.load(model_save_path)
        
    # Data setup
    data, config = get_ds(config=config, args=args)
    _, _, train_dl, valid_dl = data
    
    print(f"LENGTH TRAIN: {len(train_dl)} - LENGTH VALID: {len(valid_dl)}")
    
    model.trainer.add_train_dataloader(dl=train_dl)
    model.trainer.add_valid_dataloader(dl=valid_dl)
    
    if args.wandb:
        log_interface.watch(model.trainer)
    
    # Early Stopping setup
    early_stopping = EarlyStopping(patience=3000, min_delta=1e-4)

    best_valid_loss = float("inf")
    table_data = {}
    
    #  Training    
    for epoch in track(range(1, config["trainer"]["epochs"] + 1), description="Training Progress"):        
        args.epoch = epoch
        
        loss = model.trainer.train_step(unet_number=unet_number)    
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

        log_interface(key="train/loss", value=loss)
        
        # Validation
        if not (epoch % config["validation"]["interval"]["valid_loss"]):
            valid_loss = model.trainer.valid_step(unet_number=unet_number)
            print(f"Valid Loss = {valid_loss:.4f}")
            
            log_interface(key="valid/loss", value=valid_loss)
            
            # Early Stopping
            if early_stopping(valid_loss):
                print(f"Early stopping triggered at epoch {epoch}.")
                break
        
        log_interface.step(epoch=epoch)
        
        if not (epoch % config["validation"]["interval"]["validate_model"]) and model.trainer.is_main and config["validation"]["display_sample"]:
            for i in range(int(np.ceil(config["validation"]["sample_quantity"] / config["trainer"]["batch_size"]))):
                original_image, embed_batch, text_batch = next(iter(valid_dl))

                sample_images = model.trainer.sample(
                    text_embeds=embed_batch.to(device),
                    return_pil_images=True,
                    stop_at_unet_number=unet_number,
                    cond_scale=config["validation"]["cond_scale"]
                )

                for j, sample_image in enumerate(sample_images):
                    safe_text = re.sub(r"[ ,]+", "_", re.sub(r"\.{2,}", ".", text_batch[j]))
                    image_save_path = os.path.join(sample_dir, f"e{epoch:05d}-u{unet_number}-{j:05d}-{safe_text}.png")
                    sample_image.save(image_save_path)

                    if text_batch[j] not in table_data:
                        table_data[text_batch[j]] = {
                            "Cond Scale": config["validation"]["cond_scale"],
                            "Prompt": text_batch[j],
                            "Original Image": wandb.Image(original_image[j])
                        }
                        
                    table_data[text_batch[j]][f"Generated Image (Epoch {epoch})"] = wandb.Image(sample_image)

                if args.wandb:
                    log_interface.log_images(table_data, epoch, config["validation"]["interval"]["validate_model"])

            model.trainer.save(os.path.join(model_checkpoint_dir, f"{args.model_type}_checkpoint-u{unet_number}.pt"))
            
        # # Save best model   
        # log_interface.step(epoch=epoch)
            
        # mean_valid_loss = log_interface.log_avg.get("valid/loss", float("inf"))
        # save_dict = {
        #     "epoch": epoch,
        #     "model_state_dict": {k: v.cpu() for k, v in model.trainer.state_dict().items()},
        #     "loss": mean_valid_loss
        # }
        
        # save_path = os.path.join(model_checkpoint_dir, "last.pt")
        # torch.save(save_dict, save_path)
            
        # del save_dict
        # torch.cuda.empty_cache()

        # if mean_valid_loss <= old_valid_loss:
        #     old_valid_loss = mean_valid_loss
        #     torch.save(save_dict, os.path.join(model_checkpoint_dir, "best.pt"))
        # torch.cuda.empty_cache()
        
        # torch.save(save_dict, os.path.join(model_checkpoint_dir, "last.pt"))
        # torch.cuda.empty_cache()
        
    model.trainer.save(os.path.join(model_checkpoint_dir, f"{args.model_type}_unet{unet_number}.pt"))
    # log_interface.log_model()

    log_interface.close()
    print("Logging closed. Training complete.")