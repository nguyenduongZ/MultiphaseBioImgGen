import os, sys
import re
import gc
import torch
import wandb
import random
import numpy as np

from tqdm import tqdm
from PIL import Image

from dataset import get_ds
from model import get_model
from utils import Opt, Logging, folder_setup, save_config, EarlyStopping, plot_learning_curve 

def train_imagen(opt: Opt):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # Seed setup
    seed = opt.config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Device setup
    idx = opt.config["trainer"]["idx"]
    torch.cuda.set_device(idx)
    device = torch.device(f"cuda:{idx}" if torch.cuda.is_available() else "cpu")
    
    # Folder setup and save setting
    opt.args.exp_dir = folder_setup(opt)
    save_config(opt, opt.args.exp_dir)
    
    model_checkpoint_dir = os.path.join(opt.args.exp_dir, "model")
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    
    sample_images_dir = os.path.join(opt.args.exp_dir, "sample_images")
    os.makedirs(sample_images_dir, exist_ok=True)
    
    unet_number = opt.config["trainer"]["unet_number"]
    
    # Logging setup 
    log_interface = Logging(opt)
    
    # Model setup
    model = get_model(opt=opt, device=device)
    unet_number = opt.config["trainer"]["unet_number"]
    
    # Data setup
    train_ds, valid_ds, train_dl, valid_dl = get_ds(opt=opt)
        
    opt.args.num_train_sample = len(train_ds)
    opt.args.num_valid_sample = len(valid_ds)
    opt.args.num_train_batch = len(train_dl)
    opt.args.num_valid_batch = len(valid_dl)
        
    opt.logger.info(f"LENGTH TRAIN: {len(train_dl)} - LENGTH VALID: {len(valid_dl)}")

    model.trainer.add_train_dataloader(dl=train_dl)
    model.trainer.add_valid_dataloader(dl=valid_dl)
        
    if opt.args.wandb:
        log_interface.watch(model.trainer)
        
    # best_valid_loss = float("inf")
    early_stopping = EarlyStopping(opt)
        
    train_losses = []
    valid_losses = []
        
    iteration_bar = tqdm(range(1, (opt.config["trainer"]["iterations"]) + 1), desc=f"Training")
    for i in iteration_bar:
        log_interface.log_iteration(i)
        
        # Training loop            
        loss = model.trainer.train_step(unet_number=unet_number)
        train_losses.append(loss)
            
        log_interface(f"train/loss", loss)
            
        #    
        if not (i % opt.config["validation"]["interval"]["valid_loss"]):
            with torch.no_grad():
                valid_loss = model.trainer.valid_step(unet_number=unet_number)
            valid_losses.append(valid_loss)
                
            iteration_bar.set_postfix({"Train Loss": f"{loss:.4f}", "Valid Loss": f"{valid_loss:.4f}"})

            log_interface(f"valid/loss", valid_loss)
             
            opt.logger.info(f"Iteration {i}: Train Loss {loss:.4f} - Valid Loss {valid_loss:.4f}")    
            
            if early_stopping(valid_loss):
                opt.logger.info(f"Early stopping triggered at iteration {i}!")
                break
        
        log_interface(f"iteration", i)
        # #
        # if valid_loss < best_valid_loss and accelerator.is_main_process:
        #     best_valid_loss = valid_loss
        #     best_checkpoint_path = os.path.join(model_checkpoint_dir, f"fold{fold+1}_best.pt")
        #     model.trainer.save(best_checkpoint_path)
        #     log_interface.log_model(best_checkpoint_path, i)
        #     opt.logger.info(f"Fold {fold+1} | Best Model Saved - Valid Loss {best_valid_loss:.4f}")
            
        #
        if not (i % opt.config["validation"]["interval"]["validate_model"]):
            # 
            if opt.config["validation"]["display_sample"]:
                for k in range(int(np.ceil(opt.config["validation"]["sample_quantity"] / opt.config["trainer"]["batch_size"]))):
                    _, embed_batch, text_batch = next(iter(valid_dl))
                        
                    sample_images = model.trainer.sample(
                        text_embeds=embed_batch.to(device),
                        return_pil_images=True,
                        stop_at_unet_number=unet_number,
                        use_tqdm=False,
                        cond_scale=opt.config["validation"]["cond_scale_validation"]
                    )
                        
                    for j, sample_image in enumerate(sample_images):
                        safe_text = re.sub(r"[ ,]+", "_", re.sub(r"\.{2,}", ".", text_batch[j])) 
                        image_save_path = os.path.join(sample_images_dir, f"iter{i:06d}-u{unet_number}-{j:05d}-{safe_text}.png")
                        sample_image.save(image_save_path)
                            
                        if opt.args.wandb:
                            validation_step = (i // opt.config["validation"]["interval"]["valid_loss"]) * opt.config["validation"]["interval"]["valid_loss"]
                            log_interface.log_images(
                                {f"{safe_text}": {"Generated Image": sample_image}}, validation_step
                            )
            # 
            checkpoint_path = os.path.join(model_checkpoint_dir, f"checkpoint_iter{i}.pt")
            model.trainer.save(checkpoint_path, without_optim_and_sched=True)
            # log_interface.log_model(checkpoint_path, i)
            opt.logger.info(f"Saved checkpoint at iteration {i}")
    
    #    
    plot_learning_curve(train_losses, valid_losses)
    model.trainer.save(model_checkpoint_dir, f"{opt.args.model_type}_unet{unet_number}.pt")
    
    log_interface.close()
        
    opt.logger.info(f"Training Completed!")