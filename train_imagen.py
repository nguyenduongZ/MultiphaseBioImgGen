import os, sys
import re
import gc
import torch
import random
import numpy as np

from tqdm import tqdm
from PIL import Image
from accelerate import Accelerator

from dataset import get_ds
from model import get_model
from utils import Opt, Logging, folder_setup, save_config, EarlyStopping, plot_learning_curve

class FakeAccelerator:
    def __init__(self, device):
        self.device = device
        self.is_main_process = True  

def train_imagen(opt: Opt):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    # Device setup
    if opt.conductor["trainer"]["multi_gpu"]:
        accelerator = Accelerator(mixed_precision="fp16") 
    else:
        torch.cuda.set_device(opt.conductor['trainer']['idx'])
        device = torch.device(f"cuda:{opt.conductor['trainer']['idx']}" if torch.cuda.is_available() else "cpu")
        accelerator = FakeAccelerator(device)

    # Seed setup
    seed = opt.conductor["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Folder setup and save setting
    opt.args.exp_dir = folder_setup(opt)
    save_config(opt, opt.args.exp_dir)
    
    model_checkpoint_dir = os.path.join(opt.args.exp_dir, "model")
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    
    sample_images_dir = os.path.join(opt.args.exp_dir, "sample_images")
    os.makedirs(sample_images_dir, exist_ok=True)
    
    unet_number = opt.conductor["trainer"]["unet_number"]
    
    # Logging setup 
    log_interface = Logging(opt)
    
    # Training loop
    k_folds = 5
    fold_results = []
    
    for fold in tqdm(range(k_folds), desc="Training Folds", position=0, leave=True):
        opt.logger.info(f"Training Fold {fold + 1 }/{k_folds}")

        if 'model' in locals():
            del model.trainer 
            del model
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
        
        train_ds, valid_ds, _, train_dl, valid_dl, valid_patient_dl = get_ds(opt=opt)
        
        opt.args.num_train_sample = len(train_ds)
        opt.args.num_valid_sample = len(valid_ds)
        opt.args.num_train_batch = len(train_dl)
        opt.args.num_valid_batch = len(valid_dl)
        
        opt.logger.info(f"LENGTH TRAIN: {len(train_dl)} - LENGTH VALID: {len(valid_dl)} - LENGTH VALID PATIENT: {len(valid_patient_dl)}")

        # Model setup
        model = get_model(opt=opt, device=device)
        
        if hasattr(model, "trainer"):
            model.trainer.train_dl_iter = None
            model.trainer.valid_dl_iter = None
            model.trainer.train_dl = None
            model.trainer.valid_dl = None
            model.trainer.optimizer = None
            model.trainer.scheduler = None
    
        model.trainer.add_train_dataloader(dl=train_dl)
        model.trainer.add_valid_dataloader(dl=valid_dl)
        
        if opt.args.wandb and accelerator.is_main_process:
            log_interface.watch(model.trainer)
        
        best_valid_loss = float("inf")
        early_stopping = EarlyStopping(opt)
        
        train_losses = []
        valid_losses = []
        
        iteration_bar = tqdm(range(1, (opt.conductor["trainer"]["iterations"] // k_folds) + 1), desc=f"Fold {fold+1} Training", position=1, leave=False)
        for i in iteration_bar:
            # Training loop            
            loss = model.trainer.train_step(unet_number=unet_number)
            train_losses.append(loss)
            iteration_bar.set_postfix({"Train Loss": f"{loss:.4f}"})
            
            if accelerator.is_main_process:
                log_interface(f"fold{fold+1}/train/loss", loss, step=i)
            
            #    
            if not (i % opt.conductor["validation"]["interval"]["valid_loss"]):
                with torch.no_grad():
                    valid_loss = model.trainer.valid_step(unet_number=unet_number)
                valid_losses.append(valid_loss)
                
                iteration_bar.set_postfix({"Train Loss": f"{loss:.4f}", "Valid Loss": f"{valid_loss:.4f}"})
                
                if accelerator.is_main_process:
                    log_interface(f"fold{fold+1}/valid/loss", valid_loss, step=i)
                    log_interface(f"fold{fold+1}/iteration", i, step=i)
                
                # opt.logger.info(f"Fold {fold+1} | Iteration {i}: Train Loss {loss:.4f} - Valid Loss {valid_loss:.4f}")    
            
                if early_stopping(valid_loss):
                    opt.logger.info(f"Early stopping triggered at iteration {i} in Fold {fold+1}!")
                    break
            # #
            # if valid_loss < best_valid_loss and accelerator.is_main_process:
            #     best_valid_loss = valid_loss
            #     best_checkpoint_path = os.path.join(model_checkpoint_dir, f"fold{fold+1}_best.pt")
            #     model.trainer.save(best_checkpoint_path)
            #     log_interface.log_model(best_checkpoint_path, i)
            #     opt.logger.info(f"Fold {fold+1} | Best Model Saved - Valid Loss {best_valid_loss:.4f}")
            
            #
            if not (i % opt.conductor["validation"]["interval"]["validate_model"]):
                # 
                if opt.conductor["validation"]["display_sample"]:
                    for k in range(int(np.ceil(opt.conductor["validation"]["sample_quantity"] / opt.conductor["trainer"]["batch_size"]))):
                        _, embed_batch, text_batch = next(iter(valid_patient_dl))
                        
                        sample_images = model.trainer.sample(
                            text_embeds=embed_batch.to(device),
                            return_pil_images=True,
                            stop_at_unet_number=unet_number,
                            use_tqdm=False,
                            cond_scale=opt.conductor["validation"]["cond_scale"]
                        )
                        
                        for j, sample_image in enumerate(sample_images):
                            safe_text = re.sub(r"[ ,]+", "_", re.sub(r"\.{2,}", ".", text_batch[j])) 
                            image_save_path = os.path.join(sample_images_dir, f"fold{fold+1}_iter{i:06d}-u{unet_number}-{j:05d}-{safe_text}.png")
                            sample_image.save(image_save_path)
                            
                            if opt.args.wandb:
                                validation_step = (i // opt.conductor["validation"]["interval"]["valid_loss"]) * opt.conductor["validation"]["interval"]["valid_loss"]
                                log_interface.log_images(
                                    {f"fold{fold+1}_{safe_text}": {"Generated Image": sample_image}}, validation_step
                                )
                # 
                if accelerator.is_main_process:
                    checkpoint_path = os.path.join(model_checkpoint_dir, f"fold{fold+1}_checkpoint.pt")
                    model.trainer.save(checkpoint_path)
                    # log_interface.log_model(checkpoint_path, i)
                    opt.logger.info(f"Fold {fold+1} | Saved checkpoint at iteration {i}")
                            
        fold_results.append(best_valid_loss)
        
        if accelerator.is_main_process:
            plot_learning_curve(train_losses, valid_losses, fold+1)
        
    avg_loss = sum(fold_results) / k_folds    
    opt.logger.info(f"Training Completed! Average Validation Loss: {avg_loss:.4f}")

    if accelerator.is_main_process:   
        log_interface.close()