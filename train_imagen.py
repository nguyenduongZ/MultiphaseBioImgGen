import os, sys
sys.path.append(os.path.abspath(os.curdir))

import random
import yaml
import wandb
import argparse
import numpy as np

import torch
from torch import nn
from datetime import datetime
from tqdm import tqdm

from utils.built_opt import Opt
from model.imagen import ImagenModel
from data.getds import get_train_valid_ds, get_train_valid_dl

parser = argparse.ArgumentParser(prog="MultiBioImgGen")

parser.add_argument("--path_data_dir", default="/mnt/sda1/home/bmestaging/duong/git/MultiphaseBioImgGen/")
parser.add_argument("--log", action="store_true",
    help='toggle to use wandb for online saving')

def train_imagen(args):
    opt = Opt()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=opt.conductor["trainer"]["idx"])
    
    # Model setup
    imagen_model = ImagenModel(opt=opt, device=device)
    unet_number = opt.conductor["trainer"]["unet_number"]
    
    # Data setup
    train_ds, valid_ds = get_train_valid_ds(opt=opt)
    sample_ds = get_train_valid_ds(opt=opt, testing=True, return_text=True)
    
    train_dl, valid_dl = get_train_valid_dl(opt=opt, train_ds=train_ds, valid_ds=valid_ds)
    sample_dl = get_train_valid_dl(opt=opt, train_ds=sample_ds)
    
    opt.conductor["trainer"]["train_size"] = train_ds.__len__()
    opt.conductor["validation"]["valid_size"] = valid_ds.__len__()
    
    imagen_model.trainer.add_train_dataloader(dl=train_dl)
    imagen_model.trainer.add_valid_dataloader(dl=valid_dl)
    
    model_checkpoint_file = opt.conductor["trainer"]["PATH_MODEL_CHECKPOINT"].replace(".pt", f"-u{unet_number}.pt")
    
    if args.log:
        wandb_run = wandb.init(**opt.conductor["wandb"])
        opt.logger.info(f"Wandb run started")
        opt.logger.debug(f"Wandb run: {wandb_run}")
        
        wandb.config.update(
            {
                "model": opt.conductor["model"]["model_type"],
                "epochs": opt.conductor["trainer"]["max_epochs"],
                "learning_rate": opt.conductor["trainer"]["lr"],
                "batch_size": opt.conductor['trainer']['batch_size'],
            }
        )
        opt.logger.debug("Hyperparameters saved to WandB")
        
        if opt.conductor["model"]["model_type"] == "Imagen":
            wandb.log({"model_configs": opt.imagen})
            
        elif opt.conductor["model"]["model_type"] == "ElucidatedImagen":
            wandb.log({"model_configs": opt.elucidated_imagen})
            
        wandb.log({"training_configs": opt.conductor})
        
        opt.logger.debug("Model and training configs uploaded to WandB")
        
    path_run_dir = opt.conductor["validation"]["PATH_TRAINING_SAMPLE"] + f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}" + f"_u{unet_number}_" + opt.conductor["model"]["model_type"]
    path_run_dir = os.path.join("./", path_run_dir)
    os.mkdir(path_run_dir)
        
    model_checkpoint_path = os.path.join(path_run_dir, model_checkpoint_file)
    
    opt.logger.info(f"Model Checkpoint: {model_checkpoint_path}")
    
    if unet_number == 1:
        model_save_path = opt.conductor["trainer"]["PATH_MODEL_SAVE1"]
        
    elif unet_number == 2:
        model_save_path = opt.conductor["trainer"]["PATH_MODEL_SAVE2"]
        
    opt.logger.info("Start training of Imagen Diffusion Model")

    for epoch in tqdm(iterable=range(1, opt.conductor["trainer"]["max_epochs"] + 1), disable=False):
        # Training
        train_loss = imagen_model.trainer.train_step(unet_number=unet_number)
        
        if args.log:
            wandb.log({"train_loss": train_loss})
        
        #
        if not (epoch % opt.conductor["validation"]["interval"]["valid_loss"]):
            valid_loss = imagen_model.trainer.valid_step(unet_number=unet_number)
            opt.logger.debug(f"Epoch validation loss-unet{unet_number}: {valid_loss}")
            
            if args.log:
                wandb.log({"valid_loss": valid_loss})
            
        #
        if not (epoch % opt.conductor["validation"]["interval"]["validate_mode"]) and imagen_model.trainer.is_main:
            with torch.no_grad():
                if opt.conductor["validation"]["display_sample"]:
                    n = int(np.ceil(opt.conductor["validation"]["sample_quantity"] / opt.conductor["trainer"]["batch_size"]))
                    for i in range(n):
                        _, _, _, embed_batch, text_batch = next(iter(sample_dl))
                        sample_images = imagen_model.trainer.sample(
                            text_embeds=embed_batch.to(device),
                            return_pil_images=True,
                            stop_at_unet_number=unet_number,
                            use_tqdm=True,
                            cond_scale=opt.conductor["validation"]["cond_scale"]
                        )
                        for j, sample_image in enumerate(sample_images):
                            image_save_path = os.path.join(path_run_dir, f"e{epoch}-u{unet_number}-{j:05d}-{text_batch[j]}.png")
                            sample_image.save(image_save_path)
                
                imagen_model.trainer.save(model_checkpoint_path)
                
                if args.log:
                    wandb.log({"model": model_checkpoint_path})                    
                            
        opt.logger.info("Imagen Training Finished")
        
        imagen_model.trainer.save(model_save_path)
        
        if args.log:
            wandb.log({"model": model_save_path})
            wandb.finish()
            opt.logger.info("WandB run stopped")
            
def main():
    with open("./configs/data.yaml") as f:
        config = yaml.safe_load(f)
        
        args = parser.parse_args()
        config["PATH_DATA_DIR"] = args.path_data_dir
        config["log"] = args.log
        
        with open("./configs/data.yaml", "w") as f:
            yaml.dump(config, f)
            
        train_imagen(args)
        
if __name__ == "__main__":
    main()