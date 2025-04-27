import os, sys
sys.path.append(os.path.abspath(os.curdir))

import gc
import json
import hydra
import torch
import logging
import numpy as np
import pandas as pd
import torchvision.transforms as T

from tqdm import tqdm
from datetime import datetime
from torchvision.utils import save_image
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from src.datasets.getds import get_ds
from src.utils import MasterLogger, Logging, create_grid_image
from src.models.imagen.build_imagen import ImagenBuilder
from src.metrics import compute_fid, compute_cmmd, compute_kid, compute_clean_fid, compute_fcd, compute_lpips

def text_2_image(
    cfg, 
    sample_dl, 
    model, 
    device: torch.device,
    logger: logging.Logger,
    logging_manager: Logging,
    unet_number: int, 
    save_samples: bool, 
    save_image_tensors: bool, 
    sample_folder: str
):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    cond_scale = cfg.conductor["cond_scale"]
    seed = cfg.conductor["seed"]
    loss_weighting = cfg.conductor["testing"]["loss_weighting"]
    
    real_image_save_path = sample_folder + f"/real_images/cond_scale{cond_scale}_loss_weighting_{loss_weighting}/{seed}"
    sample_image_save_path = sample_folder + f"/sample_images/cond_scale{cond_scale}_loss_weighting_{loss_weighting}/{seed}"
    
    os.makedirs(real_image_save_path, exist_ok=True)
    os.makedirs(sample_image_save_path, exist_ok=True)

    lower_batch = cfg.conductor["testing"]["lower_batch"]
    upper_batch = cfg.conductor["testing"]["upper_batch"]
    batch_size = cfg.conductor["trainer"]["batch_size"]
    
    total_batches = upper_batch - lower_batch
    total_images_expected = total_batches * batch_size

    tqdm_bar = tqdm(total=total_batches, desc="Generating samples", dynamic_ncols=True)
    
    metrics_history = []
    
    for i, (image_batch, embed_batch, text_batch) in enumerate(sample_dl):
        if i < lower_batch:
            continue
        if i >= upper_batch:
            break        
        
        real_image_batch = image_batch.to(device)
        
        with torch.autocast("cuda"):
            with torch.no_grad():
                sample_image_batch = model.trainer.sample(
                    text_embeds=embed_batch.to(device),
                    return_pil_images=False,
                    stop_at_unet_number=unet_number,
                    cond_scale=cond_scale
                )
                
        logger.info(f"Processing batch {i}/{upper_batch} - {batch_size} images per batch")
            
        real_image_batch = torch.clamp(real_image_batch, 0, 1)
        sample_image_batch = torch.clamp(sample_image_batch, 0, 1)
        
        if save_image_tensors:
            torch.save(real_image_batch, os.path.join(real_image_save_path, f"{i:05d}_real_image_batch.pt"))
            torch.save(sample_image_batch, os.path.join(sample_image_save_path, f"{i:05d}_sample_image_batch.pt"))
            
            with open(os.path.join(real_image_save_path, f"{i:05d}_text_batch.json"), "w") as f:
                json.dump(text_batch, f)
                
        if save_samples:
            os.makedirs(os.path.join(real_image_save_path, "images"), exist_ok=True)
            os.makedirs(os.path.join(sample_image_save_path, "images"), exist_ok=True)
            
            for j, (real_img, sample_img, text) in enumerate(zip(real_image_batch, sample_image_batch, text_batch)):
                image_index = i * len(real_image_batch) + j
                
                sanitized_text = "".join(c if c.isalnum() or c in " _-" else "_" for c in text)

                real_image_filename = os.path.join(real_image_save_path, f"images/{image_index:05d}-{sanitized_text}.png")
                sample_image_filename = os.path.join(sample_image_save_path, f"images/{image_index:05d}-{sanitized_text}.png")
                
                save_image(real_img, real_image_filename)
                save_image(sample_img, sample_image_filename)

                if j == 0:
                    real_image_pil = T.ToPILImage()(real_img.cpu())
                    sample_img_pil = T.ToPILImage()(sample_img.cpu())
                    
                    grid_image = create_grid_image(
                        original_image=real_image_pil,
                        prompt=text,
                        generated_image=sample_img_pil
                    )
                    
                    logging_manager.log_images(
                        grid_image=grid_image,
                        prompt=text,
                        iteration=i,
                        index=image_index
                    )
        
        logger.info(f"Computing metrics for batch {i}")
        fid_result = compute_fid(cfg=cfg, logger=logger, real_image_save_path=real_image_save_path, sample_image_save_path=sample_image_save_path, device=device)
        clean_fid_score = compute_clean_fid(cfg=cfg, logger=logger, real_image_save_path=real_image_save_path, sample_image_save_path=sample_image_save_path)
        fcd_score = compute_fcd(cfg=cfg, logger=logger, real_image_save_path=real_image_save_path, sample_image_save_path=sample_image_save_path)
        lpips_mean = compute_lpips(cfg=cfg, logger=logger, real_image_save_path=real_image_save_path, sample_image_save_path=sample_image_save_path, device=device, timestamp=timestamp)
        
        logging_manager("FID_per_batch", fid_result, step=i)
        logging_manager("Clean_FID_per_batch", clean_fid_score, step=i)
        logging_manager("FCD_per_batch", fcd_score, step=i)
        logging_manager("LPIPS_per_batch", lpips_mean, step=i)

        metrics_history.append({
            'batch': i,
            'cond_scale': cond_scale,
            'Dataset': cfg.datasets['name'],
            'Model': cfg.models['name'],
            'FID': fid_result,
            'Clean_FID': clean_fid_score,
            'FCD': fcd_score,
            'LPIPS': lpips_mean
        })
        
        torch.cuda.empty_cache()
        
        tqdm_bar.update(1)
        
    tqdm_bar.close()
    
    logger.info(f"Image generation completed! Expected: {total_images_expected}, Generated: {(upper_batch - lower_batch) * batch_size}")
    
    logger.info("Computing final metrics on all images")
    fid_result = compute_fid(cfg=cfg, logger=logger, real_image_save_path=real_image_save_path, sample_image_save_path=sample_image_save_path, device=device)
    # cmmd_mean, cmmd_std = compute_cmmd(cfg=cfg, logger=logger, real_image_save_path=real_image_save_path, sample_image_save_path=sample_image_save_path, device=device)
    kid_mean, kid_std = compute_kid(cfg=cfg, logger=logger, real_image_save_path=real_image_save_path, sample_image_save_path=sample_image_save_path, device=device)
    clean_fid_score = compute_clean_fid(cfg=cfg, logger=logger, real_image_save_path=real_image_save_path, sample_image_save_path=sample_image_save_path)
    fcd_score = compute_fcd(cfg=cfg, logger=logger, real_image_save_path=real_image_save_path, sample_image_save_path=sample_image_save_path)
    lpips_mean = compute_lpips(cfg=cfg, logger=logger, real_image_save_path=real_image_save_path, sample_image_save_path=sample_image_save_path, device=device, timestamp=timestamp)
    
    results = pd.DataFrame({
        'cond_scale': cond_scale,
        'Dataset': cfg.datasets['name'],
        # "CMMD_mean": [cmmd_mean],
        # "CMMD_std": [cmmd_std],
        'Model': cfg.models['name'],  
        'FID': [fid_result],
        'KID_mean': [kid_mean],
        'KID_std': [kid_std],
        'Clean_FID': [clean_fid_score],
        'FCD': [fcd_score],
        'LPIPS': [lpips_mean]
    })

    results.to_csv(os.path.join(sample_image_save_path, f"{timestamp}_results.csv"), index=False)
    logger.info("Metrics computed and saved successfully!")    
    
    gc.collect()
    
@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    OmegaConf.set_readonly(cfg, False)
    
    # MasterLogger
    logger = MasterLogger(cfg).logger
    
    # Device setup
    idx = cfg.conductor["trainer"]["idx"]
    torch.cuda.set_device(idx)
    device = torch.device(f"cuda:{idx}" if torch.cuda.is_available() else "cpu")
    
    # Model setup
    model_type = cfg.models["name"]
    model = ImagenBuilder(cfg=cfg, device=device, logger=logger, testing=True)
    trainer = model.trainer
    unet_number = cfg.conductor["unet_number"]
    
    # Dataset setup
    dataset_name = cfg.datasets["name"]
    # full_ds, full_dl = get_ds(cfg=cfg, logger=logger, testing=True)
    
    # logger.info(f"Dataset {dataset_name} sizes: {len(full_ds)}")
    # logger.info(f"Dataloader {dataset_name} sizes: {len(full_dl)}")

    train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl = get_ds(cfg=cfg, logger=logger)
    
    save_samples = cfg.conductor["testing"]["save_samples"]
    save_image_tensors = cfg.conductor["testing"]["save_image_tensors"]

    #
    logging = Logging(cfg)
    
    # Folder setup
    run_dir = HydraConfig.get().run.dir
    parent_dir = os.path.dirname(run_dir)
    
    text_2_image(
        cfg=cfg,
        sample_dl=test_dl,
        model=model,
        device=device,
        logger=logger,
        logging_manager=logging,
        unet_number=unet_number,
        save_samples=save_samples,
        save_image_tensors=save_image_tensors, 
        sample_folder=parent_dir
    )
    
    logger.info("Testing done!")
    logging.close()
    
if __name__ == "__main__":
    main()