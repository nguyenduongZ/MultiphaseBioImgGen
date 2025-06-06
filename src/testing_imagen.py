import os
import json
import time
import torch
import logging
import pandas as pd
import torchvision.transforms as T

from datetime import datetime
from torchvision.utils import save_image

from src.utils import create_grid_image, simple_slugify, LogExporter
from src.metrics import compute_fid, compute_kid, compute_clean_fid, compute_fcd, compute_cmmd, compute_lpips

def testing_imagen(cfg, model, device, test_dl, run_dir, run_name):
    log_exporter = LogExporter(cfg, run_dir, run_name)
    log_exporter.watch(model=model.imagen_model, log_freq=len(test_dl))
    trainer = model.trainer
    
    #
    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    unet = cfg.conductor.unet
    cond_scale = cfg.conductor.cond_scale
    seed = cfg.conductor.seed
    
    lower_batch = cfg.conductor.testing.lower_batch
    upper_batch = cfg.conductor.testing.upper_batch
    
    batch_size = cfg.data.batch_size
    
    real_image_path = os.path.join(run_dir, f"real_images/cond_scale{cond_scale}/seed{seed}")
    sample_image_path = os.path.join(run_dir, f"samples/cond_scale{cond_scale}/seed{seed}")
    os.makedirs(real_image_path, exist_ok=True)
    os.makedirs(sample_image_path, exist_ok=True)

    checkpoint_file = os.path.join(sample_image_path, 'checkpoint.json')
    last_completed_batch = -1
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            last_completed_batch = checkpoint.get('last_completed_batch', -1)

    metrics_history = []

    total_sampling_time = 0.0
    sample_count = 0
    
    for i, (image_batch, embed_batch, text_batch) in enumerate(test_dl):
        if i < max(lower_batch, last_completed_batch + 1):
            continue
        if i >= upper_batch:
            break
        
        real_image_batch = image_batch.to(device)
        
        logging.info(f"Sampling images with unet={unet}")
        
        start_sample_time = time.time()
        with torch.autocast('cuda'):
            with torch.no_grad():
                sample_image_batch = trainer.sample(
                    text_embeds=embed_batch.to(device),
                    return_pil_images=False,
                    stop_at_unet_number=unet,
                    cond_scale=cond_scale,
                    batch_size=batch_size
                )
        sampling_duration = time.time() - start_sample_time
        total_sampling_time += sampling_duration
        sample_count += 1
        
        logging.info(f"Batch {i}: Sampling time = {sampling_duration:.4f} seconds")
        logging.info(f"Processing batch {i}/{upper_batch} - {batch_size} images per batch")
        
        real_image_batch = torch.clamp(real_image_batch, 0, 1)
        sample_image_batch = torch.clamp(sample_image_batch, 0, 1)
        
        if cfg.conductor.testing.save_image_tensors:
            torch.save(real_image_batch, os.path.join(real_image_path, f"{i:05d}_real_image_batch.pt"))
            torch.save(sample_image_batch, os.path.join(sample_image_path, f"{i:05d}_sample_image_batch.pt"))
            with open(os.path.join(real_image_path, f"{i:05d}_text_batch.json"), 'w') as f:
                json.dump(text_batch, f)
        
        if cfg.conductor.testing.save_samples:
            os.makedirs(os.path.join(real_image_path, 'images'), exist_ok=True)
            os.makedirs(os.path.join(sample_image_path, 'images'), exist_ok=True)
            
            dir1 = os.path.join(real_image_path, 'images')
            dir2 = os.path.join(sample_image_path, 'images')
            
            for j, (real_img, sample_img, text) in enumerate(zip(real_image_batch, sample_image_batch, text_batch)):
                img_index = i * len(real_image_batch) + j
                
                real_image_filename = os.path.join(dir1, f"{img_index:05d}-{simple_slugify(text)}.png")
                sample_image_filename = os.path.join(dir2, f"{img_index:05d}-{simple_slugify(text)}.png")
                
                save_image(real_img, real_image_filename)
                save_image(sample_img, sample_image_filename)
                
                if j in range(2):
                    real_image_pil = T.ToPILImage()(real_img.cpu())
                    sample_image_pil = T.ToPILImage()(sample_img.cpu())
                    grid_image = create_grid_image(real_image_pil, text, sample_image_pil)
                    log_exporter.log_images(grid_image, text)
        
        batch_interval = cfg.conductor.testing.batch_interval            
        if not (i % batch_interval):
            logging.info(f"Computing metrics for batch {i}")
            fid_result = compute_fid(cfg, real_image_path, sample_image_path, device)
            cmmd_result = compute_cmmd(cfg, dir1, dir2, device)
            clean_fid_result = compute_clean_fid(cfg, real_image_path, sample_image_path, device)
            fcd_result = compute_fcd(cfg, real_image_path, sample_image_path, device)

            log_exporter('test/FID', fid_result)
            log_exporter('test/CMMD', cmmd_result)
            log_exporter('test/Clean_FID', clean_fid_result)
            log_exporter('test/FCD', fcd_result)
            
            metrics_history.append({
                'batch': i,
                'cond_scale': cond_scale,
                'Dataset': cfg.data['name'],
                'Model': cfg.model['name'],
                'FID': fid_result,
                'CMMD': cmmd_result,
                'Clean_FID': clean_fid_result,
                'FCD': fcd_result
            })
            
        logging.info(f"Image generation completed!\n Generated: {(upper_batch - lower_batch) * batch_size}")
        
        log_exporter.step(iteration=i)

    if sample_count > 0:
        logging.info(f"Total sampling time over {sample_count} batches: {total_sampling_time:.4f} seconds")
        logging.info(f"Average sampling time per batch: {total_sampling_time / sample_count:.4f} seconds")
    else:
        logging.info("No sampling steps were executed during testing.")

    logging.info("Computing metrics on all sample images")
    fid_result = compute_fid(cfg, real_image_path, sample_image_path, device)
    cmmd_result = compute_cmmd(cfg, dir1, dir2, device)
    kid_mean, kid_std = compute_kid(cfg, real_image_path, sample_image_path, device)
    clean_fid_result = compute_clean_fid(cfg, real_image_path, sample_image_path, device)
    fcd_result = compute_fcd(cfg, real_image_path, sample_image_path, device)
    lpips_mean = compute_lpips(cfg, real_image_path, sample_image_path, device, timestamp)
    
    results = pd.DataFrame({
        'cond_scale': cond_scale,
        'Dataset': cfg.data['name'],
        'Model': cfg.model['name'],
        'FID': [fid_result],
        'CMMD': cmmd_result,
        'KID_mean': [kid_mean],
        'KID_std': [kid_std],
        'Clean_FID': [clean_fid_result],
        'FCD': [fcd_result],
        'LPIPS_mean': [lpips_mean]
    })
    results.to_csv(os.path.join(sample_image_path, f"{timestamp}_results.csv"), index=False)
    
    log_exporter.close()
