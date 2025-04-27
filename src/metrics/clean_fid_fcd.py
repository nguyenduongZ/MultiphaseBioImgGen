import os, sys
sys.path.append(os.path.abspath(os.curdir))

import torch
import logging

from cleanfid import fid as clean_fid

def compute_clean_fid(
    cfg, 
    real_image_save_path: str, 
    sample_image_save_path: str,
    device: torch.device,
    logger: logging.Logger, 
):
    if not cfg.conductor["testing"]["CleanFID"]["usage"]:
        return None
    
    fdir1 = real_image_save_path + "/images"
    fdir2 = sample_image_save_path + "/images"
    
    clean_fid_score = clean_fid.compute_fid(
        fdir1=fdir1,
        fdir2=fdir2,
        device=device,
        **cfg.conductor["testing"]["CleanFID"]["params"]
    )
    logger.info(f"Clean FID result: {clean_fid_score}")
    
    return clean_fid_score

def compute_fcd(
    cfg, 
    real_image_save_path: str, 
    sample_image_save_path: str,
    device: torch.device,
    logger: logging.Logger, 
):
    if not cfg.conductor["testing"]["FrechetCLIPDistance"]["usage"]:
        return None
        
    fdir1 = real_image_save_path + "/images"
    fdir2 = sample_image_save_path + "/images"
    
    fcd_score = clean_fid.compute_fid(
        fdir1=fdir1,
        fdir2=fdir2,
        device=device,
        **cfg.conductor["testing"]["FrechetCLIPDistance"]["params"]
    )
    logger.info(f"FCD result: {fcd_score}")
    
    return fcd_score