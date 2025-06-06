import os
import torch
import logging

from cleanfid import fid as clean_fid

def compute_clean_fid(cfg, real_image_path, sample_image_path, device: torch.device):
    if not cfg.conductor.testing.CleanFID.usage:
        return None
    
    params = cfg.conductor.testing.CleanFID.params
    logging.info('Clean_FID initialized')

    fdir1 = os.path.join(real_image_path, 'images')
    fdir2 = os.path.join(sample_image_path, 'images')
    
    clean_fid_score = clean_fid.compute_fid(
        fdir1=fdir1,
        fdir2=fdir2,
        device=device,
        use_dataparallel=False,
        **params
    )
    logging.info(f"Clean_FID result: {clean_fid_score:0.2f}")
    return clean_fid_score

def compute_fcd(cfg, real_image_path, sample_image_path, device: torch.device):
    if not cfg.conductor.testing.FrechetCLIPDistance.usage:
        return None
    
    params = cfg.conductor.testing.FrechetCLIPDistance.params
    logging.info('FCD initialized')

    fdir1 = os.path.join(real_image_path, 'images')
    fdir2 = os.path.join(sample_image_path, 'images')
    
    fcd_score = clean_fid.compute_fid(
        fdir1=fdir1,
        fdir2=fdir2,
        device=device,
        use_dataparallel=False,
        **params
    )
    logging.info(f"FCD result: {fcd_score:0.2f}")
    return fcd_score