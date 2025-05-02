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
        **cfg.conductor["testing"]["CleanFID"]["params"],
        use_dataparallel=False
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
        **cfg.conductor["testing"]["FrechetCLIPDistance"]["params"],
        use_dataparallel=False
    )
    logger.info(f"FCD result: {fcd_score}")
    
    return fcd_score

# if __name__ == "__main__":
#     from omegaconf import OmegaConf
#     cfg = OmegaConf.load("./results/testing/vindr_multiphase_imagen/unet1/cond_scale_6/2025-04-27_13-57-03/.hydra/config.yaml")
    
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger("Evaluate clean-fid & clean-kid")
#     logger.info("Starting elvaluate")
    
#     real_image_save_path = "./results/testing/vindr_multiphase_imagen/unet1/cond_scale_6/2025-04-27_13-57-03/real_images/cond_scale6_loss_weighting_p2/42"
#     sample_image_save_path = "./results/testing/vindr_multiphase_imagen/unet1/cond_scale_6/2025-04-27_13-57-03/sample_images/cond_scale6_loss_weighting_p2/42"
    
#     idx = 1
#     device = torch.device(f"cuda:{idx}" if torch.cuda.is_available() else "cpu")
    
#     clean_fid_score = compute_clean_fid(
#         cfg=cfg,
#         real_image_save_path=real_image_save_path,
#         sample_image_save_path=sample_image_save_path,
#         device=device,
#         logger=logger
#     )
    
#     fcd_score = compute_fcd(
#         cfg=cfg,
#         real_image_save_path=real_image_save_path,
#         sample_image_save_path=sample_image_save_path,
#         device=device,
#         logger=logger
#     )
    
#     print(f"Result {clean_fid_score}-{fcd_score}")