import os, sys
sys.path.append(os.path.abspath(os.curdir))

import torch
import logging

from glob import glob
from tqdm import tqdm
from torchmetrics.image.kid import KernelInceptionDistance

def compute_kid(
    cfg, 
    real_image_save_path: str, 
    sample_image_save_path: str,
    device: torch.device,
    logger: logging.Logger, 
):
    if not cfg.conductor["testing"]["KernelInceptionDistance"]["usage"]:
        return None, None
    
    kid = KernelInceptionDistance(**cfg.conductor["testing"]["KernelInceptionDistance"]["params"]).to(device)
    logger.info("KID initialized")
    
    real_image_path_list = glob(real_image_save_path + "/*_real_image_batch.pt")
    sample_image_path_list = glob(sample_image_save_path + "/*_sample_image_batch.pt")

    for real_path, sample_path in tqdm(zip(real_image_path_list, sample_image_path_list), total=len(real_image_path_list), desc="Computing KID", disable=False):
        real_image_batch = torch.load(real_path).to(device)
        sample_image_batch = torch.load(sample_path).to(device)
        
        kid.update(real_image_batch, real=True)
        kid.update(sample_image_batch, real=False)
        
    kid_mean, kid_std = kid.compute()
    kid_mean, kid_std = (kid_mean.cpu().item(), kid_std.cpu().item())
    logger.info(f"KID result: Mean {kid_mean} - STD {kid_std}")
    
    return kid_mean, kid_std

# if __name__ == "__main__":
#     from omegaconf import OmegaConf
#     cfg = OmegaConf.load("./results/testing/vindr_multiphase_imagen/unet1/cond_scale_6/2025-04-27_13-57-03/.hydra/config.yaml")
    
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger("Evaluate FID")
#     logger.info("Starting elvaluate")
    
#     real_image_save_path = "./results/testing/vindr_multiphase_imagen/unet1/cond_scale_6/2025-04-27_13-57-03/real_images/cond_scale6_loss_weighting_p2/42"
#     sample_image_save_path = "./results/testing/vindr_multiphase_imagen/unet1/cond_scale_6/2025-04-27_13-57-03/sample_images/cond_scale6_loss_weighting_p2/42"
    
#     idx = 1
#     device = torch.device(f"cuda:{idx}" if torch.cuda.is_available() else "cpu")
    
#     kid_mean, kid_std = compute_kid(
#         cfg=cfg,
#         real_image_save_path=real_image_save_path,
#         sample_image_save_path=sample_image_save_path,
#         device=device,
#         logger=logger
#     )
    
#     print(f"Result {kid_mean}, {kid_std}")
