import os, sys
sys.path.append(os.path.abspath(os.curdir))

import torch
import logging
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from datetime import datetime
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def rescale_tensor(x):
    return 2 * (x - torch.min(x)) / (torch.max(x) - torch.min(x)) - 1

def compute_lpips(
    cfg, 
    real_image_save_path: str, 
    sample_image_save_path: str,
    device: torch.device,
    logger: logging.Logger, 
    timestamp
):
    if not cfg.conductor["testing"]["LearnedPerceptualImagePatchSimilarity"]["usage"]:
        return None
    
    lpips = LearnedPerceptualImagePatchSimilarity(**cfg.conductor["testing"]["LearnedPerceptualImagePatchSimilarity"]["params"])
    scores = []
    logger.info("LPIPS initialized")
    
    real_image_path_list = glob(real_image_save_path + "*_real_image_batch.pt")
    sample_image_path_list = glob(sample_image_save_path + "*_sample_image_batch.pt")

    for real_path, sample_path in tqdm(zip(real_image_path_list, sample_image_path_list), total=len(real_image_path_list), disable=False):
        real_image_batch = rescale_tensor(torch.load(real_path).to(device))
        sample_image_batch = rescale_tensor(torch.load(sample_path).to(device))
        
        scores.append(lpips(real_image_batch, sample_image_batch).tolist())
        
        lpips_mean = np.mean(scores)
        logger.info(f"LPIPS mean result: {lpips_mean}")
        
        pd.DataFrame({
            "cond_scale": cfg.conductor["cond_scale"],
            "Dataset": cfg.datasets["name"],
            "Model": cfg.models["name"],
            "lpips": scores
        }).to_json(sample_image_save_path + timestamp + "_lpips.json")
    
    return lpips_mean

# if __name__ == "__main__":
#     from omegaconf import OmegaConf
#     cfg = OmegaConf.load("./results/testing/vindr_multiphase_imagen/unet1/cond_scale_6/2025-04-27_13-57-03/.hydra/config.yaml")
    
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger("Evaluate LPIPS")
#     logger.info("Starting elvaluate")
    
#     real_image_save_path = "./results/testing/vindr_multiphase_imagen/unet1/cond_scale_6/2025-04-27_13-57-03/real_images/cond_scale6_loss_weighting_p2/42"
#     sample_image_save_path = "./results/testing/vindr_multiphase_imagen/unet1/cond_scale_6/2025-04-27_13-57-03/sample_images/cond_scale6_loss_weighting_p2/42"
    
#     idx = 1
#     device = torch.device(f"cuda:{idx}" if torch.cuda.is_available() else "cpu")
    
#     lpips_mean = compute_lpips(
#         cfg=cfg,
#         real_image_save_path=real_image_save_path,
#         sample_image_save_path=sample_image_save_path,
#         device=device,
#         logger=logger,
#         timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
#     )
    
#     print(f"Result {lpips_mean}")