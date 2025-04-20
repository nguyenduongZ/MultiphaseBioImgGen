import os, sys
sys.path.append(os.path.abspath(os.curdir))

import torch
import logging

from glob import glob
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance


def compute_fid(cfg, logger: logging.Logger, real_image_save_path, sample_image_save_path, device):
    if not cfg.conductor["testing"]["FrechetInceptionDistance"]["usage"]:
        return None
    
    fid = FrechetInceptionDistance(**cfg.conductor["testing"]["FrechetInceptionDistance"]["params"]).to(device)
    logger.info("FID initialized")

    real_image_path_list = sorted(glob(os.path.join(real_image_save_path, "*_real_image_batch.pt")))
    sample_image_path_list = sorted(glob(os.path.join(sample_image_save_path, "*_sample_image_batch.pt")))

    if len(real_image_path_list) == 0 or len(sample_image_path_list) == 0:
        raise ValueError(f"Image not found! Check path:\nReal: {real_image_save_path}\nFake: {sample_image_save_path}")

    assert len(real_image_path_list) == len(sample_image_path_list), "The number of real and fake photos does not match!"

    for real_path, sample_path in tqdm(zip(real_image_path_list, sample_image_path_list), total=len(real_image_path_list), disable=False):
        real_image_batch = torch.load(real_path).to(device)
        sample_image_batch = torch.load(sample_path).to(device)

        if real_image_batch.shape[0] < 2 or sample_image_batch.shape[0] < 2:
            raise RuntimeError("Each batch must have at least 2 images")

        fid.update(real_image_batch, real=True)
        fid.update(sample_image_batch, real=False)

    fid_result = fid.compute().cpu().item()
    logger.info(f"FID result: {fid_result}")
        
    return fid_result

