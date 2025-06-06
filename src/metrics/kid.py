import os
import torch
import logging

from glob import glob
from tqdm import tqdm
from torchmetrics.image.kid import KernelInceptionDistance

def compute_kid(cfg, real_image_path, sample_image_path, device: torch.device):
    if not cfg.conductor.testing.KernelInceptionDistance.usage:
        return None, None
    
    params = cfg.conductor.testing.KernelInceptionDistance.params
    kid = KernelInceptionDistance(**params).to(device=device)
    logging.info('KID initialized')
    
    real_image_path_list = sorted(glob(os.path.join(real_image_path, '*_real_image_batch.pt')))
    sample_image_path_list = sorted(glob(os.path.join(sample_image_path, '*_sample_image_batch.pt')))
    
    length_real_image_path_list = len(real_image_path_list)
    length_sample_image_path_list = len(sample_image_path_list)
    if length_real_image_path_list == 0 or length_sample_image_path_list == 0:
        raise ValueError(
            f"Image not found!\n Check path:\nReal: {real_image_path}\nFake: {real_image_path}"
        )
    assert length_real_image_path_list == length_sample_image_path_list, \
        "The number of real and fake photos does not match!"
        
    for real_path, sample_path in tqdm(zip(real_image_path_list, sample_image_path_list), total=len(real_image_path_list), disable=False):
        real_image_batch = torch.load(real_path).to(device)
        sample_image_batch = torch.load(sample_path).to(device)
        
        if real_image_batch.shape[0] < 2 or sample_image_batch.shape[0] < 2:
            raise RuntimeError("Each batch must have at least 2 images")
        
        kid.update(real_image_batch, real=True)
        kid.update(sample_image_batch, real=False)
        
    kid_mean, kid_std = kid.compute()
    kid_mean, kid_std = (kid_mean.cpu().item(), kid_std.cpu().item())
    logging.info(f"KID result: Mean {kid_mean:0.2f} - STD {kid_std:0.2f}")
    return kid_mean, kid_std    