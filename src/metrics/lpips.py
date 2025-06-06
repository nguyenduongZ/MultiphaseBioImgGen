import os
import torch
import logging
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def rescale_tensor(x):
    return 2 * (x - torch.min(x)) / (torch.max(x) - torch.min(x)) - 1

def compute_lpips(cfg, real_image_path, sample_image_path, device: torch.device, timestamp):
    if not cfg.conductor.testing.LearnedPerceptualImagePatchSimilarity.usage:
        return None
    
    params = cfg.conductor.testing.LearnedPerceptualImagePatchSimilarity.params
    lpips = LearnedPerceptualImagePatchSimilarity(**params)
    scores = []
    logging.info('LPIPS initialized')
    
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
        real_image_batch = rescale_tensor(torch.load(real_path).to(device))
        sample_image_batch = rescale_tensor(torch.load(sample_path).to(device))
        
        if real_image_batch.shape[0] < 2 or sample_image_batch.shape[0] < 2:
            raise RuntimeError("Each batch must have at least 2 images")
        
        scores.append(lpips(real_image_batch, sample_image_batch).tolist())
        lpips_mean = np.mean(scores)
        logging.info(f"LPIPS mean result: {lpips_mean:0.2f}")
        
        pd.DataFrame({
            'cond_scale': cfg.conductor['cond_scale'],
            'Dataset': cfg.data['name'],
            'Model': cfg.model['name'],
            'lpips': scores
        }).to_json(sample_image_path + timestamp + '_lpips.json')
        
        return lpips_mean