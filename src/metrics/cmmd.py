import os, sys
sys.path.append(os.path.abspath(os.curdir))

import torch
import logging
import numpy as np

from glob import glob
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

class CMMDMetric:
    def __init__(
        self, 
        model_name: str = "openai/clip-vit-large-patch14-336",
        normalize: bool = True,
        subsets: int = 50,
        subset_size: int = 100,
        device: torch.device = None
    ):
        self.device = device
        self.normalize = normalize
        self.subsets = subsets
        self.subset_size = subset_size
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name).eval().to(device)
        self.input_image_size = self.image_processor.crop_size["height"]
        self.real_embeds = []
        self.fake_embeds = []
        
    def _resize_bicubic(self, images, size):
        images = images.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        images = torch.nn.functional.interpolate(
            images, size=(size, size), mode="bicubic", align_corners=False
        )
        images = images.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        return images
    
    def update(self, images, real):
        if images.shape[1] != 3:
            raise ValueError(f"Expected 3-channel images, got {images.shape[1]} channels")
        if self.normalize:
            images = images * 2.0 - 1.0  # Convert [0, 1] to [-1, 1] for CLIP
        images = self._resize_bicubic(images, self.input_image_size)
        inputs = self.image_processor(images=images, return_tensors="pt", do_rescale=False)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            embeddings = self.model(**inputs).image_embeds
        if real:
            self.real_embeds.append(embeddings)
        else:
            self.fake_embeds.append(embeddings)
            
    def compute(self):
        if not self.real_embeds or not self.fake_embeds:
            raise RuntimeError("No embeddings available to compute CMMD!")
        
        real_embeds = torch.cat(self.real_embeds, dim=0)
        fake_embeds = torch.cat(self.fake_embeds, dim=0)
        
        if real_embeds.shape[0] < self.subset_size or fake_embeds.shape[0] < self.subset_size:
            raise RuntimeError(f"Need at least {self.subset_size} samples for real and fake embeddings!")
        
        # Compute MMD with polynomial kernel
        def poly_kernel(X, Y, degree=3, gamma=None, coef0=1.0):
            if gamma is None:
                gamma = 1.0 / X.shape[1]
            K = (gamma * torch.mm(X, Y.t()) + coef0) ** degree
            return K

        mmd_values = []
        for _ in range(self.subsets):
            # Randomly sample subsets
            real_indices = torch.randperm(real_embeds.shape[0])[:self.subset_size]
            fake_indices = torch.randperm(fake_embeds.shape[0])[:self.subset_size]
            real_subset = real_embeds[real_indices]
            fake_subset = fake_embeds[fake_indices]

            XX = poly_kernel(real_subset, real_subset)
            YY = poly_kernel(fake_subset, fake_subset)
            XY = poly_kernel(real_subset, fake_subset)

            m = real_subset.shape[0]
            n = fake_subset.shape[0]

            # MMD^2 = E[k(x,x)] + E[k(y,y)] - 2*E[k(x,y)]
            mmd2 = (
                (XX.sum() - XX.diag().sum()) / (m * (m - 1)) +
                (YY.sum() - YY.diag().sum()) / (n * (n - 1)) -
                2 * XY.sum() / (m * n)
            )
            mmd_values.append(mmd2.cpu().item())

        # Reset embeddings
        self.real_embeds = []
        self.fake_embeds = []
        
        cmmd_mean = np.mean(mmd_values)
        cmmd_std = np.std(mmd_values) if len(mmd_values) > 1 else 0.0
        return cmmd_mean, cmmd_std
    
def compute_cmmd(cfg, logger: logging.Logger, real_image_save_path, sample_image_save_path, device):
    if not cfg.conductor["testing"]["CMMD"]["usage"]:
        logger.info("CMMD not enabled, skipping.")
        return None, None
    
    cmmd_params = cfg.conductor["testing"]["CMMD"]["params"]
    cmmd = CMMDMetric(**cmmd_params).to(device)
    logger.info("CMMD initialized")
    
    real_image_path_list = sorted(glob(os.path.join(real_image_save_path, "*_real_image_batch.pt")))
    sample_image_path_list = sorted(glob(os.path.join(sample_image_save_path, "*_sample_image_batch.pt")))
    
    if len(real_image_path_list) == 0 or len(sample_image_path_list) == 0:
        raise ValueError(f"Image not found! Check path:\nReal: {real_image_save_path}\nFake: {sample_image_save_path}")

    assert len(real_image_path_list) == len(sample_image_path_list), "The number of real and fake photos does not match!"

    for real_path, sample_path in tqdm(
        zip(real_image_path_list, sample_image_path_list),
        total=len(real_image_path_list),
        desc="Computing CMMD",
        disable=False
    ):
        real_image_batch = torch.load(real_path).to(device)
        sample_image_batch = torch.load(sample_path).to(device)

        if real_image_batch.shape[0] < 2 or sample_image_batch.shape[0] < 2:
            raise RuntimeError("Each batch must have at least 2 images!")

        # Ensure images are in [0, 1]
        real_image_batch = torch.clamp(real_image_batch, 0, 1)
        sample_image_batch = torch.clamp(sample_image_batch, 0, 1)

        cmmd.update(real_image_batch, real=True)
        cmmd.update(sample_image_batch, real=False)

    cmmd_mean, cmmd_std = cmmd.compute()
    logger.info(f"CMMD result: Mean {cmmd_mean:.6f} - STD {cmmd_std:.6f}")
        
    return cmmd_mean, cmmd_std
    