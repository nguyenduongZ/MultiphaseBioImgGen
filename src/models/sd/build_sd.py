import os, sys
sys.path.append(os.path.abspath(os.curdir))

import torch
import logging
import torch.nn as nn

from torch.optim import AdamW
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler

class StableDiffusionBuilder(nn.Module):
    def __init__(
        self,
        cfg,
        device: torch.device,
        logger: logging.Logger,
        testing: bool = False
    ):
        super().__init__()
        model_id = cfg.models["pretrained_model_id"]
        
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder=cfg.models["unet"]["subfolder"]).to(device)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder=cfg.models["vae"]["subfolder"]).to(device)
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder=cfg.models["scheduler"]["subfolder"])
        
        self.device = device
        self.logger = logger
        self.cond_scale = cfg.conductor["cond_scale"]
        self.lr = cfg.conductor["lr"]
        
        self.optimizer = AdamW(self.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        
        if testing:
            checkpoint_path = cfg.conductor["testing"]["PATH_MODEL_TESTING"]
            if checkpoint_path and os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                self.load_state_dict(state_dict, strict=False)
                logger.info(f"StableDiffusion Loaded checkpoint from {checkpoint_path}")
            else:
                logger.warning(f"StableDiffusion Checkpoint not found at {checkpoint_path}")
    
    def train_step(self, batch):
        self.train()
        self.optimizer.zero_grad()
        
        images, text_embeddings = batch
        images = images.to(self.device)
        text_embeddings = text_embeddings.to(self.device)

        # Encode image to latent space
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215  # Scale per SD spec

        # Sample timestep and noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device).long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Predict noise
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample

        # Compute loss
        loss = self.loss_fn(noise_pred, noise)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def valid_step(self, batch):
        self.eval()
        with torch.no_grad():
            images, text_embeddings, _ = batch
            images = images.to(self.device)
            text_embeddings = text_embeddings.to(self.device)

            # Encode image to latent space
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215

            # Sample timestep and noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device).long()
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample

            # Compute loss
            loss = self.loss_fn(noise_pred, noise)

        return loss.item()
    
    def sample(self, text_embeds, num_inference_steps=50, return_pil_images=False):
        self.eval()
        bsz = text_embeds.shape[0]
        latent_shape = (bsz, 4, 64, 64)

        latents = torch.randn(latent_shape).to(self.device)
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps:
            noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=torch.zeros_like(text_embeds)).sample
            noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeds).sample
            noise_pred = noise_pred_uncond + self.cond_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        images = self.vae.decode(latents / 0.18215).sample
        images = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)

        if return_pil_images:
            from torchvision.transforms.functional import to_pil_image
            return [to_pil_image(img.cpu()) for img in images]
        return images