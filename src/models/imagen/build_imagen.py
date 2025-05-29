import os
import torch
import logging

from copy import deepcopy
from omegaconf import OmegaConf
from pydantic import ValidationError
from imagen_pytorch import Unet, ImagenTrainer, load_imagen_from_checkpoint

from src.models.imagen.config import ImagenConfig, ElucidatedImagenConfig

def get_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb

class ImagenBuilder:
    def __init__(self, cfg, device: torch.device = None):
        super().__init__()
        self.cfg = cfg
        self.model_name = self.cfg.model.name
        self.device = device
        
        self.unet1, self.unet2 = self.build_unets()
        self.build_imagen()
        self.build_trainer()
        
    def build_unets(self):
        if self.model_name == 'imagen':
            self.unet1_cfg = OmegaConf.to_container(self.cfg.model.imagen.unets[0], resolve=True)
            self.unet2_cfg = OmegaConf.to_container(self.cfg.model.imagen.unets[1], resolve=True)
        elif self.model_name == 'elucidated_imagen':
            self.unet1_cfg = OmegaConf.to_container(self.cfg.model.elucidated_imagen.unets[0], resolve=True)
            self.unet2_cfg = OmegaConf.to_container(self.cfg.model.elucidated_imagen.unets[1], resolve=True)
        return Unet(**self.unet1_cfg), Unet(**self.unet2_cfg)
        
    def build_imagen(self):
        unet = self.cfg.conductor.unet
        if self.model_name == 'imagen':
            cond_drop_prob = self.cfg.model.imagen.cond_drop_prob
            if unet == 1:
                dim = self.cfg.model.imagen.unets[0].dim
            elif unet == 2:
                dim = self.cfg.model.imagen.unets[1].dim
        elif self.model_name == 'elucidated_imagen':
            cond_drop_prob = self.cfg.model.elucidated_imagen.cond_drop_prob
            if unet == 1:
                dim = self.cfg.model.elucidated_imagen.unets[0].dim
            elif unet == 2:
                dim = self.cfg.model.elucidated_imagen.unets[1].dim
        checkpoint_path = os.path.join(f".results/checkpoints/dim{dim}/cond_drop_prob{cond_drop_prob}", f"checkpoint-final.pt")

        if os.path.exists(checkpoint_path):
            if self.cfg.conductor.mode == 'testing':
                self.imagen_model = load_imagen_from_checkpoint(checkpoint_path, load_ema_if_available=True)
            elif self.cfg.conductor.mode == 'training':
                self.imagen_model = load_imagen_from_checkpoint(checkpoint_path)
            logging.info(f"Load model from checkpoint: {checkpoint_path}")
        else:
            try:
                if self.model_name == 'imagen':
                    imagen_cfg = deepcopy(OmegaConf.to_container(self.cfg.model.imagen, resolve=True))
                    imagen_cfg.pop('unets', None)
                    self.imagen_model = ImagenConfig(
                        unets=[dict(**self.unet1_cfg), dict(**self.unet2_cfg)],
                        **imagen_cfg
                    ).create()
                elif self.model_name == 'elucidated_imagen':
                    elucidated_imagen_cfg = deepcopy(OmegaConf.to_container(self.cfg.model.elucidated_imagen, resolve=True))
                    elucidated_imagen_cfg.pop('unets', None)
                    self.imagen_model = ElucidatedImagenConfig(
                        unets=[dict(**self.unet1_cfg), dict(**self.unet2_cfg)],
                        **elucidated_imagen_cfg
                    ).create()
            except ValidationError as e:
                logging.error(f"Wrong format: {e}")
                raise e
            
            logging.info(f"Create Imagen Model from config")
            
        if torch.cuda.is_available():
            self.imagen_model = self.imagen_model.to(device=self.device)

        logging.info(f"Model Size: {(get_model_size(self.imagen_model) / 1024):.2f} GB")
        
    def build_trainer(self):
        self.trainer = ImagenTrainer(
            imagen=self.imagen_model,
            split_valid_from_train=self.cfg.conductor.trainer.split_valid_from_train,
            dl_tuple_output_keywords_names=self.cfg.conductor.trainer.dl_tuple_output_keywords_names
        )

        if torch.cuda.is_available():
            self.trainer = self.trainer.to(device=self.device)