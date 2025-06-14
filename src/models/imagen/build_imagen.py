import os
import re
import torch
import logging

from glob import glob
from copy import deepcopy
from omegaconf import OmegaConf
from imagen_pytorch import Unet, ImagenTrainer, ImagenConfig, ElucidatedImagenConfig, load_imagen_from_checkpoint

from src.utils import folder_setup

def get_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_all_gb = (param_size + buffer_size) / (1024 ** 3)  
    return size_all_gb

class ImagenBuilder:
    def __init__(self, cfg, device: torch.device = None):
        super().__init__()
        self.cfg = cfg
        self.model_name = self.cfg.model.name
        self.mode = self.cfg.conductor.mode
        self.device = device
        
        self.unets = self.build_unets()
        self.build_imagen()
        self.build_trainer()
        
    def build_unets(self):
        model = self.cfg.model[self.model_name]
        unet_cfgs = OmegaConf.to_container(model.get('unets', []), resolve=True)
        self.unet_cfgs = unet_cfgs 
        
        unets = []
        for idx, unet_cfg in enumerate(unet_cfgs):
            try:
                unet = Unet(**unet_cfg)
                unets.append(unet)
            except Exception as e:
                logging.warning(f"Failed to build Unet {idx + 1}: {e}")
        
        return unets

    def build_imagen(self):
        checkpoint_path = self.cfg.conductor.checkpoint_path
            
        if os.path.exists(checkpoint_path):
            if self.mode == 'testing':
                self.imagen_model = load_imagen_from_checkpoint(checkpoint_path, load_ema_if_available=True)
            elif self.mode == 'training':
                self.imagen_model = load_imagen_from_checkpoint(checkpoint_path)
            logging.info(f"Load model from checkpoint: {checkpoint_path}")
        else:
            imagen_cfg = deepcopy(OmegaConf.to_container(self.cfg.model[self.model_name], resolve=True))
            imagen_cfg.pop('unets', None)
            
            imagen_config_klass = ElucidatedImagenConfig if self.model_name == 'elucidated_imagen' else ImagenConfig
            
            self.imagen_model = imagen_config_klass(
                unets=[dict(**cfg) for cfg in self.unet_cfgs],
                **imagen_cfg
            ).create()
            logging.info(f"Create Imagen Model from config")
            
        if torch.cuda.is_available():
            self.imagen_model = self.imagen_model.to(self.device)
        
        logging.info(f"Model Size: {(get_model_size(self.imagen_model)):.2f} GB")
        
    def build_trainer(self):
        if self.cfg.data.t5_embeddings.usage:
            dl_tuple_output_keywords_names = ('images', 'text_embeds')
        else:
            dl_tuple_output_keywords_names = ('images', 'text_embeds', 'text_masks')
        
        self.trainer = ImagenTrainer(
            imagen=self.imagen_model,
            split_valid_from_train=self.cfg.conductor.trainer.split_valid_from_train,
            dl_tuple_output_keywords_names=dl_tuple_output_keywords_names
        )

        if torch.cuda.is_available():
            self.trainer = self.trainer.to(device=self.device)