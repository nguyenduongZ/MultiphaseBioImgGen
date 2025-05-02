import os, sys
sys.path.append(os.path.abspath(os.curdir))

import glob
import torch
import logging

from omegaconf import OmegaConf, ListConfig
from imagen_pytorch import Unet, ImagenTrainer, ImagenConfig, ElucidatedImagenConfig, load_imagen_from_checkpoint

from src.models.imagen.decorators import check_text_encoder, model_starter

class ImagenBuilder:
    def __init__(
        self, 
        cfg, 
        device: torch.device,
        logger: logging.Logger, 
        testing=False
    ):
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.is_elucidated = cfg.models["is_elucidated"] # False | True
        self.testing = testing
        
        self.unet1, self.unet2 = self.build_unets()
        self.build_imagen()
        self.build_trainer()
        
    def _to_tuple_fields(self, cfg_dict):
        for key in ["dim_mults", "layer_attns", "layer_cross_attns", "num_resnet_blocks"]:
            if key in cfg_dict and isinstance(cfg_dict[key], (list, ListConfig)):
                cfg_dict[key] = tuple(cfg_dict[key])
        
        return cfg_dict        
    
    def build_unets(self):
        self.unet1_cfg = self._to_tuple_fields(OmegaConf.to_container(self.cfg.models["unet1"], resolve=True))
        self.unet2_cfg = self._to_tuple_fields(OmegaConf.to_container(self.cfg.models["unet2"], resolve=True))
        
        unet1 = Unet(**self.unet1_cfg)
        unet2 = Unet(**self.unet2_cfg)
            
        return unet1, unet2
    
    @model_starter
    @check_text_encoder
    def build_imagen(self):
        path_model_load = os.path.join("./", self.cfg.conductor["trainer"]["PATH_MODEL_LOAD"])
        
        if self.testing:
            test_model_save = os.path.join("./", self.cfg.conductor["testing"]["PATH_MODEL_TESTING"])
            self.imagen_model = load_imagen_from_checkpoint(test_model_save, load_ema_if_available=True)
        elif glob.glob(path_model_load):
            self.imagen_model = load_imagen_from_checkpoint(path_model_load, load_ema_if_available=True)
        else:
            if self.is_elucidated:
                self.imagen_model = ElucidatedImagenConfig(
                    unets=[dict(**self.unet1_cfg), dict(**self.unet2_cfg)],
                    **self.cfg.models["elucidated_imagen"]
                ).create()
            else:
                self.imagen_model = ImagenConfig(
                    unets=[dict(**self.unet1_cfg), dict(**self.unet2_cfg)],
                    **self.cfg.models["imagen"]
                ).create()

        if torch.cuda.is_available():
            self.imagen_model = self.imagen_model.to(self.device)

        def get_model_size(model):
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            size_all_mb = (param_size + buffer_size) / 1024**2
            
            return size_all_mb
        
        size_mb = get_model_size(self.imagen_model)
        self.logger.info(f"Model size: {size_mb:.2f} MB")
            
    def build_trainer(self):
        self.trainer = ImagenTrainer(
            imagen=self.imagen_model,
            split_valid_from_train=self.cfg.conductor["trainer"]["split_valid_from_train"],
            dl_tuple_output_keywords_names=self.cfg.conductor["trainer"]["dl_tuple_output_keywords_names"],
        )
        
        if torch.cuda.is_available():
            self.trainer = self.trainer.to(self.device)