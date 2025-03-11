import os, sys
import torch

from glob import glob
from accelerate import Accelerator
from imagen_pytorch import Unet, ImagenTrainer, ImagenConfig, ElucidatedImagenConfig, load_imagen_from_checkpoint

from utils import Opt, check_text_encoder, model_starter

class ImagenModel():
    def __init__(self, opt: Opt, device: torch.device, testing=False):
        self.opt = opt
        self.testing = testing
        
        self.is_elucidated = (opt.conductor["model"]["model_type"] == "ElucidatedImagen")
            
        self.accelerator = Accelerator(mixed_precision="fp16") if opt.conductor["trainer"]["multi_gpu"] else None
        self.device = self.accelerator.device if self.accelerator else device
        
        # Initialize Unet1 & Unet2
        self.unet1, self.unet2 = self._set_unets_()
        
        # Initialize Imagen Model
        self._set_imagen_()
        
        # Initialize Imagen Trainer
        self._set_trainer_()
        
    def _set_unets_(self):
        if self.is_elucidated:
            unet1 = Unet(**self.opt.elucidated_imagen["unet1"])
            unet2 = Unet(**self.opt.elucidated_imagen["unet2"])
            
        else:
            unet1 = Unet(**self.opt.imagen["unet1"])
            unet2 = Unet(**self.opt.imagen["unet2"])           
        
        return unet1, unet2
    
    def _load_checkpoint(self):
        opt = self.opt
        path_model_load = opt.conductor["trainer"]["PATH_MODEL_LOAD"]

        if self.testing:
            test_model_save = opt.conductor["testing"]["PATH_MODEL_TESTING"]
            return load_imagen_from_checkpoint(test_model_save)        
        
        elif (opt.conductor["trainer"]["use_existing_model"] and glob(path_model_load)):
            return load_imagen_from_checkpoint(path_model_load)
            
        return None
    
    @check_text_encoder
    @model_starter
    def _set_imagen_(self):
        imagen_checkpoint = self._load_checkpoint()
        
        if imagen_checkpoint:
            self.imagen_model = imagen_checkpoint
            
        else:
            config = self.opt.elucidated_imagen if self.is_elucidated else self.opt.imagen
            imagen_class = ElucidatedImagenConfig if self.is_elucidated else ImagenConfig

            self.imagen_model = imagen_class(
                unets=[
                    dict(**config["unet1"]), dict(**config["unet2"])
                ],
                **config["elucidated_imagen"] if self.is_elucidated else config["imagen"]
            ).create()
            
        self.imagen_model = self.imagen_model.to(self.device)
            
    def _set_trainer_(self):
        opt = self.opt
        
        self.trainer = ImagenTrainer(
            imagen=self.imagen_model,
            split_valid_from_train=opt.conductor["trainer"]["split_valid_from_train"],
            dl_tuple_output_keywords_names=opt.conductor["trainer"]["dl_tuple_output_keywords_names"]
        )

        if self.accelerator:
            self.trainer, self.imagen_model = self.accelerator.prepare(self.trainer, self.imagen_model)
        
        else:    
            self.trainer = self.trainer.to(self.device)