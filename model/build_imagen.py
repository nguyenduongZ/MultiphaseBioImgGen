import os, sys
import torch

from glob import glob
from imagen_pytorch import Unet, ImagenTrainer, ImagenConfig, ElucidatedImagenConfig, load_imagen_from_checkpoint
from utils import check_text_encoder, model_starter

class ImagenModel():
    def __init__(self, config, device: torch.device, testing=False):
        model_type = config["model"]["model_type"]
        assert model_type == "Imagen" or model_type == "ElucidatedImagen"
        
        if model_type == "Imagen":
            self.is_elucidated = False
            
        elif model_type == "ElucidatedImagen":
            self.is_elucidated = True
        
        self.config = config
        self.device = device    
        self.testing = testing
        
        self.unet1, self.unet2 = self._set_unets_(elucidated=self.is_elucidated)
        self._set_imagen_()
        self._set_trainer_()
    
    def _set_unets_(self, elucidated: bool):
        self.model_key = "ElucidatedImagen" if elucidated else "Imagen"

        self.unet1 = Unet(**self.config["model"][self.model_key]["unets"]["unet1"])
        self.unet2 = Unet(**self.config["model"][self.model_key]["unets"]["unet2"])
        
        return self.unet1, self.unet2
    
    @model_starter
    @check_text_encoder
    def _set_imagen_(self):
        path_model_load = self.config["trainer"]["PATH_MODEL_LOAD"]
        
        if self.testing:
            test_model_save = self.config["testing"]["PATH_MODEL_TESTING"]
            self.imagen_model = load_imagen_from_checkpoint(test_model_save)
        
        elif (self.config["trainer"]["use_existing_model"] and len(glob(path_model_load)) > 0):
            self.imagen_model = load_imagen_from_checkpoint(path_model_load)
            
        else:
            if self.is_elucidated:
                self.imagen_model = ElucidatedImagenConfig(
                    unets=[
                        dict(**self.config["model"][self.model_key]["unets"]["unet1"]), 
                        dict(**self.config["model"][self.model_key]["unets"]["unet2"])],
                    **self.config["model"][self.model_key]["elucidated_imagen"]
                ).create()
                
            else:
                self.imagen_model = ImagenConfig(
                    unets=[
                        dict(**self.config["model"][self.model_key]["unets"]["unet1"]), 
                        dict(**self.config["model"][self.model_key]["unets"]["unet2"])],
                    **self.config["model"][self.model_key]["imagen"]
                ).create()
            
        if not self.config["trainer"]["multi_gpu"]:
            self.imagen_model = self.imagen_model.to(self.device)
            
    def _set_trainer_(self):
        self.trainer = ImagenTrainer(
            imagen=self.imagen_model,
            split_valid_from_train=self.config["trainer"]["split_valid_from_train"],
            dl_tuple_output_keywords_names=self.config["trainer"]["dl_tuple_output_keywords_names"]
        )
        
        if not self.config["trainer"]["multi_gpu"]:
            self.trainer = self.trainer.to(self.device)