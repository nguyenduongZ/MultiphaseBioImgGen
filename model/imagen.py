import os, sys
sys.path.append(os.path.abspath(os.curdir))
import torch

from glob import glob
from utils.built_opt import Opt
from utils.decorators import check_text_encoder, model_starter
from imagen_pytorch import Unet, ImagenTrainer, ImagenConfig, ElucidatedImagenConfig, load_imagen_from_checkpoint

class ImagenModel():
    def __init__(self, opt: Opt, device: torch.device, testing=False):
        assert opt.conductor["model"]["model_type"] == "Imagen" or opt.conductor["model"]["model_type"] == "ElucidatedImagen"
        
        if opt.conductor["model"]["model_type"] == "ElucidatedImagen":
            self.is_elucidated = True
        
        elif opt.conductor["model"]["model_type"] == "Imagen":
            self.is_elucidated = False
           
        self.testing = testing
        self.unet_number = opt.conductor["trainer"]["unet_number"]
        
        self.unet1, self.unet2 = self._get_unets_(opt=opt, is_elucidated=self.is_elucidated)
        self._set_imagen_(opt=opt, device=device)
        self._set_trainer_(opt=opt, device=device)
        
    def _get_unets_(self, opt: Opt, is_elucidated: bool):
        if is_elucidated:
            unet1 = Unet(**opt.elucidated_imagen["unet1"])
            unet2 = Unet(**opt.elucidated_imagen["unet2"])
            
        else: 
            unet1 = Unet(**opt.imagen["unet1"])
            unet2 = Unet(**opt.imagen["unet2"])
            
        return unet1, unet2
    
    @model_starter
    @check_text_encoder
    def _set_imagen_(self, opt: Opt, device: torch.device):
        if self.unet_number == 1:
            path_model_load = opt.conductor["trainer"]["PATH_MODEL_LOAD1"]
        
        elif self.unet_number == 2:
            path_model_load = opt.conductor["trainer"]["PATH_MODEL_LOAD2"]
            
        if self.testing:
            test_model_load = opt.conductor["testing"]["PATH_MODEL_TESTING"]
            
            self.imagen_model = load_imagen_from_checkpoint(test_model_load)
            
        elif (opt.conductor["trainer"]["use_existing_model"] and glob(path_model_load)):
            self.imagen_model = load_imagen_from_checkpoint(path_model_load)
            
        else:
            if self.is_elucidated:
                self.imagen_model = ElucidatedImagenConfig(
                    unets=[
                        dict(**opt.elucidated_imagen["unet1"]),
                        dict(**opt.elucidated_imagen["unet2"])
                    ],
                    **opt.elucidated_imagen["elucidated_imagen"]
                ).create()
                
            else:
                self.imagen_model = ImagenConfig(
                    unets=[
                        dict(**opt.imagen["unet1"]),
                        dict(**opt.imagen["unet2"])
                    ],
                    **opt.imagen["imagen"]                    
                ).create()
                
        if torch.cuda.is_available() and not opt.conductor["trainer"]["multi_gpu"]:
            self.imagen_model = self.imagen_model.to(device)
            
    def _set_trainer_(self, opt: Opt, device: torch.device):
        self.trainer = ImagenTrainer(
            imagen=self.imagen_model,
            split_valid_from_train=opt.conductor["trainer"]["split_valid_from_train"],
            dl_tuple_output_keywords_names=opt.conductor["trainer"]["dl_tuple_output_keywords_names"]
        )
        
        if torch.cuda.is_available() and not opt.conductor["trainer"]["multi_gpu"]:
            self.trainer = self.trainer.to(device)