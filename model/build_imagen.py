import os, sys
import torch

from glob import glob
from omegaconf import OmegaConf, ListConfig
from imagen_pytorch import Unet, ImagenTrainer, ImagenConfig, ElucidatedImagenConfig, load_imagen_from_checkpoint

from utils import Opt, check_text_encoder, model_starter

class ImagenModel():
    def __init__(self, opt: Opt, device: torch.device, testing=False):
        self.config = opt.config
        self.device = device
        self.testing = testing

        self.model_type = self.config["model"]["model_type"]
        self.is_elucidated = self.model_type == "ElucidatedImagen"

        self.unet1, self.unet2 = self._set_unets_()
        self._set_imagen_()
        self._set_trainer_()

    def _convert_unet_config(self, unet_config):
        unet_config = OmegaConf.to_container(unet_config, resolve=True)

        tuple_fields = ["dim_mults", "layer_attns", "layer_cross_attns"]
        for field in tuple_fields:
            if field in unet_config and isinstance(unet_config[field], (list, ListConfig)):
                unet_config[field] = tuple(unet_config[field])

        if "num_resnet_blocks" in unet_config:
            if isinstance(unet_config["num_resnet_blocks"], (list, ListConfig)):
                unet_config["num_resnet_blocks"] = tuple(map(int, unet_config["num_resnet_blocks"]))

        return unet_config

    def _set_unets_(self):
        unet1_config = self._convert_unet_config(self.config["model"][self.model_type]["unet1"])
        unet2_config = self._convert_unet_config(self.config["model"][self.model_type]["unet2"])

        return Unet(**unet1_config), Unet(**unet2_config)

    @model_starter
    @check_text_encoder
    def _set_imagen_(self):
        path_model_load = self.config["trainer"]["PATH_MODEL_LOAD"]

        if self.testing:
            self.imagen_model = load_imagen_from_checkpoint(self.config["testing"]["PATH_MODEL_TESTING"])
            
        elif self.config["trainer"]["use_existing_model"] and glob(path_model_load):
            self.imagen_model = load_imagen_from_checkpoint(path_model_load)
            
        else:
            imagen_config_class = ElucidatedImagenConfig if self.is_elucidated else ImagenConfig
            self.imagen_model = imagen_config_class(
                unets=[
                    self._convert_unet_config(self.config["model"][self.model_type]["unet1"]),
                    self._convert_unet_config(self.config["model"][self.model_type]["unet2"])
                ],
                **self.config["model"][self.model_type].get("elucidated_imagen" if self.is_elucidated else "imagen", {})
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

def main():
    opt = Opt()
    device = torch.device("cuda")

    imagen_model = ImagenModel(opt=opt, device=device)
    opt.logger.debug(f'Imagen Model: {imagen_model.trainer}')

if __name__ == "__main__":
    main()
