import os
import re
import torch
import logging

from glob import glob
from copy import deepcopy
from torchinfo import summary
from omegaconf import OmegaConf
from imagen_pytorch import Unet, ImagenTrainer, ImagenConfig, ElucidatedImagenConfig, load_imagen_from_checkpoint

from src.utils import folder_setup

def get_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb

def print_model_summary(model, input_shape, model_name):
    try:
        logging.info(f"\n=== {model_name} Summary ===")
        model_summary = summary(model, input_size=input_shape, verbose=0)
        logging.info(f"\n{model_summary}")
    except Exception as e:
        logging.warning(f"Could not generate summary for {model_name}: {e}")

class ImagenBuilder:
    def __init__(self, cfg, device: torch.device = None):
        super().__init__()
        self.cfg = cfg
        self.model_name = self.cfg.model.name
        self.mode = self.cfg.conductor.mode
        self.device = device
        
        self.unet1, self.unet2 = self.build_unets()
        self.build_imagen()
        self.build_trainer()
        
    def build_unets(self):
        model = self.cfg.model[self.model_name]
        self.unet1_cfg = OmegaConf.to_container(model['unets'][0], resolve=True)
        self.unet2_cfg = OmegaConf.to_container(model['unets'][1], resolve=True)
        
        unet1 = Unet(**self.unet1_cfg)
        unet2 = Unet(**self.unet2_cfg)
        
        return unet1, unet2

    def build_imagen(self):
        unet = self.cfg.conductor.unet
        run_dir, _ = folder_setup(self.cfg)

        if self.mode == 'training':
            if unet == 1:
                ckpt_pattern = os.path.join(run_dir, "checkpoints/checkpoint-iter*.pt")
                checkpoint_files = glob(ckpt_pattern)
                if checkpoint_files:
                    def extract_iter(path):
                        match = re.search(r'checkpoint-iter(\d+)\.pt', path)
                        return int(match.group(1)) if match else -1
                    checkpoint_files.sort(key=extract_iter)
                    checkpoint_path = checkpoint_files[-1]
                else:
                    checkpoint_path = None

            elif unet == 2:
                ckpt_pattern = os.path.join(run_dir, "checkpoints/checkpoint-iter*.pt")
                checkpoint_files = glob(ckpt_pattern)
                if checkpoint_files:
                    checkpoint_files.sort(key=lambda p: int(re.search(r'iter(\d+)', p).group(1)))
                    checkpoint_path = checkpoint_files[-1]
                else:
                    final_ckpt_path = os.path.join(run_dir, "checkpoints/checkpoint-final.pt")
                    checkpoint_path = final_ckpt_path if os.path.exists(final_ckpt_path) else None

        elif self.mode == 'testing':
            checkpoint_path = os.path.join(run_dir, "checkpoints/checkpoint-final.pt")
            
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
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
                unets=[
                    dict(**self.unet1_cfg), 
                    dict(**self.unet2_cfg)
                ],
                **imagen_cfg
            ).create()
            logging.info(f"Create Imagen Model from config")
            
        if torch.cuda.is_available():
            self.imagen_model = self.imagen_model.to(self.device)
        
        logging.info(f"Model Size: {(get_model_size(self.imagen_model) / 1024):.2f} GB")
        
        print_model_summary(self.unet1, input_shape=(1, 3, 64, 64), model_name="Unet1")
        print_model_summary(self.unet2, input_shape=(1, 3, 256, 256), model_name="Unet2")
        
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