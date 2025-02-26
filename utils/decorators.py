import os, sys
import yaml

from glob import glob

def check_text_encoder(func):
    def wrapper(self, *args, **kwargs):
        config = self.config
        
        if config["model"]["Imagen"]["imagen"]["text_encoder_name"] == "google/t5-v1_1-small":
            assert config["model"]["Imagen"]["imagen"]["text_embed_dim"] == 512, "Text embed dim must be changed to 512"
            
        elif config["model"]["Imagen"]["imagen"]["text_encoder_name"] == "google/t5-v1_1-base":
            assert config["model"]["Imagen"]["imagen"]["text_embed_dim"] == 768, "Text embed dim must be changed to 768"
            
        elif config["model"]["Imagen"]["imagen"]["text_encoder_name"] == "google/t5-v1_1-large":
            assert config["model"]["Imagen"]["imagen"]["text_embed_dim"] == 1024, "Text embed dim must be changed to 1024"
        
        else:
            raise ValueError("Choose an existing encoder: 'google/t5-v1_1-small', 'google/t5-v1_1-base', 'google/t5-v1_1-large'")
            
        return func(self, *args, **kwargs)
        
    return wrapper

def check_dataset_name(func):
    def wrapper(self, *args, **kwargs):
        assert self.config["data"]["ds"] == "VinDrMultiphase", "Choose an existing dataset: VinDrMultiphase"

        return func(self, *args, **kwargs)
    
    return wrapper

def model_starter(func):
    def wrapper(self, *args, **kwargs):
        config = self.config
        
        if config["trainer"]["use_existing_model"] or glob(config["trainer"]["PATH_MODEL_CHECKPOINT"]):
            print("Load Imagen model from Checkpoint for further Training")
            
        else:
            print("Create new Imagen Model")
        
        return func(self, *args, **kwargs)
    
    return wrapper