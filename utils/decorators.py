import os, sys

from glob import glob

from .build_opt import Opt

def check_text_encoder(func):
    opt = Opt()
    def wrapper(*args, **kwargs):
        if opt.imagen["imagen"]["text_encoder_name"] == "google/t5-v1_1-small":
            assert opt.imagen["imagen"]["text_embed_dim"] == 512, "Text embed dim must be changed to 512"        
        
        elif opt.imagen["imagen"]["text_encoder_name"] == "google/t5-v1_1-base":
            assert opt.imagen["imagen"]["text_embed_dim"] == 768, "Text embed dim must be changed to 768"

        elif opt.imagen["imagen"]["text_encoder_name"] == "google/t5-v1_1-large":
            assert opt.imagen["imagen"]["text_embed_dim"] == 1024, "Text embed dim must be changed to 1024"
            
        else:
            assert True, "Choose an existing encoder: 'google/t5-v1_1-small' or 'google/t5-v1_1-base' or 'google/t5-v1_1-large'"
            
        return func(*args, **kwargs)
    
    return wrapper

def check_dataset_name(func):
    opt = Opt()
    def wrapper(*args, **kwargs):
        assert opt.dataset["data"]["ds"] == "VinDrMultiphase"
        
        return func(*args, **kwargs)

    return wrapper

def model_starter(func):
    opt = Opt()
    def wrapper(*args, **kwargs):
        if opt.conductor["trainer"]["use_existing_model"] and glob(opt.conductor["trainer"]["PATH_MODEL_CHECKPOINT"]):
            opt.logger.info("Load Imagen model from Checkpoint for further Training")
            
        else:
            opt.logger.info("Create new Imagen model")
            
        return func(*args, **kwargs)

    opt.logger.debug("Imagen Model built")

    return wrapper            