import os, sys
sys.path.append(os.path.abspath(os.curdir))

from glob import glob
from utils.built_opt import Opt

def check_text_encoder(func):
    opt=Opt()
    def wrapper(*args, **kwargs):
        if opt.imagen["imagen"]["text_encoder_name"] == "google/t5-v1_1-base":
            assert opt.imagen["imagen"]["text_embed_dim"] == 768, "Text embed dim must be changed to 768"
            
        else:
            assert True, "Choose an existing encoder: 'google/t5-v1_1-base' or ..."
            
        return func(*args, **kwargs)
        
    return wrapper

def check_dataset_name(func):
    opt = Opt()
    def wrapper(*args, **kwargs):
        assert opt.data["data"]["dataset"] == "VinDrMultiphase", "Choose an existing dataset: 'VinDrMultiphase'"
        
        return func(*args, **kwargs)
        
    return wrapper

def model_starter(func):
    opt=Opt()
    def wrapper(*args, **kwargs):
        if opt.conductor["trainer"]["use_existing_model"] or glob(opt.conductor["trainer"]["PATH_MODEL_CHECKPOINT"]):
            opt.logger.info('Load Imagen model from Checkpoint for further Training')
            
        else:
            opt.logger.info('Create new Imagen model')
            
        return func(*args, **kwargs)
    
    opt.logger.debug('Imagen Model built')
    
    return wrapper