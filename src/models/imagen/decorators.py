import os, sys
sys.path.append(os.path.abspath(os.curdir))

import glob

def check_text_encoder(func):
    def wrapper(self, *args, **kwargs):
        model_cfg = self.cfg.models
        
        if model_cfg["text_encoder_name"] == "google/t5-v1_1-small":
            assert model_cfg["text_embed_dim"] == 512, "Text embed dim must be changed to 512"
        elif model_cfg["text_encoder_name"] == "google/t5-v1_1-base":
            assert model_cfg["text_embed_dim"] == 768, "Text embed dim must be changed to 768"
        elif model_cfg["text_encoder_name"] == "google/t5-v1_1-large":
            assert model_cfg["text_embed_dim"] == 1024, "Text embed dim must be changed to 1024"
        else:
            assert True, "Choose an existing encoder: 'google/t5-v1_1-small' or 'google/t5-v1_1-base' or 'google/t5-v1_1-large'"
        
        return func(self, *args, **kwargs)
    
    return wrapper

def model_starter(func):
    def wrapper(self, *args, **kwargs):
        trainer_cfg = self.cfg.conductor["trainer"]
        
        if glob.glob(trainer_cfg["PATH_MODEL_CHECKPOINT"]):
            self.logger.info("Load Imagen model from Checkpoint for further Training")
        else:
            self.logger.info("Create new Imagen model")
            if not trainer_cfg["PATH_MODEL_LOAD"]:
                self.logger.error("No model path defined for loading!")
                raise FileNotFoundError("Model path not found for loading.")
            
        return func(self, *args, **kwargs)
    
    return wrapper