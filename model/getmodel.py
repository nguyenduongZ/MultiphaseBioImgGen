from .build_imagen import ImagenModel

from utils import Opt

def get_imagen_model(opt: Opt, device, testing=False):
    opt.config["model"]["model_type"] = "Imagen"
    
    return ImagenModel(opt=opt, device=device, testing=testing)

def get_elucidated_imagen_model(opt: Opt, device, testing=False):
    opt.config["model"]["model_type"] = "ElucidatedImagen"
    
    return ImagenModel(opt=opt, device=device, testing=testing)

def get_model(opt: Opt, device, testing=False):
    model_type = opt.config["model"]["model_type"]
    
    model_mapping = {
        "Imagen": get_imagen_model,  
        "ElucidatedImagen": get_elucidated_imagen_model
    }
    
    if model_type not in model_mapping:
        raise ValueError(f"Model {model_type} is not supported. Choose 'Imagen' or 'ElucidatedImagen'.")

    return model_mapping[model_type](opt, device, testing)