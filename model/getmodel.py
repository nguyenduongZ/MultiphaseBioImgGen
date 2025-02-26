from .build_imagen import ImagenModel

def get_imagen_model(config, device, testing=False):
    config["model"]["model_type"] = "Imagen" 
    
    return ImagenModel(config=config, device=device, testing=testing)

def get_elucidated_imagen_model(config, device, testing=False):
    config["model"]["model_type"] = "ElucidatedImagen" 
    
    return ImagenModel(config=config, device=device, testing=testing)

def get_model(config, model_type, device, testing=False):
    model_mapping = {
        "Imagen": get_imagen_model,  
        "ElucidatedImagen": get_elucidated_imagen_model
    }
    
    if model_type not in model_mapping:
        raise ValueError(f"Model {model_type} is not supported. Choose 'Imagen' or 'ElucidatedImagen'.")

    return model_mapping[model_type](config, device, testing)