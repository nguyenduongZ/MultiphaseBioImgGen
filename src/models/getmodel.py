from src.models.imagen.build_imagen import ImagenBuilder

def get_model(cfg, device):
    model_name = cfg.model.name
    model_mapping = {
        'elucidated_imagen': ImagenBuilder,
        'imagen': ImagenBuilder
    }
    
    model = model_mapping[model_name](cfg=cfg, device=device)
    return model