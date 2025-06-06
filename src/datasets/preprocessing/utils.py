import numpy as np
import pandas as pd

from .t5 import text_t5_embedding

DEFAULT_BATCH_SIZE = 128

def convert_pixel_to_hu(dcm):
    if not hasattr(dcm, 'pixel_array'):
        raise ValueError('DICOM file does not contain pixel data.')
    pixel_array = dcm.pixel_array.astype(np.float32)
    slope = getattr(dcm, 'RescaleSlope', 1)
    intercept = getattr(dcm, 'RescaleIntercept', 0)
    return slope * pixel_array + intercept

def apply_window(img, window_center, window_width):
    if window_center is None or window_width is None:
        raise ValueError('Missing window_center or window_width parameters.')
    window_width = max(window_width, 1.0)
    min_val = window_center - (window_width / 2)
    max_val = window_center + (window_width / 2)
    img = np.clip(img, min_val, max_val)
    img = ((img - min_val) / (max_val - min_val))
    return img.astype(np.float32)

def get_text_embeddings(
    cfg,
    split: str,
    df: pd.DataFrame,
    full_df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE
):
    if cfg.model.name == 'imagen':
        embedding_name = cfg.model.imagen.text_encoder_name
    elif cfg.model.name == 'elucidated_imagen':
        embedding_name = cfg.model.elucidated_imagen.text_encoder_name
        
    embedding_mapping = {
        't5': text_t5_embedding,
        # Add here
    }
    
    embedding_func = None
    for key in embedding_mapping:
        if key in embedding_name.lower():
            embedding_func = embedding_mapping[key]
            break

    if embedding_func is None:
        raise ValueError(f"Unsupported embedding model: {embedding_name}")
    
    return embedding_func(
        cfg=cfg,
        split=split,
        df=df,
        full_df=full_df,
        batch_size=batch_size
    )