import pandas as pd

from .t5 import text_t5_embedding

DEFAULT_BATCH_SIZE = 128

def get_text_embeddings(
    cfg,
    split: str,
    df: pd.DataFrame,
    full_df: pd.DataFrame,
    batch_size: int = None
):
    if cfg.model.name == 'imagen':
        embedding_name = cfg.model.imagen.text_encoder_name
    elif cfg.model.name == 'elucidated_imagen':
        embedding_name = cfg.model.elucidated_imagen.text_encoder_name
        
    embedding_mapping = {
        't5': text_t5_embedding
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