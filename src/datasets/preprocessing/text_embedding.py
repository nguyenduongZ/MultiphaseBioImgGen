import os, sys
sys.path.append(os.path.abspath(os.curdir))

import logging
import pandas as pd

from .t5_embedding import get_text_t5_embedding
from .clip_embedding import get_text_clip_embedding
from .biobert_embedding import get_text_biobert_embedding

DEFAULT_BATCH_SIZE = 64

def get_text_embedding(
    cfg, 
    dataset_name: str,
    split: str, 
    df: pd.DataFrame,
    full_df: pd.DataFrame,
    logger: logging.Logger,
    batch_size=DEFAULT_BATCH_SIZE
):
    text_encoder_name = cfg.datasets["text_encoder_name"]
    text_encoder_mapping = {
        "t5": get_text_t5_embedding,
        "biobert": get_text_biobert_embedding,
        "clip": get_text_clip_embedding,
    }
    
    return text_encoder_mapping[text_encoder_name](
        cfg=cfg, 
        dataset_name=dataset_name, 
        split=split, df=df, 
        full_df=full_df, 
        logger=logger, 
        batch_size=batch_size
    )
