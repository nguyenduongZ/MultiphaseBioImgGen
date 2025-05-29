# Adapted from: https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/t5.py
# Author: Phil Wang (@lucidrains)
# License: MIT

import os
import torch
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from imagen_pytorch.t5 import *

def text_t5_embedding(
    cfg,
    split: str,
    df: pd.DataFrame,
    full_df: pd.DataFrame,
    batch_size: int = None
):
    dataset_name = cfg.data.name
    path_t5_embedding_file = cfg.data.t5_embeddings[split]
    if os.path.exists(path_t5_embedding_file):
        data = torch.load(path_t5_embedding_file, map_location='cpu', weights_only=False)
        logging.info(f"Loaded existing T5 embeddings from {path_t5_embedding_file}")
        return data['text_embeddings']
    
    os.makedirs(os.path.dirname(path_t5_embedding_file), exist_ok=True)
    
    # Encode unique prompts
    text_unique_list = df['prompt'].unique().tolist()
    all_texts_unique_list = full_df["prompt"].unique().tolist()
    
    if cfg.model.name == 'imagen':
        DEFAULT_T5_NAME = cfg.model.imagen.text_encoder_name
    elif cfg.model.name == 'elucidated_imagen':
        DEFAULT_T5_NAME = cfg.model.elucidated_imagen.text_encoder_name
    model, tokenizer = get_model_and_tokenizer(DEFAULT_T5_NAME)
    token_lens = [len(tokenizer.encode(text, truncation=True)) for text in all_texts_unique_list]
    
    optimal_length = int(np.percentile(token_lens, 95))
    
    embedding_dim = model.config.d_model
    while (optimal_length * embedding_dim) % 64 != 0:
        optimal_length += 1
    logging.info(f"Computed optimal MAX_LENGTH for T5 embeddings ({dataset_name}-{split}): {optimal_length}")
        
    prompt_to_embedding = {}
        
    for i in tqdm(range(0, len(text_unique_list), batch_size), desc=f"Embedding {dataset_name}-{split} with T5"):
        batch_texts = text_unique_list[i:i+batch_size]
        
        encoded = tokenizer.batch_encode_plus(
            batch_texts,
            return_tensors='pt',
            padding='max_length',
            max_length=optimal_length,
            truncation=True
        )
        input_ids = encoded.input_ids.cpu()
        attn_mask = encoded.attention_mask.cpu()
        
        batch_embeddings = t5_encode_tokenized_text(
            token_ids=input_ids,
            attn_mask=attn_mask,
            name=DEFAULT_T5_NAME
        ).cpu()

        for text, emb in zip(batch_texts, batch_embeddings):
            prompt_to_embedding[text] = emb
            
    text_embeddings = [prompt_to_embedding[prompt] for prompt in df['prompt']]
        
    torch.save({'text_embeddings': text_embeddings}, path_t5_embedding_file)
    logging.info(f"Saved T5 embeddings to {path_t5_embedding_file}")
    return text_embeddings
