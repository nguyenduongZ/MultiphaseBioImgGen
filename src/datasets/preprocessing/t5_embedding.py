# Adapted from: https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/t5.py
# Author: Phil Wang (@lucidrains)
# License: MIT

import os, sys
sys.path.append(os.path.abspath(os.curdir))

import torch
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from imagen_pytorch.t5 import t5_encode_tokenized_text, get_model_and_tokenizer, DEFAULT_T5_NAME

DEFAULT_BATCH_SIZE = 64

def get_text_t5_embedding(
    cfg, 
    dataset_name: str, 
    split: str, 
    df: pd.DataFrame,
    full_df: pd.DataFrame,
    batch_size: str,
    logger: logging.Logger = None, 
):
    """
    Generate T5 text embeddings for prompts in the provided DataFrame.

    Args:
        cfg: Configuration object containing dataset paths, model parameters, etc.
        dataset_name (str): Name of the dataset (used for logging/debugging purposes).
        split (str): Data split identifier ('train', 'valid', or 'test').
        df (pd.DataFrame): Subset of the dataset containing the prompts to embed.
        full_df (pd.DataFrame): The complete dataset used to calculate the optimal max token length.
        logger (logging.Logger): Logger for logging the embedding process.
        batch_size (int, optional): Number of prompts to process per batch. Defaults to DEFAULT_BATCH_SIZE.

    Returns:
        List[torch.Tensor]: List of T5 embeddings ordered to match the prompts in `df["prompt"]`.
    """
    path_t5_embedding_file = cfg.datasets["t5_embeddings"][split]
    if os.path.exists(path_t5_embedding_file):
        data = torch.load(path_t5_embedding_file, map_location="cpu", weights_only=False)
        logger.info(f"Loaded existing embeddings from {path_t5_embedding_file}")
        return data["text_embeddings"]
    
    os.makedirs(os.path.dirname(path_t5_embedding_file), exist_ok=True)

    # Encode unique prompts
    texts_unique_list = df["prompt"].unique().tolist()

    _, tokenizer = get_model_and_tokenizer(DEFAULT_T5_NAME)
    all_texts_unique_list = full_df["prompt"].unique().tolist()
    token_lens = [len(tokenizer.encode(text, truncation=True)) for text in all_texts_unique_list]
    
    optimal_length = int(np.percentile(token_lens, 95))
    while (optimal_length * cfg.models["text_embed_dim"]) % 64 != 0:
        optimal_length += 1
        
    logger.info(f"Computed optimal MAX_LENGTH for T5 embeddings ({dataset_name}-{split}): {optimal_length}")

    prompt_to_embedding = {}
    
    for i in tqdm(range(0, len(texts_unique_list), batch_size), desc=f"Embedding {dataset_name}-{split} with T5"):
        batch_texts = texts_unique_list[i:i + batch_size]

        encoded = tokenizer.batch_encode_plus(
            batch_texts,
            return_tensors="pt",
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
            
    text_embeddings = [prompt_to_embedding[prompt] for prompt in df["prompt"]]
                
    torch.save({"text_embeddings": text_embeddings}, path_t5_embedding_file)
    logger.info(f"Saved T5 embeddings to {path_t5_embedding_file}")
    
    return text_embeddings