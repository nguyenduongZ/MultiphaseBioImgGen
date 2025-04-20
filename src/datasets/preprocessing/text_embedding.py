import os, sys
sys.path.append(os.path.abspath(os.curdir))

import torch
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from imagen_pytorch.t5 import t5_encode_tokenized_text, get_model_and_tokenizer, DEFAULT_T5_NAME

DEFAULT_BATCH_SIZE = 64

def get_text_t5_embedding(
    cfg, 
    dataset_name: str, 
    split: str, 
    df: pd.DataFrame,
    full_df: pd.DataFrame,
    logger: logging.Logger, 
    batch_size=DEFAULT_BATCH_SIZE
):
    path_t5_embedding_file = cfg.datasets["t5_embeddings"][split]
    if os.path.exists(path_t5_embedding_file):
        text_embeddings = torch.load(path_t5_embedding_file, map_location="cpu", weights_only=False)
        logger.info(f"Loaded existing embeddings from {path_t5_embedding_file}")
        return text_embeddings
    
    os.makedirs(os.path.dirname(path_t5_embedding_file), exist_ok=True)
    text_embeddings_dict = {}
    texts_list = df["prompt"].tolist()
    
    unique_texts = set(texts_list)
    logger.info(f"Total prompts: {len(texts_list)}, Unique prompts: {len(unique_texts)}")
    if len(unique_texts) < len(texts_list):
        logger.warning(f"Found {len(texts_list) - len(unique_texts)} duplicate prompts")

    _, tokenizer = get_model_and_tokenizer(DEFAULT_T5_NAME)
    all_texts_unique = full_df["prompt"].tolist()
    token_lens = [len(tokenizer.encode(text, truncation=True)) for text in all_texts_unique]
    optimal_length = int(np.percentile(token_lens, 95))
    while (optimal_length * cfg.models["text_embed_dim"]) % 64 != 0:
        optimal_length += 1
    logger.info(f"Computed optimal MAX_LENGTH for T5 embeddings ({dataset_name}-{split}): {optimal_length}")

    for i in tqdm(range(0, len(texts_list), batch_size), desc=f"Embedding {dataset_name}-{split} with T5"):
        batch_texts = texts_list[i:i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(texts_list))))
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

        for j, (emb, idx) in enumerate(zip(batch_embeddings, batch_indices)):
            text_embeddings_dict[idx] = emb
                
    torch.save(text_embeddings_dict, path_t5_embedding_file)
    logger.info(f"Saved T5 embeddings to {path_t5_embedding_file}")
    
    return text_embeddings_dict

def get_text_biobert_embedding(
    cfg, 
    dataset_name: str, 
    split: str, 
    df: pd.DataFrame,
    logger: logging.Logger,
    batch_size=DEFAULT_BATCH_SIZE
):
    path_biobert_embedding_file = cfg.datasets["biobert_embeddings"][split]
    if os.path.exists(path_biobert_embedding_file):
        text_embeddings = torch.load(path_biobert_embedding_file, map_location="cpu", weights_only=False)
        logger.info(f"Loaded existing embeddings from {path_biobert_embedding_file}")
        return text_embeddings
    
    os.makedirs(os.path.dirname(path_biobert_embedding_file), exist_ok=True)
    text_embeddings_dict = {}
    texts_list = df["prompt"].tolist()

    # Load BioBERT model and tokenizer
    model_name = "dmis-lab/biobert-base-cased-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    token_lens = [len(tokenizer.encode(text, truncation=True)) for text in texts_list]
    optimal_length = int(np.percentile(token_lens, 95))
    while (optimal_length * 768) % 64 != 0:
        optimal_length += 1
    logger.info(f"Computed MAX_LENGTH for BioBERT ({dataset_name}-{split}): {optimal_length}")

    for i in tqdm(range(0, len(texts_list), batch_size), desc=f"Embedding {dataset_name}-{split} with BioBERT"):
        batch_texts = texts_list[i:i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(texts_list))))
        encoded = tokenizer.batch_encode_plus(
            batch_texts,
            return_tensors="pt",
            padding='max_length',
            max_length=optimal_length,
            truncation=True
        )
        input_ids = encoded["input_ids"]
        attn_mask = encoded["attention_mask"]

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attn_mask = attn_mask.cuda()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            embeddings = outputs.last_hidden_state[:, 0, :]  # lấy CLS token

        embeddings = embeddings.cpu()

        for j, (emb, idx) in enumerate(zip(embeddings, batch_indices)):
            text_embeddings_dict[idx] = emb

    torch.save(text_embeddings_dict, path_biobert_embedding_file)
    logger.info(f"Saved BioBERT embeddings to {path_biobert_embedding_file}")

    return text_embeddings_dict
