import os, sys
sys.path.append(os.path.abspath(os.curdir))

import torch
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def get_text_biobert_embedding(
    cfg, 
    dataset_name: str, 
    split: str, 
    df: pd.DataFrame,
    full_df: pd.DataFrame,
    logger: logging.Logger,
    batch_size: str
):
    """
    Generate BioBERT text embeddings for prompts in the provided DataFrame.

    Args:
        cfg: Configuration object containing dataset paths, model parameters, etc.
        dataset_name (str): Name of the dataset (used for logging/debugging purposes).
        split (str): Data split identifier ('train', 'valid', or 'test').
        df (pd.DataFrame): Subset of the dataset containing the prompts to embed.
        full_df (pd.DataFrame): The complete dataset used to calculate the optimal max token length.
        logger (logging.Logger): Logger for logging the embedding process.
        batch_size (int, optional): Number of prompts to process per batch. Defaults to DEFAULT_BATCH_SIZE.

    Returns:
        List[torch.Tensor]: List of BioBERT embeddings ordered to match the prompts in `df["prompt"]`.
    """
    path_biobert_embedding_file = cfg.datasets["biobert_embeddings"][split]
    if os.path.exists(path_biobert_embedding_file):
        data = torch.load(path_biobert_embedding_file, map_location="cpu", weights_only=False)
        logger.info(f"Loaded existing BioBERT embeddings from {path_biobert_embedding_file}")
        return data["text_embeddings"]
    
    os.makedirs(os.path.dirname(path_biobert_embedding_file), exist_ok=True)
    
    # Encode unique prompts
    texts_unique_list = df["prompt"].unique().tolist()

    # Load BioBERT model and tokenizer
    model_name = "dmis-lab/biobert-base-cased-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    all_texts_unique_list = full_df["prompt"].unique().tolist()
    token_lens = [len(tokenizer.encode(text, truncation=True)) for text in all_texts_unique_list]
    
    optimal_length = int(np.percentile(token_lens, 95))
    while (optimal_length * 768) % 64 != 0:
        optimal_length += 1
        
    logger.info(f"Computed MAX_LENGTH for BioBERT embeddings ({dataset_name}-{split}): {optimal_length}")

    prompt_to_embedding = {}
    
    for i in tqdm(range(0, len(texts_unique_list), batch_size), desc=f"Embedding {dataset_name}-{split} with BioBERT"):
        batch_texts = texts_unique_list[i:i + batch_size]

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
            batch_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

        batch_embeddings = batch_embeddings.cpu()

        for text, emb in zip(batch_texts, batch_embeddings):
            prompt_to_embedding[text] = emb

    text_embeddings = [prompt_to_embedding[prompt] for prompt in df["prompt"]]
    
    torch.save({"text_embeddings": text_embeddings}, path_biobert_embedding_file)
    logger.info(f"Saved BioBERT embeddings to {path_biobert_embedding_file}")

    return text_embeddings