import os, sys
sys.path.append(os.path.abspath(os.curdir))

import clip
import torch
import logging
import pandas as pd

from tqdm import tqdm

def get_text_clip_embedding(
    cfg,
    dataset_name: str,
    split: str,
    df: pd.DataFrame,
    full_df: pd.DataFrame,
    batch_size: str,
    logger: logging.Logger = None
):
    idx = cfg.conductor['trainer']['idx']
    device = torch.device(f"cuda:{idx}" if torch.cuda.is_available() else "cpu")
    model, _ = clip.load("ViT-B/32", device=device)
    
    path_clip_embedding_file = cfg.datasets["clip_embeddings"][split]
    if os.path.exists(path_clip_embedding_file):
        data = torch.load(path_clip_embedding_file, map_location="cpu", weights_only=False)
        logger.info(f"Loaded existing CLIP embeddings from {path_clip_embedding_file}")
        return data["text_embeddings"]
    
    os.makedirs(os.path.dirname(path_clip_embedding_file), exist_ok=True)

    texts_unique_list = df["prompt"].unique().tolist()
    prompt_to_embedding = {}
    
    for i in tqdm(range(0, len(texts_unique_list), batch_size), desc=f"Embedding {dataset_name}-{split} with CLIP"):
        batch_texts = texts_unique_list[i:i+batch_size]
        tokenized = clip.tokenize(batch_texts).to(device)
        with torch.no_grad():
            text_features = model.encode_text(tokenized).cpu()
        
        for text, emb in zip(batch_texts, text_features):
            prompt_to_embedding[text] = emb
        
    text_embeddings = [prompt_to_embedding[prompt] for prompt in df["prompt"]]
    torch.save({"text_embeddings": text_embeddings}, path_clip_embedding_file)
    logger.info(f"Saved CLIP embeddings to {path_clip_embedding_file}")
    return text_embeddings