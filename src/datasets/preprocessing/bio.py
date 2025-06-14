import os
import torch
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def get_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

def biomed_encode_text(
    cfg,
    split: str,
    df: pd.DataFrame,
    full_df: pd.DataFrame,
    batch_size: int = 64,
):
    dataset_name = cfg.data.name
    path_embedding_file = cfg.data.biomed_embeddings[split]
    
    if os.path.exists(path_embedding_file):
        data = torch.load(path_embedding_file, map_location='cpu')
        logging.info(f"Loaded existing BioMed embeddings from {path_embedding_file}")
        return data['text_embeddings']

    os.makedirs(os.path.dirname(path_embedding_file), exist_ok=True)

    text_unique_list = df['prompt'].unique().tolist()
    all_texts_unique_list = full_df['prompt'].unique().tolist()

    if cfg.model.name == 'imagen':
        model_name = cfg.model.imagen.text_encoder_name
    else:
        model_name = cfg.model.elucidated_imagen.text_encoder_name

    model, tokenizer = get_model_and_tokenizer(model_name)

    token_lens = [len(tokenizer.encode(text, truncation=True)) for text in all_texts_unique_list]
    optimal_length = int(np.percentile(token_lens, 95))
    while (optimal_length * model.config.hidden_size) % 64 != 0:
        optimal_length += 1
    logging.info(f"Computed optimal MAX_LENGTH for {model_name} embeddings ({dataset_name}-{split}): {optimal_length}")

    prompt_to_embedding = {}

    for i in tqdm(range(0, len(text_unique_list), batch_size), desc=f"Embedding {dataset_name}-{split} with {model_name}"):
        batch_texts = text_unique_list[i:i + batch_size]

        encoded = tokenizer.batch_encode_plus(
            batch_texts,
            return_tensors='pt',
            padding='max_length',
            max_length=optimal_length,
            truncation=True
        )

        input_ids = encoded['input_ids']
        attn_mask = encoded['attention_mask']

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            hidden_states = outputs.last_hidden_state  # (B, L, D)

            # Mean pooling
            input_mask_expanded = attn_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
            sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
            embeddings = sum_embeddings / sum_mask

        for text, emb in zip(batch_texts, embeddings):
            prompt_to_embedding[text] = emb.cpu()

    # Final full embeddings
    text_embeddings = [prompt_to_embedding[prompt] for prompt in df['prompt']]

    torch.save({'text_embeddings': text_embeddings}, path_embedding_file)
    logging.info(f"Saved BioMed embeddings to {path_embedding_file}")
    return text_embeddings
