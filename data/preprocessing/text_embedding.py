import os, sys
sys.path.append(os.path.abspath(os.curdir))

import re
import torch
import numpy as np

from tqdm import tqdm
from data.preprocessing.t5 import t5_encode_text
from utils.built_opt import Opt

torch.cuda.empty_cache()

def get_text_t5_embedding(opt: Opt, dataset_name: str, unique_list: list):
    batch_size = opt.conductor["trainer"]["batch_size"]
    path_t5_embedding_file = opt.data["data"][dataset_name]["PATH_T5_EMBEDDING_FILE"]
    
    if opt.data["data"]["use_existing_data_files"] and os.path.exists(path_t5_embedding_file):
        text_embeds = torch.load(path_t5_embedding_file, map_location=torch.device("cpu"))
    
    else:
        opt.logger.debug("Create text embedding")
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(unique_list), batch_size), desc="Create text embedding"):
            batch_texts = unique_list[i:i+batch_size]
            batch_embeds = t5_encode_text(texts=batch_texts, device_index=opt.conductor["trainer"]["idx"])
            all_embeddings.append(batch_embeds.cpu())
            
        text_embeds = torch.cat(all_embeddings, dim=0)
        
        path_t5_embedding_dir = os.path.dirname(path_t5_embedding_file)
        if not os.path.exists(path_t5_embedding_dir):
            os.makedirs(path_t5_embedding_dir, exist_ok=True)
            
        torch.save(text_embeds, f=path_t5_embedding_file)
        
    return text_embeds