import os, sys
import cv2
import torch
import pickle
import pandas as pd
import albumentations as A

from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

from .preprocessing import initialize_t5, compute_global_max_length, t5_encode_text
from .basedataset import BaseDataset
from utils import check_dataset_name, check_text_encoder

class VinDrMultiphase(BaseDataset):
    def __init__(self, config):
        self.dataset_name = "VinDrMultiphase"
        self.config = config

        initialize_t5(idx=config["trainer"]["idx"])
        
        self._load_data_()
        self._set_df_splits_()
        self._set_text_embeddings_()
        
    def _load_data_(self):
        csv_file = self.config["data"][self.dataset_name]["PATH_METADATA_PROMPT_FILE"]
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        self.df = pd.read_csv(csv_file, low_memory=False)
        
    @check_dataset_name    
    def _set_df_splits_(self):
        valid_size = self.config["data"][self.dataset_name]["valid_size"]
        random_state = self.config["seed"]
        
        unique_ids = self.df["StudyInstanceUID"].unique()
        train_ids, valid_ids = train_test_split(unique_ids, test_size=valid_size, random_state=random_state)
        
        self.df_train = self.df[self.df["StudyInstanceUID"].isin(train_ids)]
        self.df_valid = self.df[self.df["StudyInstanceUID"].isin(valid_ids)]

    @check_text_encoder    
    def _set_text_embeddings_(self):
        
        batch_size = self.config["trainer"]["batch_size"]        
        
        global_max_length = compute_global_max_length(df_train=self.df_train, df_valid=self.df_valid, batch_size=batch_size)
        
        for split, df in zip(["train", "valid"], [self.df_train, self.df_valid]):
            path_t5_embedding_file = self.config["data"][self.dataset_name][f"PATH_{split.upper()}_EMBEDDING_FILE"]
            
            if os.path.exists(path_t5_embedding_file):
                self.embeddings = torch.load(path_t5_embedding_file, map_location=torch.device("cpu"), weights_only=False)
                print(f"Loaded existing text embeddings from {path_t5_embedding_file}")
                continue
            
            text_embeds_dict = {}
            unique_texts = df["prompt"].unique().tolist()    
                
            for i in tqdm(range(0, len(unique_texts), batch_size), desc=f"Generating {split} T5 embeddings"):
                batch_texts = unique_texts[i : i + batch_size]
                batch_embeddings, _ = t5_encode_text(text_list=batch_texts, max_length=global_max_length)
                
                for text, emb in zip(batch_texts, batch_embeddings):
                    text_embeds_dict[text] = emb.cpu()
                
            torch.save(text_embeds_dict, f=path_t5_embedding_file, pickle_protocol=4)
            print(f"Saved {split} text embeddings to {path_t5_embedding_file}")
            
    def get_datasets(self):
        train_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=3, p=0.5, border_mode = cv2.BORDER_CONSTANT),
            ]
        )
        
        train_ds = BaseDataset(
            config=self.config,
            dataset_name=self.dataset_name,
            df=self.df_train,
            split="train",
            return_text=False,
            aug_transform=train_transform
        )
        
        valid_ds = BaseDataset(
            config=self.config,
            dataset_name=self.dataset_name,
            df=self.df_valid,
            split="valid",
            return_text=True
        )
        
        return train_ds, valid_ds
    
    def get_sample_ds(self):
        sample_df = pd.concat([self.df_train, self.df_valid], ignore_index=True)
        
        print(f"Train size: {len(self.df_train)}")
        print(f"Valid size: {len(self.df_valid)}")
        print(f"Sample size (train + valid): {len(sample_df)}")

        path_sample_embedding_file = self.config["data"][self.dataset_name]["PATH_TEST_EMBEDDING_FILE"]
        
        if os.path.exists(path_sample_embedding_file):
            embeddings = torch.load(path_sample_embedding_file, map_location=torch.device("cpu"), weights_only=False)
            
            # missing_texts = [text for text in sample_df["prompt"].unique() if text not in embeddings] # DEBUG

            # print(f"Total unique texts in sample_ds: {len(sample_df['prompt'].unique())}") # DEBUG
            # print(f"Total embeddings available: {len(embeddings)}")  # DEBUG
            # print(f"Missing text count: {len(missing_texts)}") # DEBUG

            # if missing_texts:
            #     print("Example missing texts:", missing_texts[:10]) # DEBUG

            print(f"Loaded combined text embeddings from {path_sample_embedding_file}")
            
        else:
            embeddings = {}

            for split in ["train", "valid"]:
                path_embedding_file = self.config["data"][self.dataset_name][f"PATH_{split.upper()}_EMBEDDING_FILE"]
                
                if os.path.exists(path_embedding_file):
                    split_embeddings = torch.load(path_embedding_file, map_location=torch.device("cpu"), weights_only=False)
                    
                    for key, value in split_embeddings.items(): # DEBUG
                        if key not in embeddings: # DEBUG
                            # print(f"Warning: Duplicate key found: {key}") # DEBUG
                            embeddings[key] = value # DEBUG
                else: # DEBUG
                    raise FileNotFoundError(f"Embedding file not found: {path_embedding_file}") # DEBUG

            
            torch.save(embeddings, path_sample_embedding_file)
            print(f"Saved combined text embeddings to {path_sample_embedding_file}")
            
        sample_ds = BaseDataset(
            config=self.config,
            dataset_name=self.dataset_name,
            df=sample_df,
            split="test",
            return_text=True
        )
        
        return sample_ds