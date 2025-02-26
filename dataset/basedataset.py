import os, sys
import torch
import numpy as np
import pandas as pd
import albumentations as A

from PIL import Image
from imagen_pytorch.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, config, dataset_name: str, df: pd.DataFrame, split: str, return_text=False, aug_transform=None):
        self.dataset_name = dataset_name
        self.df = df
        self.return_text = return_text
        self.aug_transform = aug_transform
        
        super().__init__(
            folder=config["data"][self.dataset_name]["PATH_PNG_DIR"],
            image_size=config["data"][self.dataset_name]["image_size"]
        )
        
        self.path_embedding_file = config["data"][self.dataset_name][f"PATH_{split.upper()}_EMBEDDING_FILE"]
        self.embeddings = self._load_embeddings_()
        
    def _load_embeddings_(self):
        if os.path.exists(self.path_embedding_file):
            return torch.load(self.path_embedding_file, map_location=torch.device("cpu"), weights_only=False)
        
        else:
            raise FileNotFoundError(f"Embedding file not found: {self.path_embedding_file}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        text = row["prompt"]
        
        # Image
        image = np.array(Image.open(path).convert("RGB"))
        resize_transform = A.Resize(self.image_size, self.image_size)
        image = resize_transform(image=image)["image"]
        
        if self.aug_transform:
            image = self.aug_transform(image=image)["image"]

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255
    
        # Text embedding
        if text not in self.embeddings:
            raise KeyError(f"Embedding not found for text: {text}")
        text_embedding = self.embeddings[text]

        if self.return_text:
            return image, text_embedding, text
        else:
            return image, text_embedding  