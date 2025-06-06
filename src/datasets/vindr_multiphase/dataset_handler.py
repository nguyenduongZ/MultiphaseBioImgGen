import os
import cv2
import torch
import logging
import pydicom
import logging
import numpy as np
import pandas as pd
import albumentations as A

from imagen_pytorch.data import Dataset
from sklearn.model_selection import train_test_split

from src.datasets.preprocessing import convert_pixel_to_hu, apply_window, get_text_embeddings

class VindrMultiphase(Dataset):
    def __init__(self, cfg, split: str):
        self.cfg = cfg
        self.split = split
        super().__init__(
            folder=self.cfg.data.PATH_DICOM_DIR, 
            image_size=self.cfg.data.augmentation.image_size
        )
        
        self.set_df_splits()
        if self.split == 'train':
            self.df = self.train_df
        elif self.split == 'valid':
            self.df = self.valid_df
        elif self.split == 'test':
            self.df = self.test_df
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        self.set_transforms()
        self.set_text_embeddings()
        
    def set_df_splits(self):
        csv_path = self.cfg.data.PATH_METADATA_PROMPT_FILE
        if not os.path.exists(csv_path):
            logging.error(f"Metadata CSV not found: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        df.reset_index(inplace=True)
        
        label_mapping = self.cfg.data.classes
        if not all(label in label_mapping for label in df['Label'].unique()):
            unknown_labels = set(df['Label'].unique()) - set(label_mapping.keys())
            print(f"{df['Label'].unique()}")
            msg = f"Unknown labels found: {unknown_labels}"
            logging.error(msg)
            raise ValueError(msg)
        df['LabelIndex'] = df['Label'].map(label_mapping)
        
        unique_studies = df['StudyInstanceUID'].unique()
        train_studies, temp_studies = train_test_split(unique_studies, train_size=0.8, random_state=self.cfg.conductor.seed)
        valid_studies, test_studies = train_test_split(temp_studies, train_size=0.5, random_state=self.cfg.conductor.seed)

        # 80% 10% 10%
        self.train_df = df[df['StudyInstanceUID'].isin(train_studies)].reset_index(drop=True)
        self.valid_df = df[df['StudyInstanceUID'].isin(valid_studies)].reset_index(drop=True)
        self.test_df = df[df['StudyInstanceUID'].isin(test_studies)].reset_index(drop=True)
        
        self.train_df.attrs['split'] = 'train'
        self.train_df.attrs['split'] = 'valid'
        self.train_df.attrs['split'] = 'test'
        self.full_df = df
        
    def set_transforms(self):
        if self.split == 'train':
            transforms = [
                A.HorizontalFlip(**self.cfg.data.augmentation.horizontal_flip.params),
                A.Affine(**self.cfg.data.augmentation.affine.params, mode=cv2.BORDER_CONSTANT)
            ]
        else:
            transforms = []
            
        self.transform = A.Compose(transforms)

    def set_text_embeddings(self):
        text_embeddings = get_text_embeddings(
            cfg=self.cfg,
            split=self.split,
            df=self.df,
            full_df=self.full_df
        )
        
        self.embeddings = {i: emb for i, emb in enumerate(text_embeddings)}
        
        if len(self.embeddings) != len(self.df):
            msg = f"Embedding length ({len(self.embeddings)}) does not match DataFrame length ({len(self.df)}) for {self.split}"
            logging.error(msg)
            raise ValueError(msg)
        
    def __len__(self):
        if self.df is None:
            msg = f"{self.split.capitalize()} DataFrame is not set"
            logging.error(msg)
            raise ValueError(msg)
        
        return len(self.group_keys) if self.return_3d else len(self.df)
    
    def __getitem__(self, idx):
        if self.df is None or self.embeddings is None:
            msg = f"{self.split.capitalize()} DataFrame or embeddings not initialized"
            logging.error(msg)
            raise ValueError(msg)
        
        text_embedding = self.embeddings[idx]
        if idx not in self.embeddings:
            msg = f"Embedding for index {idx} not found"
            logging.error(msg)
            raise ValueError(msg)

        row = self.df.iloc[idx]
        path = row['image_path']
        
        if not os.path.exists(path):
            msg = f"DICOM file not found: {path}"
            logging.error(msg)
            raise FileNotFoundError(msg)
        try:
            dcm = pydicom.dcmread(path, force=True)
        except Exception as e:
            logging.error(f"Failed to read DICOM file {path}: {e}")
            raise
        hu_image = convert_pixel_to_hu(dcm) # (H, W)
        window_center = self.cfg.data['preprocessing']['window_center']
        window_width = self.cfg.data['preprocessing']['window_width']
        image = apply_window(hu_image, window_center, window_width) # (H, W)
        image = self.transform(image=image)['image'] # (H, W)
        image = torch.from_numpy(image).unsqueeze(0).float() # (1, H, W)            
        image = image.expand(3, -1, -1) # (3, H, W)
        return image, text_embedding, row['prompt']