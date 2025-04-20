import os, sys
sys.path.append(os.path.abspath(os.curdir))

import cv2
import torch
import pydicom
import logging
import numpy as np
import pandas as pd
import albumentations as A

from imagen_pytorch.data import Dataset
from sklearn.model_selection import train_test_split

from src.datasets.preprocessing.text_embedding import get_text_t5_embedding

def convert_pixel_to_hu(dcm):
    if not hasattr(dcm, "pixel_array"):
        raise ValueError("DICOM file does not contain pixel data.")
    pixel_array = dcm.pixel_array.astype(np.float32)
    slope = getattr(dcm, "RescaleSlope", 1)
    intercept = getattr(dcm, "RescaleIntercept", 0)
    
    return slope * pixel_array + intercept

def apply_window(img, window_center, window_width):
    if window_center is None or window_width is None:
        raise ValueError("Missing window_center or window_width parameters.")
    window_width = max(window_width, 1.0)
    min_val = window_center - (window_width / 2)
    max_val = window_center + (window_width / 2)
    img = np.clip(img, min_val, max_val)
    img = ((img - min_val) / (max_val - min_val))
    
    return img.astype(np.float32)

class VinDrMultiphase(Dataset):
    _embeddings = {}
    
    def __init__(
        self, 
        cfg, 
        logger: logging.Logger, 
        split: str, 
        return_text=False
    ):
        self.cfg = cfg
        self.logger = logger
        self.split = split
        self.return_text = return_text

        self.image_size = cfg.datasets["augmentation"]["image_size"]
        
        super().__init__(
            folder=cfg.datasets["PATH_DICOM_DIR"],
            image_size=self.image_size
        )
        
        self.set_df_splits()
        if self.split == "train":
            self.df = self.train_df
        elif self.split == "valid":
            self.df = self.valid_df
        elif self.split == "test":
            self.df = self.test_df
        else:
            raise ValueError(f"Invalid split: {self.split}")

        self.set_transforms()
        self.set_text_embeddings()
        
    def set_df_splits(self):
        csv_path = self.cfg.datasets["PATH_METADATA_PROMPT_FILE"]
        if not os.path.exists(csv_path):
            self.logger.error(f"Metadata CSV not found: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        df.reset_index(inplace=True)
        
        unique_studies = df["StudyInstanceUID"].unique()
        train_studies, temp_studies = train_test_split(unique_studies, train_size=0.8, random_state=42)
        valid_studies, test_studies = train_test_split(temp_studies, train_size=0.5, random_state=42)
        
        # 80% 10% 10%
        self.train_df = df[df["StudyInstanceUID"].isin(train_studies)].reset_index(drop=True)
        self.valid_df = df[df["StudyInstanceUID"].isin(valid_studies)].reset_index(drop=True)
        self.test_df = df[df["StudyInstanceUID"].isin(test_studies)].reset_index(drop=True)
        
        self.train_df.attrs["split"] = "train"
        self.valid_df.attrs["split"] = "valid"
        self.test_df.attrs["split"] = "test"
        self.full_df = df
        
    def set_transforms(self):
        transforms = [
            A.Resize(self.image_size, self.image_size)
        ]
        if self.split == "train":
            transforms.extend(
                [
                    A.HorizontalFlip(p=self.cfg.datasets["augmentation"]["horizontal_flip"]["p"]),
                    A.ShiftScaleRotate(
                        shift_limit=self.cfg.datasets["augmentation"]["shift_scale_rotate"]["shift_limit"], 
                        scale_limit=self.cfg.datasets["augmentation"]["shift_scale_rotate"]["scale_limit"], 
                        rotate_limit=self.cfg.datasets["augmentation"]["shift_scale_rotate"]["rotate_limit"], 
                        p=self.cfg.datasets["augmentation"]["shift_scale_rotate"]["p"], 
                        border_mode = cv2.BORDER_CONSTANT),
                ]
            )
        self.transform = A.Compose(transforms)
        
    def set_text_embeddings(self):
        text_embeddings_dict = get_text_t5_embedding(self.cfg, "vindr_multiphase", self.split, self.df, self.full_df, self.logger)

        self.embeddings = {}
        for idx, row in self.df.iterrows():
            if idx in text_embeddings_dict:
                self.embeddings[idx] = text_embeddings_dict[idx]
            else:
                msg = f"Embedding for index {idx} not found in embeddings for {self.split} split"
                self.logger.error(msg)
                raise ValueError(msg)
        
        if len(self.embeddings) != len(self.df):
            msg = f"Embedding length ({len(self.embeddings)}) does not match DataFrame length ({len(self.df)}) for {self.split}"
            self.logger.error(msg)
            raise ValueError(msg)
        
    def __len__(self):
        if self.df is None:
            msg = f"{self.split.capitalize()} DataFrame is not set"
            self.logger.error(msg)
            raise ValueError(msg)
        
        return len(self.df)

    def __getitem__(self, idx):
        if self.df is None or self.embeddings is None:
            msg = f"{self.split.capitalize()} DataFrame or embeddings not initialized"
            self.logger.error(msg)
            raise ValueError(msg)
        
        row = self.df.iloc[idx]
        path = row["image_path"]
        prompt = row["prompt"]
        
        if not os.path.exists(path):
            msg = f"DICOM file not found: {path}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)
        
        try:
            dcm = pydicom.dcmread(path, force=True)
        except Exception as e:
            self.logger.error(f"Failed to read DICOM file {path}: {e}")
            raise
        
        hu_image = convert_pixel_to_hu(dcm)
        window_center = self.cfg.datasets["preprocessing"]["window_center"]
        window_width = self.cfg.datasets["preprocessing"]["window_width"]
        image = apply_window(hu_image, window_center, window_width)
        image = self.transform(image=image)["image"]
        image = torch.from_numpy(image).unsqueeze(0).float()
        image = image.expand(3, -1, -1)
        
        text_embedding = self.embeddings[idx]
        
        if idx not in self.embeddings:
            msg = f"Embedding for index {idx} not found"
            self.logger.error(msg)
            raise ValueError(msg)
        
        # self.logger.debug(f"Index: {idx}, Image Path: {path}, Prompt: {prompt}, Embedding Shape: {text_embedding.shape}")
        
        return (image, text_embedding, prompt) if self.return_text else (image, text_embedding)
