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

from src.datasets.preprocessing.text_embedding import get_text_embedding

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
        use_text_embedding: bool = True,
        return_3d: bool = False 
    ):
        self.cfg = cfg
        self.logger = logger
        self.split = split
        self.use_text_embedding = use_text_embedding
        self.return_3d = return_3d
        
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

        if self.return_3d:
            self.grouped_df = self.df.groupby(["StudyInstanceUID", "SeriesInstanceUID"])
            self.group_keys = list(self.grouped_df.keys())
        
        self.set_transforms()
        if self.use_text_embedding:
            self.set_text_embeddings()
        else:
            self.embeddings = None
        
    def set_df_splits(self):
        csv_path = self.cfg.datasets["PATH_METADATA_PROMPT_FILE"]
        if not os.path.exists(csv_path):
            self.logger.error(f"Metadata CSV not found: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        df.reset_index(inplace=True)

        label_mapping = {
            "Non contrast": 0,
            "Arterial": 1,
            "Venous": 2,
            "Other": 3
        }
        if not all(label in label_mapping for label in df["Label"].unique()):
            unknown_labels = set(df["Label"].unique()) - set(label_mapping.keys())
            msg = f"Unknown labels found: {unknown_labels}"
            self.logger.error(msg)
            raise ValueError(msg)
        
        df["Label"] = df["Label"].map(label_mapping)

        unique_studies = df["StudyInstanceUID"].unique()
        train_studies, temp_studies = train_test_split(unique_studies, train_size=0.8, random_state=self.cfg.conductor["seed"])
        valid_studies, test_studies = train_test_split(temp_studies, train_size=0.5, random_state=self.cfg.conductor["seed"])
        
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
        text_embeddings = get_text_embedding(
            cfg=self.cfg,
            dataset_name="vindr_multiphase",
            split=self.split,
            df=self.df,
            full_df=self.full_df,
            logger=self.logger
        )

        self.embeddings = {i: emb for i, emb in enumerate(text_embeddings)}
        
        if len(self.embeddings) != len(self.df):
            msg = f"Embedding length ({len(self.embeddings)}) does not match DataFrame length ({len(self.df)}) for {self.split}"
            self.logger.error(msg)
            raise ValueError(msg)
        
    def __len__(self):
        if self.df is None:
            msg = f"{self.split.capitalize()} DataFrame is not set"
            self.logger.error(msg)
            raise ValueError(msg)
        
        return len(self.group_keys) if self.return_3d else len(self.df)

    def __getitem__(self, idx):
        if self.df is None or self.embeddings is None:
            msg = f"{self.split.capitalize()} DataFrame or embeddings not initialized"
            self.logger.error(msg)
            raise ValueError(msg)
        
        # 3D
        if self.return_3d:
            study_id, series_id = self.group_keys[idx]
            group = self.grouped_df.get_group((study_id, series_id))
            
            group = group.sort_values("InstanceNumber")
            
            image_paths = group["image_path"].tolist()
            label = group["Label"].iloc[0]

            slices = []
            for path in image_paths:
                if not os.path.exists(path):
                    msg = f"DICOM file not found: {path}"
                    self.logger.error(msg)
                    raise FileNotFoundError(msg)
                
                try:
                    dcm = pydicom.dcmread(path, force=True)  # (H, W)
                except Exception as e:
                    self.logger.error(f"Failed to read DICOM file {path}: {e}")
                    raise
                
                hu_image = convert_pixel_to_hu(dcm)
                window_center = self.cfg.datasets["preprocessing"]["window_center"]
                window_width = self.cfg.datasets["preprocessing"]["window_width"]
                image = apply_window(hu_image, window_center, window_width)
                
                image = self.transform(image=image)["image"]
                
                slices.append(image)
                
            volume = np.stack(slices, axis=0) # (D, H, W)
            volume = torch.from_numpy(volume).float()
            volume = volume.unsqueeze(0) # (1, D, H, W)
            
            return volume, label
        
        # 2D    
        else:
            row = self.df.iloc[idx]
            path = row["image_path"]
            prompt = row["prompt"]
            label = row["Label"]
            
            if not os.path.exists(path):
                msg = f"DICOM file not found: {path}"
                self.logger.error(msg)
                raise FileNotFoundError(msg)
            
            try:
                dcm = pydicom.dcmread(path, force=True) # (H, W)
            except Exception as e:
                self.logger.error(f"Failed to read DICOM file {path}: {e}")
                raise

            hu_image = convert_pixel_to_hu(dcm) # (H, W)
            window_center = self.cfg.datasets["preprocessing"]["window_center"]
            window_width = self.cfg.datasets["preprocessing"]["window_width"]
            image = apply_window(hu_image, window_center, window_width) # (H, W)
            image = self.transform(image=image)["image"] # (H, W)
            image = torch.from_numpy(image).unsqueeze(0).float() # (1, H, W)            
            image = image.expand(3, -1, -1) # (3, H, W)
            
            if self.use_text_embedding:
                text_embedding = self.embeddings[idx]
            
                if idx not in self.embeddings:
                    msg = f"Embedding for index {idx} not found"
                    self.logger.error(msg)
                    raise ValueError(msg)
                        
                return image, text_embedding, prompt
            else:
                return image, None, prompt, label