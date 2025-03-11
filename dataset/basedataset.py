import os, sys
import torch
import pydicom
import numpy as np
import pandas as pd
import albumentations as A

from PIL import Image
from imagen_pytorch.data import Dataset

from utils import Opt

def convert_pixel_to_hu(dcm):
    if not hasattr(dcm, "pixel_array"):
        raise ValueError("DICOM file does not contain pixel data.")
    
    pixel_array = dcm.pixel_array.astype(np.float32)
    slope = getattr(dcm, "RescaleSlope", 1)
    intercept = getattr(dcm, "RescaleIntercept", 0)
    
    hu_image = slope * pixel_array + intercept
    
    return hu_image

def apply_window(img, window_center, window_width):
    if window_center is None or window_width is None:
        raise ValueError("Missing window_center or window_width parameters.")
    
    window_width = max(window_width, 1.0)
    
    min_val = window_center - (window_width / 2)
    max_val = window_center + (window_width / 2)
    
    img = np.clip(img, min_val, max_val)
    img = ((img - min_val) / (max_val - min_val)) # [0, 1]
    # img = (img * 2) - 1  # [-1, 1]
    
    return img.astype(np.float32)

class BaseDataset(Dataset):
    def __init__(self, opt: Opt, dataset_name: str, df: pd.DataFrame, split: str, return_text=False, aug_transform=None):
        self.opt = opt
        self.dataset_name = dataset_name
        self.df = df
        self.return_text = return_text
        self.aug_transform = aug_transform
        self.image_size = opt.dataset["data"][self.dataset_name]["image_size"]
        
        super().__init__(
            folder=opt.dataset["data"][self.dataset_name]["PATH_DICOM_DIR"],
            image_size=self.image_size
        )
        
        self.path_embedding_file = opt.dataset["data"][self.dataset_name][f"PATH_{split.upper()}_EMBEDDING_FILE"]
        self.embeddings = self._load_embeddings_()
        
    def _load_embeddings_(self):
        if os.path.exists(self.path_embedding_file):
            embeddings = torch.load(self.path_embedding_file, map_location=torch.device("cpu"), weights_only=False)
            
            return embeddings
        
        else:
            raise FileNotFoundError(f"Embedding file not found: {self.path_embedding_file}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = row["StudyInstanceUID"]
        path = row["image_path"]
        
        # Image    
        def read_dicom(path):
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            
            dcm = pydicom.dcmread(path, force=True)
            
            if not hasattr(dcm, "pixel_array"):
                raise ValueError(f"DICOM file {path} does not contain pixel data.")
            
            hu_image = convert_pixel_to_hu(dcm)

            window_center = self.opt.dataset["data"][self.dataset_name]["window_center"]
            window_width = self.opt.dataset["data"][self.dataset_name]["window_width"]

            return apply_window(hu_image, window_center, window_width)

        image = read_dicom(path)

        resize_transform = A.Resize(self.image_size, self.image_size)         
        image = resize_transform(image=image)["image"]
            
        if self.aug_transform:
            image = self.aug_transform(image=image)["image"]

        image = torch.from_numpy(image).unsqueeze(0).float()
        image = image.expand(3, -1, -1)
        
        # Text embedding
        text = row["prompt"]
        instance_num = str(row["InstanceNumber"])
        key = f"{str(study_id)}_{instance_num}"
        
        if key not in self.embeddings:
            raise KeyError(f"Embedding not found for key: {key}")
        text_embedding = self.embeddings[key]

        if self.return_text:
            return image, text_embedding, text
        else:
            return image, text_embedding  