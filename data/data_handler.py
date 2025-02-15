import os, sys
sys.path.append(os.path.abspath(os.curdir))

import torch
import bisect
import numpy as np
import pandas as pd
import albumentations as A

from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
from data.preprocessing.df_coding import get_df_coding
from data.preprocessing.text_embedding import get_text_t5_embedding
from utils.decorators import check_dataset_name, check_text_encoder
from utils.built_opt import Opt

class BaseDataset(Dataset):
    def __init__(self, dataset_name: str, opt: Opt, return_text=False, mode="train"):
        super().__init__()
        
        self.TEXT_ENCODER_NAME = opt.imagen["imagen"]["text_encoder_name"]
        self.DATASET = dataset_name
        
        self.return_text = return_text
        self.multi_gpu = opt.conductor["trainer"]["multi_gpu"]
                
        self.image_folder = opt.data["data"][self.DATASET]["PATH_PNG_DIR"]
        self.image_size = opt.data["data"]["image_size"]
        
        self.df = None
        self.text_embeds = None
        
        self.mode = mode
        
        self.resize = A.Compose(
            [
                A.Resize(self.image_size, self.image_size)
            ]
        )
        
        self.aug_transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.2),
                A.Affine(scale=(0.9, 1.1), translate_percent=(0.0625, 0.0625), rotate=(-3, 3), p=0.5)
            ]
        )
        
        self.to_tensor = A.Compose(
            [
                ToTensorV2()
            ]
        )
        
    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, m):
        if m not in ['train', 'val']:
            raise ValueError(f"mode cannot be {m}, must be ['train', 'val']")
        self.__mode = m
          
    @check_text_encoder
    def _set_text_embeds_(self, opt: Opt):
        self.text_unique_list = self.df["prompt"].unique().tolist()
        
        self.text_embeds = get_text_t5_embedding(
            opt=opt,
            dataset_name=self.DATASET,
            unique_list=self.text_unique_list
        )
        
        opt.logger.debug("Text embedding created" + str(self.text_embeds.size()))
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_path = self.df.loc[idx, "image_path"]
        study_id = self.df.loc[idx, "StudyInstanceUID"]
        label = self.df.loc[idx, "Label"]
        
        image = np.array(Image.open(image_path).convert("RGB"))
        image = self.resize(image=image)["image"]

        if self.mode == "train":
            image = self.aug_transforms(image=image)["image"]
            
        image = self.to_tensor(image=image)["image"]
                    
        text = self.df.loc[idx, "prompt"]
        index_unique = self.text_unique_list.index(text)
        text_embedding = self.text_embeds[index_unique]
        
        if self.return_text:
            return image, study_id, label, text_embedding, text
        
        else:
            return image, study_id, label, text_embedding
        
class VinDrMultiphase(BaseDataset):
    def __init__(self, opt: Opt, return_text=False):
        super().__init__(dataset_name="VinDrMultiphase", opt=opt, return_text=return_text)
        
        self._set_df_(opt=opt)
        self._set_text_embeds_(opt=opt)

    @check_dataset_name
    def _set_df_(self, opt: Opt):
        self.df = get_df_coding(opt=opt, dataset_name=self.DATASET)