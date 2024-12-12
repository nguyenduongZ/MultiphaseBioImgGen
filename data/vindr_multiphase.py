import os, sys
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A

from transformers import T5Tokenizer, T5EncoderModel
from torch.utils.data import Dataset
from PIL import Image

def get_text_embedding(prompt, tokenizer, encoder, max_length, cache_path):
    # Ensure the directory for the cache file exists
    cache_dir = os.path.dirname(cache_path)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(cache_path):
        # Load from cache
        text_embeds = torch.load(cache_path)
    else:
        # Generate and save to cache
        text_tokens = tokenizer(
            prompt,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=max_length
        ).to("cuda")
        text_embeds = encoder(**text_tokens).last_hidden_state.squeeze(0)
        torch.save(text_embeds, cache_path)
    
    return text_embeds

class VindrMultiphase(Dataset):
    def __init__(self, args, df: pd.DataFrame, split):
        self._split = split
        self.__mode = "train" if self._split == "train" else ("valid" if self._split == "valid" else "test")

        # Resize
        self.resize = A.Compose(
            [
                A.Resize(args.image_size, args.image_size),
            ]
        )

        # Augmentation
        self.aug_transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=3, p=0.5, border_mode = cv2.BORDER_CONSTANT),
            ]
        )

        df_filtered = df[df['instance_number'] % 5 == 0]
        # df_filtered = df_filtered[df_filtered["StudyInstanceUID"] == "1.2.392.200036.9116.2.5.1.37.2418751871.1573609663.550134"]

        self.imgs = df_filtered["image"].values
        self.prompts = df_filtered["prompt"].values
        self.labels = df_filtered["Label"].values

        self.tokenizer = T5Tokenizer.from_pretrained(args.t5_model)
        self.t5_encoder = T5EncoderModel.from_pretrained(args.t5_model).cuda()
        self.max_length = args.max_length
        self.cache_dir = args.cache_dir
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        filename = self.imgs[index]
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File does not exist: {filename}")

        # Load and preprocess the image
        image = np.array(Image.open(filename).convert("RGB"))
        resized = self.resize(image=image)

        if self.__mode == "train":
            transformed_image = self.aug_transforms(image=resized["image"])["image"]
        else:
            transformed_image = resized["image"]

        torch_image = torch.from_numpy(transformed_image).permute(2, 0, 1).float()

        # Generate or load text embeddings
        prompt = self.prompts[index]
        cache_path = os.path.join(self.cache_dir, self.__mode, f"text_embedding_{index}.pt")
        text_embeds = get_text_embedding(prompt, self.tokenizer, self.t5_encoder, self.max_length, cache_path)

        # Load label
        label = self.labels[index]

        return torch_image, text_embeds

    @property
    def mode(self):
        return self.__mode
    
    @mode.setter
    def mode(self, m):
        if m not in ['train', 'valid', 'test']:
            raise ValueError(f"mode cannot be {m} and must be ['train', 'valid', 'test']")
        else:
            self.__mode = m 