import os, sys
import cv2
import torch
import pandas as pd
import albumentations as A

from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold

from .preprocessing import initialize_t5, compute_global_max_length, t5_encode_text
from .basedataset import BaseDataset
from utils import check_dataset_name, check_text_encoder, Opt

class VinDrMultiphase(BaseDataset):
    def __init__(self, opt: Opt):
        self.dataset_name = "VinDrMultiphase"
        self.opt = opt

        initialize_t5(idx=opt.conductor["trainer"]["idx"])
        
        self._load_data_()
        self._set_df_splits_()
        self._set_text_embeddings_()
        
    def _load_data_(self):
        csv_file = self.opt.dataset["data"][self.dataset_name]["PATH_METADATA_PROMPT_FILE"]
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        self.df = pd.read_csv(csv_file, low_memory=False)
        
    @check_dataset_name    
    def _set_df_splits_(self):
        random_state = self.opt.conductor["seed"]
        valid_size = self.opt.dataset["data"][self.dataset_name]["valid_size"]
        k_folds = self.opt.dataset["data"][self.dataset_name]["k_folds"]
        
        unique_ids = self.df["StudyInstanceUID"].unique()
        train_ids, valid_ids = train_test_split(unique_ids, test_size=valid_size, random_state=random_state)

        self.df_train = self.df[self.df["StudyInstanceUID"].isin(train_ids)].copy()
        self.df_valid_patient = self.df[self.df["StudyInstanceUID"].isin(valid_ids)].copy()

        df_slices = self.df_train[["StudyInstanceUID", "InstanceNumber", "Label"]].copy()
        df_slices = df_slices.groupby(["StudyInstanceUID", "Label"]).apply(
            lambda x: x.sort_values("InstanceNumber").iloc[::5]
        ).reset_index(drop=True)
        
        df_slices["FoldLabel"] = df_slices["StudyInstanceUID"] + "_" + df_slices["Label"]

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        
        self.kfolds = []

        for train_idx, valid_idx in skf.split(df_slices, df_slices["FoldLabel"]):
            train_slices_fold = df_slices.iloc[train_idx]
            valid_slices_fold = df_slices.iloc[valid_idx]
            
            df_train_fold = self.df_train[self.df_train["InstanceNumber"].isin(train_slices_fold["InstanceNumber"])]
            df_valid_fold = self.df_train[self.df_train["InstanceNumber"].isin(valid_slices_fold["InstanceNumber"])]
            
            self.kfolds.append((df_train_fold, df_valid_fold))

    @check_text_encoder    
    def _set_text_embeddings_(self):
        batch_size = self.opt.conductor["trainer"]["batch_size"]        
        global_max_length = compute_global_max_length(df_train=self.df_train, df_valid=self.df_valid_patient, batch_size=batch_size)
        
        for split, df in {"train": self.df_train, "valid_patient": self.df_valid_patient}.items():
            path_embedding_file = self.opt.dataset["data"][self.dataset_name][f"PATH_{split.upper()}_EMBEDDING_FILE"]
            self._generate_embeddings(df, path_embedding_file, global_max_length, batch_size, split)
        
        path_valid_fold_embedding_file = self.opt.dataset["data"][self.dataset_name]["PATH_VALID_FOLD_EMBEDDING_FILE"]
        if not os.path.exists(path_valid_fold_embedding_file):
            self.opt.logger.info(f"Creating combined validation embeddings...")
            
            combined_embeddings = {}
            for fold_idx, (_, df_valid_fold) in enumerate(self.kfolds):
                fold_embeddings = self._generate_embeddings(
                    df_valid_fold, None, global_max_length, batch_size, f"valid fold {fold_idx}"
                )
                combined_embeddings.update(fold_embeddings)
                
            torch.save(combined_embeddings, path_valid_fold_embedding_file)
            self.opt.logger.info(f"Saved all validation embeddings to {path_valid_fold_embedding_file}")    
        
        else:
            self.opt.logger.info(f"Loaded existing validation embeddings from {path_valid_fold_embedding_file}")
            
        path_sample_embedding_file = self.opt.dataset["data"][self.dataset_name]["PATH_TEST_EMBEDDING_FILE"]
        if not os.path.exists(path_sample_embedding_file):
            combined_embeddings = {}
            
            for fold_idx in range(len(self.kfolds)):
                path_valid_fold_embedding_file = self.opt.dataset["data"][self.dataset_name][f"PATH_VALID_FOLD_EMBEDDING_FILE"]
                if os.path.exists(path_valid_fold_embedding_file):
                    fold_embeddings = torch.load(path_valid_fold_embedding_file, map_location="cpu", weights_only=False)
                    combined_embeddings.update(fold_embeddings)
                
            path_valid_patient_embedding_file = self.opt.dataset["data"][self.dataset_name]["PATH_VALID_PATIENT_EMBEDDING_FILE"]
            if os.path.exists(path_valid_patient_embedding_file):    
                valid_patient_embeddings = torch.load(path_valid_patient_embedding_file, map_location="cpu", weights_only=False)
                combined_embeddings.update(valid_patient_embeddings)
            
            torch.save(combined_embeddings, path_sample_embedding_file)
            self.opt.logger.info(f"Saved sample text embeddings to {path_sample_embedding_file}")
            
    def _generate_embeddings(self, df, path_file, global_max_length, batch_size, desc):
        if path_file and os.path.exists(path_file):  
            self.opt.logger.info(f"Loaded existing {desc} text embeddings from {path_file}")      
            return torch.load(path_file, map_location="cpu", weights_only=False)

        self.opt.logger.info(f"Creating text embeddings for {desc}...")
        text_embeds_dict = {}

        df = df.copy()
        df["Study_Instance_Text"] = (
            df["StudyInstanceUID"].astype(str) + "_" +
            df["InstanceNumber"].astype(str)
        )
        unique_texts = df["Study_Instance_Text"].unique().tolist()
        
        for i in tqdm(range(0, len(unique_texts), batch_size), desc=f"Generating {desc} T5 embeddings"):
            batch_texts = unique_texts[i : i + batch_size]
            batch_embeddings, _ = t5_encode_text(
                text_list=[t.split("_", 1)[1] for t in batch_texts], 
                max_length=global_max_length
            )
            
            for study_instance_text, emb in zip(batch_texts, batch_embeddings):
                key = study_instance_text
                text_embeds_dict[key] = emb.cpu()
                self.opt.logger.debug(f"Saved embedding key: {key}") 
        
        if path_file:
            torch.save(text_embeds_dict, path_file)
            self.opt.logger.info(f"Saved {desc} text embeddings to {path_file}")
            return None
        
        else:
            return text_embeds_dict            
                
    def get_datasets(self, fold_idx=0):    
        if fold_idx >= len(self.kfolds):
            raise ValueError(f"fold_idx {fold_idx} exceeds folds count {len(self.kfolds)}")
            
        train_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=3, p=0.5, border_mode = cv2.BORDER_CONSTANT),
            ]
        )
        
        df_train_fold, df_valid_fold = self.kfolds[fold_idx]
        
        train_ds = BaseDataset(
            opt=self.opt,
            dataset_name=self.dataset_name,
            df=df_train_fold,
            split="train",
            return_text=False,
            aug_transform=train_transform
        )
        
        valid_ds = BaseDataset(
            opt=self.opt,
            dataset_name=self.dataset_name,
            df=df_valid_fold,
            split=f"valid_fold",
            return_text=True
        )
        
        return train_ds, valid_ds
    
    def get_valid_patient_ds(self):
        return BaseDataset(
            opt=self.opt,
            dataset_name=self.dataset_name,
            df=self.df_valid_patient,
            split="valid_patient",
            return_text=True
        )    
    
    def get_sample_ds(self):
        sample_df = pd.concat([df for _, df in self.kfolds] + [self.df_valid_patient], ignore_index=True)
        
        sample_ds = BaseDataset(
            opt=self.opt,
            dataset_name=self.dataset_name,
            df=sample_df,
            split="test",
            return_text=True
        )
        
        return sample_ds