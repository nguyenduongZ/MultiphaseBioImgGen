import os, sys
sys.path.append(os.path.abspath(os.curdir))
import numpy as np

from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from data.data_handler import VinDrMultiphase
from utils.built_opt import Opt

class BaseDataLoader(DataLoader):
    
    def __init__(self, opt: Opt, dataset):

        super().__init__(
            dataset,
            batch_size=opt.conductor["trainer"]["batch_size"],
            shuffle=opt.conductor["trainer"]["shuffle"],
            num_workers=opt.conductor["trainer"]["num_workers"],
            worker_init_fn=lambda id: np.random.seed(id*opt.conductor["testing"]["sample_seed"])
        )
        
def get_vindr_multiphase_ds(opt: Opt, return_text: bool):
    if opt.conductor["model"]["model_type"] == "Imagen":
        return VinDrMultiphase(opt=opt, return_text=return_text)
    
    elif opt.conductor["model"]["model_type"] == "ElucidatedImagen":
        return VinDrMultiphase(opt=opt, return_text=return_text)
    
def split_ds_by_studyID(dataset, train_ratio=0.85, val_ratio=0.15, seed=42):
    assert train_ratio + val_ratio == 1
    
    df = dataset.df
    
    study_dict = {}
    for idx in range(len(df)):
        study_id = df.loc[idx, "StudyInstanceUID"]
        if study_id not in study_dict:
            study_dict[study_id] = []
        study_dict[study_id].append(idx)
        
    study_ids = list(study_dict.keys())
    
    np.random.seed(seed)
    np.random.shuffle(study_ids)
    
    train_study, valid_study = train_test_split(study_ids, test_size=val_ratio, random_state=seed)
    
    train_indices = [idx for s in train_study for idx in study_dict[s]]
    val_indices = [idx for s in valid_study for idx in study_dict[s]]
    
    return train_indices, val_indices

def get_train_valid_ds(opt: Opt, testing=False, return_text=False):
    if opt.data["data"]["dataset"] == "VinDrMultiphase":
        dataset = get_vindr_multiphase_ds(opt=opt, return_text=return_text)
    
    if testing and opt.conductor["testing"]["only_testing"]:
        return dataset
    
    else:
        train_idx, valid_idx = split_ds_by_studyID(dataset=dataset, train_ratio=0.85, val_ratio=0.15)
        
        train_ds = Subset(dataset, train_idx)
        valid_ds = Subset(dataset, valid_idx)
        
        train_ds.dataset.mode = "train"
        valid_ds.dataset.mode = "val"
        
        return train_ds, valid_ds
    
def get_train_valid_dl(opt: Opt, train_ds, valid_ds=None):
    train_dl = BaseDataLoader(opt=opt, dataset=train_ds)
    
    if valid_ds:
        valid_dl = BaseDataLoader(opt=opt, dataset=valid_ds)
        
        return train_dl, valid_dl
    
    else:
        return train_dl