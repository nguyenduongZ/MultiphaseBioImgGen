import logging
import numpy as np

from torch.utils.data import DataLoader, ConcatDataset

from src.datasets.vindr_multiphase.vindr_multiphase import VinDrMultiphase

def get_vindr_multiphase_ds(cfg, logger: logging.Logger, use_text_embedding: bool = True, return_3d: bool = False, testing: bool = False):
    train_ds = VinDrMultiphase(cfg=cfg, logger=logger, split="train", use_text_embedding=use_text_embedding, return_3d=return_3d)
    valid_ds = VinDrMultiphase(cfg=cfg, logger=logger, split="valid", use_text_embedding=use_text_embedding, return_3d=return_3d)
    test_ds = VinDrMultiphase(cfg=cfg, logger=logger, split="test", use_text_embedding=use_text_embedding, return_3d=return_3d)
    
    cfg.datasets["num_train_sample"] = len(train_ds)
    cfg.datasets["num_valid_sample"] = len(valid_ds)
    cfg.datasets["num_test_sample"] = len(test_ds)
    
    full_ds = ConcatDataset([train_ds, valid_ds, test_ds])
    
    dl_kwargs = {
        "batch_size": cfg.conductor["trainer"]["batch_size"],
        "num_workers": cfg.conductor["trainer"]["num_workers"],
        "pin_memory": cfg.conductor["trainer"]["pin_memory"],
        "worker_init_fn": lambda worker_id: np.random.seed(cfg.conductor["seed"] + 100 * worker_id)
    }
    
    if testing:
        full_dl = DataLoader(full_ds, **dl_kwargs, shuffle=False)
        return full_ds, full_dl
    else:
        train_dl = DataLoader(train_ds, **dl_kwargs, shuffle=cfg.conductor["trainer"]["shuffle"])
        valid_dl = DataLoader(valid_ds, **dl_kwargs, shuffle=cfg.conductor["trainer"]["shuffle"])
        test_dl = DataLoader(test_ds, **dl_kwargs, shuffle=cfg.conductor["trainer"]["shuffle"])
        
        cfg.datasets["num_train_batch"] = len(train_dl)
        cfg.datasets["num_valid_batch"] = len(valid_dl)
        cfg.datasets["num_test_batch"] = len(test_dl)
    
        return train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl

def get_ds(cfg, logger: logging.Logger, use_text_embedding: bool = True, return_3d: bool = False, testing: bool = False):
    dataset_name = cfg.datasets["name"]
    dataset_mapping = {
        "vindr_multiphase": get_vindr_multiphase_ds
    }
    
    return dataset_mapping[dataset_name](cfg=cfg, logger=logger, use_text_embedding=use_text_embedding, return_3d=return_3d, testing=testing)