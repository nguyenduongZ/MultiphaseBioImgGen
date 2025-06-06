import numpy as np

from torch.utils.data import DataLoader

from src.datasets.vindr_multiphase.dataset_handler import VinDrMultiphase

def get_vindr_multiphase_ds(cfg):
    assert cfg.data.name == 'vindr_multiphase', 'Create DataLoader for VinDrMultiphase'
    dl_kwargs = {
        'batch_size': cfg.data.batch_size,
        'num_workers': cfg.data.num_workers,
        'pin_memory': cfg.data.pin_memory,
        'shuffle': cfg.data.shuffle,
        'worker_init_fn': lambda worker_id: np.random.seed(cfg.conductor.seed + 100 * worker_id)
    }
    
    if cfg.conductor.mode == 'testing':
        test_ds = VinDrMultiphase(cfg=cfg, split='test')
        test_dl = DataLoader(test_ds, **dl_kwargs)
        
        cfg.data.num_test_sample = len(test_ds)
        cfg.data.num_test_batch = len(test_dl)
        return test_ds, test_dl
    else:
        train_ds = VinDrMultiphase(cfg=cfg, split='train')
        valid_ds = VinDrMultiphase(cfg=cfg, split='valid')
        
        train_dl = DataLoader(train_ds, **dl_kwargs)
        valid_dl = DataLoader(valid_ds, **dl_kwargs)
        
        cfg.data.num_train_sample = len(train_ds)
        cfg.data.num_valid_sample = len(valid_ds)
        cfg.data.num_train_batch = len(train_dl)
        cfg.data.num_valid_batch = len(valid_dl)
        return train_ds, train_dl, valid_ds, valid_dl
    
def get_ds(cfg):
    dataset_name = cfg.data.name
    dataset_mapping = {
        'vindr_multiphase': get_vindr_multiphase_ds
    }
    
    data = dataset_mapping[dataset_name](cfg=cfg)
    return data 