from torch.utils.data import DataLoader
from .vindrmultiphase import VinDrMultiphase
from utils import Opt

def get_vindr_multiphase_ds(opt: Opt, testing=False): 
    vindr_dataset = VinDrMultiphase(opt=opt)
    
    batch_size = opt.conductor["trainer"]["batch_size"]
    pin_memory = opt.conductor["trainer"]["pin_memory"]
    num_workers = opt.conductor["trainer"]["num_workers"]
    
    if testing:
        sample_ds = vindr_dataset.get_sample_ds()
        sample_dl = DataLoader(sample_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
        
        return sample_ds, sample_dl
        
    else:
        train_ds, valid_ds = vindr_dataset.get_datasets()
        valid_patient_ds = vindr_dataset.get_valid_patient_ds()
        
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
        valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
        valid_patient_dl = DataLoader(valid_patient_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
        
        return train_ds, valid_ds, valid_patient_ds, train_dl, valid_dl, valid_patient_dl
    
def get_ds(opt: Opt, testing=False):
    ds_mapping = {
        "VinDrMultiphase": get_vindr_multiphase_ds
    }
    
    dataset_name = opt.dataset["data"]["ds"]
    
    if dataset_name not in ds_mapping:
        raise ValueError(f"Dataset {dataset_name} is not supported. Available datasets: {list(ds_mapping.keys())}")

    return ds_mapping[dataset_name](opt=opt, testing=testing)