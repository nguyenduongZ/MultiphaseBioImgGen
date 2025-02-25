from torch.utils.data import DataLoader

from .vindrmultiphase import VinDrMultiphase

def get_vindr_multiphase_ds(config, args, testing=False):        
    vindr_dataset = VinDrMultiphase(config)
    
    batch_size = config["trainer"]["batch_size"]
    pin_memory = config["trainer"]["pin_memory"]
    num_workers = config["trainer"]["num_workers"]
    
    if testing:
        sample_ds = vindr_dataset.get_sample_ds()
        
        sample_dl = DataLoader(sample_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
        
        args.num_sample = len(sample_ds)
        args.num_batch = len(sample_dl)
        
        for i in range(5):  
            try:
                print(sample_ds[i])
            except Exception as e:
                print(f"⚠️ Error at index {i}: {e}")

        return (sample_ds, sample_dl), config
    
    else:
        train_ds, valid_ds = vindr_dataset.get_datasets()
        
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
        valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
        
        args.num_train_sample = len(train_ds)
        args.num_valid_sample = len(valid_ds)
        args.num_train_batch = len(train_dl)
        args.num_valid_batch = len(valid_dl)
        
        return (train_ds, valid_ds, train_dl, valid_dl), config
    
def get_ds(config, args, testing=False):
    ds_mapping = {
        "VinDrMultiphase": get_vindr_multiphase_ds
    }
    
    dataset_name = config["data"]["ds"]
    if dataset_name not in ds_mapping:
        raise ValueError(f"Dataset {dataset_name} is not supported. Available datasets: {list(ds_mapping.keys())}")
    
    data, config = ds_mapping[dataset_name](config, args, testing=testing)
    
    return data, config