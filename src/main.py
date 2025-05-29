import os
import torch
import hydra
import logging

from omegaconf import DictConfig, OmegaConf

from src.datasets.getds import get_ds
from src.models.getmodel import get_model
from src.utils import setup_seed, LogExporter, folder_setup

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    run_dir, run_name = folder_setup(cfg)
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, "hydra.run.dir", run_dir)
    OmegaConf.set_readonly(cfg, False)
    
    # Seed setup
    seed = cfg.conductor.seed
    setup_seed(seed=seed)
    
    # Device setup
    idx = cfg.conductor.idx
    device = torch.device(f"cuda:{idx}" if torch.cuda.is_available() else "cpu")
    
    #
    log_exporter = LogExporter(cfg, run_dir, run_name)

    # Model setup
    model = get_model(cfg=cfg, device=device)
    
    # Data setup
    data = get_ds(cfg=cfg)
    
    if cfg.conductor.mode == 'training':
        train_ds, train_dl, valid_ds, valid_dl = data
        logging.info(f"Dataset sizes: train={len(train_ds)}, valid={len(valid_ds)}")
        logging.info(f"Dataloader sizes: train={len(train_dl)}, valid={len(valid_dl)}")
        
        from src.training_imagen import train_imagen
        train_imagen(
            cfg=cfg, model=model, device=device,
            train_dl=train_dl, valid_dl=valid_dl,
            log_exporter=log_exporter, run_dir=run_dir
        )
        
    elif cfg.conductor.mode == 'testing':
        test_ds, test_dl = data
        logging.info(f"Dataset sizes: test={len(test_ds)}")
        logging.info(f"Dataloader sizes: test={len(test_dl)}")
        
    logging.info("DONE!!!")
    
if __name__ == "__main__":
    main()