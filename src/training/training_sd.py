import os, sys
sys.path.append(os.path.abspath(os.curdir))

import torch
import hydra
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from src.datasets.getds import get_ds
from src.models.sd.build_sd import StableDiffusionBuilder
from src.utils import MasterLogger, Logging, EarlyStopping, create_grid_image, setup_seed

@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    OmegaConf.set_readonly(cfg, False)
    
    # MasterLogger
    logger = MasterLogger(cfg).logger
    
    # Logging Setup (WanDB + TensorBoard)
    logging = Logging(cfg)
    
    # Seed setup
    seed = cfg.conductor["seed"]
    setup_seed(seed)

    # Device setup
    idx = cfg.conductor["trainer"]["idx"]
    torch.cuda.set_device(idx)
    device = torch.device(f"cuda:{idx}" if torch.cuda.is_available() else "cpu")

    # Dataset setup
    dataset_name = cfg.datasets["name"]
    train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl = get_ds(cfg=cfg, logger=logger)

    logger.info(f"Dataset sizes: train={len(train_ds)}, valid={len(valid_ds)}, test={len(test_ds)}")
    logger.info(f"Dataloader sizes: train={len(train_dl)}, valid={len(valid_dl)}, test={len(test_dl)}")

    # Model setup
    model = StableDiffusionBuilder(cfg=cfg, device=device, logger=logger)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.conductor["lr"])
    loss_fn = nn.MSELoss()

    # Folder setup
    run_dir = HydraConfig.get().run.dir
    parent_dir = os.path.dirname(run_dir)
    checkpoint_dir = os.path.join(parent_dir, "checkpoints")
    sample_images_dir = os.path.join(parent_dir, "samples")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_images_dir, exist_ok=True)
    
    # model_checkpoint_file = f"checkpoint-iter{{i:06d}}.pt"
    # model_final_save_file = f"checkpoint-final.pt"
    # model_checkpoint_path = lambda i: os.path.join(checkpoint_dir, model_checkpoint_file.format(i=i))
    # model_save_path = os.path.join(checkpoint_dir, model_final_save_file)

    # Early Stopping
    early_stopping = EarlyStopping(
        patience=cfg.conductor["trainer"]["early_stopping"]["patience"],
        min_delta=cfg.conductor["trainer"]["early_stopping"]["min_delta"],
        logger=logger,
        verbose=True
    )
    
    #
    logger.info(f"Starting Stable Diffusion training for {dataset_name}")
    best_loss = float("inf")
    num_epochs = cfg.conductor["trainer"]["epochs"]
    batch_size = cfg.conductor["trainer"]["batch_size"]
    sample_interval = cfg.conductor["validation"]["interval"]["validate_model"]
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch}/{num_epochs}"):
            images = batch[0].to(device)
            text_embeddings = batch[1].to(device)
            
            loss = model.train_step((images, text_embeddings))
            total_loss += loss
        
        avg_loss = total_loss / len(train_dl)
        logging("train_loss", avg_loss, step=epoch)
        logger.info(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")
        
        # Validation
        if epoch % cfg.conductor.validation.interval.valid_loss == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in valid_dl:
                    images = batch[0].to(device)
                    text_embeds = batch[1].to(device)
                    val_loss += model.valid_step((images, text_embeds))
            val_loss /= len(valid_dl)
            logging("valid_loss", val_loss, step=epoch)
            logger.info(f"Epoch {epoch}: Valid Loss = {val_loss:.4f}")
            
            if early_stopping(val_loss, iteration=epoch):
                break

        # Sampling
        if epoch % sample_interval == 0:
            model.eval()
            with torch.no_grad():
                for i, (real_imgs, text_embeds, prompts) in enumerate(test_dl):
                    real_imgs, text_embeds = real_imgs.to(device), text_embeds.to(device)
                    gen_imgs = model.sample(text_embeds)
                    
                    to_pil = T.ToPILImage()
                    real_pils = [to_pil(img.cpu()) for img in real_imgs]
                    gen_pils = [to_pil(img.cpu()) for img in gen_imgs]

                    for j in range(min(len(real_pils), cfg.conductor.sample_quantity)):
                        grid = create_grid_image(real_pils[j], prompts[j], gen_pils[j])
                        grid_path = os.path.join(sample_images_dir, f"epoch{epoch:03d}_{i:03d}_{j:02d}.png")
                        grid.save(grid_path)
                        logging.log_images(grid, prompts[j], iteration=epoch, index=j)

                    break  

        # Checkpoint
        if epoch % sample_interval == 0:
            save_path = os.path.join(checkpoint_dir, f"checkpoint-epoch{epoch:03d}.pt")
            model.save(save_path)
            logger.info(f"Saved checkpoint: {save_path}")

    final_path = os.path.join(checkpoint_dir, "checkpoint-final.pt")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Saved final checkpoint: {final_path}")

if __name__ == "__main__":
    main()