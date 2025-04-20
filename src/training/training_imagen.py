import os, sys
sys.path.append(os.path.abspath(os.curdir))

import torch
import hydra
import numpy as np
import torchvision.transforms as T

from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from src.datasets.getds import get_ds
from src.models.imagen.build_imagen import ImagenBuilder
from src.utils import MasterLogger, Logging, EarlyStopping, create_grid_image, setup_seed

def clear_old_checkpoints(current_i, checkpoint_dir, max_checkpoints=5):
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint-iter")])
    if len(checkpoint_files) > max_checkpoints:
        for file in checkpoint_files[:-max_checkpoints]:
            os.remove(os.path.join(checkpoint_dir, file))

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

    # Model setup
    model_type = cfg.models["name"]
    model = ImagenBuilder(cfg=cfg, device=device, logger=logger)
    trainer = model.trainer
    unet_number = cfg.conductor["unet_number"]
    cond_scale = cfg.conductor["cond_scale"]
    
    # Dataset setup
    dataset_name = cfg.datasets["name"]
    train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl = get_ds(cfg=cfg, logger=logger)

    logger.info(f"Dataset sizes: train={len(train_ds)}, valid={len(valid_ds)}, test={len(test_ds)}")
    logger.info(f"Dataloader sizes: train={len(train_dl)}, valid={len(valid_dl)}, test={len(test_dl)}")

    for v, (img, emb, prompt) in enumerate(test_dl):
        print(prompt[:3]) 
        if v > 2:
            break
    
    trainer.add_train_dataloader(dl=train_dl)
    trainer.add_valid_dataloader(dl=valid_dl)
    
    # Folder setup
    run_dir = HydraConfig.get().run.dir
    parent_dir = os.path.dirname(run_dir)
    checkpoint_dir = os.path.join(parent_dir, "checkpoints")
    sample_images_dir = os.path.join(parent_dir, "samples")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_images_dir, exist_ok=True)
    
    model_checkpoint_file = f"checkpoint-iter{{i:06d}}.pt"
    model_final_save_file = f"checkpoint-final.pt"
    model_checkpoint_path = lambda i: os.path.join(checkpoint_dir, model_checkpoint_file.format(i=i))
    model_save_path = os.path.join(checkpoint_dir, model_final_save_file)

    # Early Stopping
    early_stopping = EarlyStopping(
        patience=cfg.conductor["trainer"]["early_stopping"]["patience"],
        min_delta=cfg.conductor["trainer"]["early_stopping"]["min_delta"],
        logger=logger,
        verbose=True
    )
    
    #
    logger.info(f"Starting training for {dataset_name}")
    total_iterations = cfg.conductor["trainer"]["iterations"]
    iteration_bar = tqdm(range(1, total_iterations + 1), desc="Training", dynamic_ncols=True)
    postfix_dict = {
        "Train": 0.0,
        "Valid": 0.0,
        "Iter": f"0/{total_iterations}"
    }
    iteration_bar.set_postfix(postfix_dict)
    last_checkpoint_i = None
    for i in iteration_bar:
        # Training
        iteration_bar.set_description(f"Training {dataset_name}_{model_type}_{unet_number}")
        loss = trainer.train_step(unet_number=unet_number)
        
        logging("train_loss", loss, step=i)
        
        postfix_dict["Train"] = f"{loss:.4f}"
        postfix_dict["Iter"] = f"{i}/{total_iterations}"
        iteration_bar.set_postfix(postfix_dict)
        
        # Validation
        if not (i % cfg.conductor["validation"]["interval"]["valid_loss"]):
            iteration_bar.set_description("Validating")
            valid_loss = trainer.valid_step(unet_number=unet_number)
            
            logging("valid_loss", valid_loss, step=i)
            
            postfix_dict["Valid"] = f"{valid_loss:.4f}"
            iteration_bar.set_postfix(postfix_dict)
            
            if early_stopping(valid_loss, iteration=i):
                break
            
        # Samples
        if not (i % cfg.conductor["validation"]["interval"]["validate_model"]):
            if cfg.conductor["validation"]["display_samples"]:
                try:
                    test_iter = iter(test_dl)
                    n = int(np.ceil(cfg.conductor["sample_quantity"] / cfg.conductor["trainer"]["batch_size"]))
                    for _ in range(n):
                        orig_images, embed_batch, text_batch = next(test_iter)
                        sample_images = trainer.sample(
                            text_embeds=embed_batch.to(device),
                            return_pil_images=True,
                            stop_at_unet_number=unet_number,
                            use_tqdm=False,
                            cond_scale=cond_scale
                        )
                        
                        to_pil = T.ToPILImage()
                        orig_images_pil = [to_pil(img) for img in orig_images]
                        
                        # DEBUG
                        logger.debug("Debug Info:")
                        logger.debug("orig_images:", len(orig_images_pil))
                        logger.debug("embed_batch:", len(embed_batch))
                        logger.debug("text_batch:", len(text_batch))
                        logger.debug("sample_images:", len(sample_images))
                        assert len(orig_images_pil) == len(text_batch) == len(sample_images), "Mismatch in sample data lengths!"
                        
                        for j in range(len(orig_images_pil)):
                            orig_img = orig_images_pil[j]
                            prompt = text_batch[j]
                            gen_img = sample_images[j]
                            image_path = test_ds.df.iloc[j]["image_path"]
                            logger.debug(f"Sample {j}: Image Path: {image_path}, Prompt: {prompt}")
                            
                            grid_image = create_grid_image(orig_img, prompt, gen_img)
                            image_save_path = os.path.join(sample_images_dir, f"iter{i:06d}-{j:05d}.png")
                            grid_image.save(image_save_path)

                            logging.log_images(grid_image, prompt, iteration=i, index=j)

                except Exception as e:
                    logger.error(f"Error generating samples at iteration {i}: {e}")
            
            checkpoint_path = model_checkpoint_path(i)
            trainer.save(checkpoint_path)
            logger.info(f"Saved checkpoint at iteration {i}: {checkpoint_path}")
            cfg.conductor["trainer"]["PATH_MODEL_CHECKPOINT"] = checkpoint_path
            clear_old_checkpoints(i, checkpoint_dir, max_checkpoints=cfg.conductor.validation.max_checkpoints)
            last_checkpoint_i = i       

    trainer.save(model_save_path)
    logger.info(f"Training completed! Final model saved at: {model_save_path}")
    cfg.conductor["trainer"]["PATH_MODEL_LOAD"] = model_save_path
    
    if last_checkpoint_i is not None:
        last_checkpoint_path = model_checkpoint_path(last_checkpoint_i)
        if os.path.exists(last_checkpoint_path):
            os.remove(last_checkpoint_path)

    logging.close()
    
if __name__ == "__main__":
    main()