import os
import torch
import logging
import torchvision.transforms as T

from collections import deque

from src.utils import create_grid_image

def train_imagen(cfg, model, device, train_dl, valid_dl, log_exporter, run_dir):
    log_exporter.watch(model.imagen_model)
    trainer = model.trainer
    trainer.add_train_dataloader(dl=train_dl)
    trainer.add_valid_dataloader(dl=valid_dl)
    
    #
    dataset_name = cfg.data.name
    unet = cfg.conductor.unet
    cond_scale = cfg.conductor.cond_scale
    max_batch_size = cfg.conductor.max_batch_size if 'max_batch_size' in cfg else 1
    
    # Folder setup
    checkpoint_path = './results/checkpoints'
    sample_images_path = os.path.join(run_dir, f"samples_condscale{cond_scale}")
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(sample_images_path, exist_ok=True)

    model_checkpoint_file = f"checkpoint-iter{{i:06d}}.pt"
    model_checkpoint_path = lambda i: os.path.join(checkpoint_path, model_checkpoint_file.format(i=i))
    model_final_save_path = os.path.join(checkpoint_path, "checkpoint-final.pt")
    
    checkpoint_queue = deque(maxlen=1)
    
    #
    logging.info(f"Starting Imagen training for {dataset_name}")
    iterations = cfg.conductor.iterations
    
    for i in range(iterations):
        loss = trainer.train_step(unet_number=unet, max_batch_size=max_batch_size)
        log_exporter(key='train/loss', value=loss)
        
        if not (i % cfg.conductor.validate_at_every) and i > 0 and trainer.is_main:
            valid_loss = trainer.valid_step(unet_number=unet, max_batch_size = max_batch_size)
            log_exporter(key='valid/loss', value=valid_loss)
            logging.info(f"[Iteration {i}] Train Loss: {loss:.4f} | Valid Loss: {valid_loss:.4f}")
            
        if not (i % cfg.conductor.save_at_every) and i > 0 and trainer.is_main:
            if cfg.conductor.display_samples:
                try:
                    orig_images, embed_batches, text_batches = next(iter(valid_dl))
                    sample_images = trainer.sample(
                        text_embeds=embed_batches.to(device),
                        return_pil_images=True,
                        stop_at_unet_number = unet,
                        cond_scale=cond_scale
                    )

                    to_pil = T.ToPILImage()
                    orig_images_pil = [to_pil(img) for img in orig_images]
                    
                    assert len(orig_images_pil) == len(text_batches) == len(sample_images), \
                        "Mismatch in sample data lengths!"
                        
                    for j in range(len(orig_images_pil)):
                        orig_img, prompt, gen_img = orig_images_pil[j], text_batches[j], sample_images[j]
                        grid_image = create_grid_image(orig_img, prompt, gen_img)
                        image_save_path = os.path.join(sample_images_path, f"iter{i:06d}-{j:05d}.png")
                        grid_image.save(image_save_path)
                        log_exporter.log_images(grid_image, prompt)
                except Exception as e:
                    logging.error(f"Error generating samples at Iteration {i}: {e}")
                    
        if not (i % cfg.conductor.save_at_every) and i > 0 and trainer.is_main:
            checkpoint_file = model_checkpoint_path(i)
            trainer.save(checkpoint_file)
            
            # Remove previous checkpoint if needed
            if len(checkpoint_queue) == checkpoint_queue.maxlen:
                old_checkpoint = checkpoint_queue.popleft()
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
                    logging.info(f"Removed old checkpoint: {old_checkpoint}")
            
            checkpoint_queue.append(checkpoint_file)
            
        log_exporter.step(iteration=i)
        
    trainer.save(model_final_save_path)
    logging.info(f"Saved final checkpoint: {model_final_save_path}")
    
    log_exporter.close()