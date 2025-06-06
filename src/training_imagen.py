import os
import time
import logging
import torchvision.transforms as T

from tqdm import tqdm

from src.utils import create_grid_image, LogExporter

def train_imagen(cfg, model, device, train_dl, valid_dl, run_dir, run_name):
    log_exporter = LogExporter(cfg, run_dir, run_name)
    log_exporter.watch(model=model.imagen_model, log_freq=len(train_dl))
    trainer =  model.trainer
    
    trainer.add_train_dataloader(dl=train_dl)
    trainer.add_valid_dataloader(dl=valid_dl)

    #
    dataset_name = cfg.data.name
    unet = cfg.conductor.unet
    cond_scale = cfg.conductor.cond_scale
    
    # Folder setup
    checkpoint_path = os.path.join(run_dir, f"checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    sample_images_path = os.path.join(run_dir, f"samples_condscale{cond_scale}")
    os.makedirs(sample_images_path, exist_ok=True)

    model_checkpoint_file = f"checkpoint-iter{{i:06d}}.pt"
    model_checkpoint_path = lambda i: os.path.join(checkpoint_path, model_checkpoint_file.format(i=i))
    model_final_save_path = os.path.join(checkpoint_path, "checkpoint-final.pt")
    
    #
    logging.info(f"Starting Imagen training for {dataset_name}")
    iterations = cfg.conductor.iterations
    pbar = tqdm(range(iterations), desc="Training", ncols=100)
    valid_loss = None

    total_train_time = 0.0
    total_sample_time = 0.0
    sample_count = 0
    
    for i in pbar:
        start_train = time.time()
        loss = trainer.train_step(unet_number=unet)
        end_train = time.time()
        train_step_time = end_train - start_train
        total_train_time += train_step_time
        log_exporter(key='train/loss', value=loss)
        log_exporter(key='train/time_per_step', value=train_step_time)

        if not (i % cfg.conductor.validate_at_every) and i > 0 and trainer.is_main:
            valid_loss = trainer.valid_step(unet_number=unet)
            log_exporter(key='valid/loss', value=valid_loss)
            # logging.info(f"[Iteration {i}] Train Loss: {loss:.4f} | Valid Loss: {valid_loss:.4f}")
        
        # Sample    
        if not (i % cfg.conductor.sample_at_every) and i > 0 and trainer.is_main:
            start_sample = time.time()
            if cfg.conductor.display_samples:
                try:
                    orig_images, embed_batches, text_batches = next(iter(valid_dl))
                    sample_images = trainer.sample(
                        text_embeds=embed_batches.to(device),
                        return_pil_images=True,
                        stop_at_unet_number=unet,
                        use_tqdm=False,
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
            
            end_sample = time.time()
            sample_time = end_sample - start_sample
            total_sample_time += sample_time
            sample_count += 1
            log_exporter(key='sample/time', value=sample_time)
                    
        if not (i % cfg.conductor.save_at_every) and i > 0 and trainer.is_main:
            checkpoint_file = model_checkpoint_path(i)
            trainer.save(checkpoint_file)
            logging.info(f"Saved checkpoint at iteration {i}: {checkpoint_file}")

        avg_train_time = total_train_time / (i+1)
        avg_sample_time = total_sample_time / sample_count if sample_count > 0 else 0
        pbar.set_postfix({
            'Train Loss': f"{loss:.4f}",
            'Valid Loss': f"{valid_loss:.4f}" if valid_loss is not None else "N/A",
            'Avg Train Time (s)': f"{avg_train_time:.4f}",
            'Avg Sample Time (s)': f"{avg_sample_time:.4f}" if sample_count > 0 else "N/A"
        })
        
        log_exporter.step(iteration=i)
        
    trainer.save(model_final_save_path)
    logging.info(f"Saved final checkpoint: {model_final_save_path}")

    logging.info(f"Total training time: {total_train_time:.4f} seconds for {iterations} iterations")
    logging.info(f"Average training time per step: {total_train_time / iterations:.4f} seconds")
    if sample_count > 0:
        logging.info(f"Total sampling time: {total_sample_time:.4f} seconds over {sample_count} sampling calls")
        logging.info(f"Average sampling time per sample: {total_sample_time / sample_count:.4f} seconds")
    else:
        logging.info("No sampling steps were executed during training.")

    log_exporter.close()