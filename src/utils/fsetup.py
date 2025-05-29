import os
import time
import wandb

from datetime import datetime
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

def folder_setup(cfg):
    dataset_name = cfg.data.name
    model_name = cfg.model.name
    mode = cfg.conductor.mode
    unet = cfg.conductor.unet
    
    if model_name in ['imagen', 'elucidated_imagen']:
        if model_name == 'imagen':
            cond_drop_prob = cfg.model.imagen.cond_drop_prob
            if unet == 1:
                dim = cfg.model.imagen['unets'][0]['dim']
            elif unet == 2:
                dim = cfg.model.imagen['unets'][1]['dim']
        elif model_name == 'elucidated_imagen':
            cond_drop_prob = cfg.model.elucidated_imagen.cond_drop_prob
            if unet == 1:
                dim = cfg.model.elucidated_imagen['unets'][0]['dim']
            elif unet == 2:
                dim = cfg.model.elucidated_imagen['unets'][1]['dim']
        sub_path = f"{dataset_name}_{model_name}/unet{unet}/dim{dim}/cond_drop_prob{cond_drop_prob}"
    else:
        sub_path = f"{dataset_name}_{model_name}"

    run_name = sub_path.replace('/', '_')
    run_name = f"{mode}_{sub_path}"
    run_dir = f"results/{mode}/{sub_path}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_name

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class LogExporter:
    def __init__(self, cfg, run_dir, run_name):
        self.cfg = cfg
        self.log = {}
        self.counts = {}
        self.iteration = 0

        cfg_flat = OmegaConf.to_container(self.cfg, resolve=True)

        self.run = None
        self.writer = None

        if self.cfg.conductor.wandb.usage:
            self.run = wandb.init(
                project=self.cfg.conductor.wandb.wandb_proj,
                name=run_name,
                entity=self.cfg.conductor.wandb.wandb_entity,
                config=cfg_flat,
                force=True
            )
        if self.cfg.conductor.tensorboard.usage:
            self.writer = SummaryWriter(run_dir)
            flat_cfg = flatten_dict(cfg_flat)
            self.writer.add_text(
                'hyperparameters',
                '|param|value|\n|-|-|\n' + '\n'.join([f"|{k}|{v}|" for k, v in flat_cfg.items()])
            )
    
    def __call__(self, key, value):
        if key in self.log:
            self.log[key] += value
            self.counts[key] += 1
        else:
            self.log[key] = value
            self.counts[key] = 1
    
    def update_wandb(self):
        if self.run:
            for log_key in self.log_avg:
                self.run.log({log_key: self.log_avg[log_key]}, step=self.iteration)

    def update_board(self):
        if self.writer:
            for log_key in self.log_avg:
                self.writer.add_scalar(log_key, self.log_avg[log_key], self.iteration)
                
    def reset_epoch(self):
        self.log = {}
        self.counts = {}

    def reset(self):
        self.reset_epoch()
        self.iteration = 0

    def step(self, iteration):
        self.iteration = iteration
        self.log_avg = {
            log_key: log_value / self.counts.get(log_key, 1)
            for log_key, log_value in self.log.items()
        }

        self.update_wandb()
        self.update_board()
        self.reset_epoch()
    
    def watch(self, model):
        if self.run:
            print(f"train_batch: {self.cfg.data['num_train_batch']}")
            self.run.watch(
                models=model, log='all', log_graph=True,
                log_freq=self.cfg.data['num_train_batch']
            )

    def log_images(self, grid_image, prompt):
        if self.run:
            self.run.log(
                {
                    f"Generated/{prompt}": wandb.Image(grid_image, caption=f"Iteration {self.iteration}"),
                },
                step=self.iteration
            )

    def close(self):
        if self.run:
            self.run.finish()
        if self.writer:
            self.writer.close()