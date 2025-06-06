import os
import wandb

from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

def cfg_to_name(cfg):
    model_name = cfg.model.name
    unet_idx = cfg.conductor.unet - 1
    unet_cfg = cfg.model[model_name]['unets'][unet_idx]
    
    dim = unet_cfg['dim']
    cond_dim = unet_cfg.get('cond_dim')
    if cond_dim is None:
        cond_dim = dim

    mults = '-'.join(map(str, unet_cfg['dim_mults']))
    res = unet_cfg['num_resnet_blocks']
    attn = ''.join(['1' if x else '0' for x in unet_cfg['layer_attns']])
    xattn = ''.join(['1' if x else '0' for x in unet_cfg['layer_cross_attns']])
    return f'dim{dim}_cond{cond_dim}_mults{mults}_res{res}_attn{attn}_xattn{xattn}'

def folder_setup(cfg):
    dataset_name = cfg.data.name
    model_name = cfg.model.name
    mode = cfg.conductor.mode
    unet = cfg.conductor.unet
    cond_drop_prob = cfg.model[model_name]['cond_drop_prob']
    
    if model_name in ['imagen', 'elucidated_imagen']:
        sub_path = f"{dataset_name}_{model_name}/unet{unet}"
    else:
        sub_path = f"{dataset_name}_{model_name}"
    path = cfg_to_name(cfg)
    sub_path = os.path.join(sub_path, path, f"cond_drop_prob{cond_drop_prob}")
    
    run_name = f"{mode}_{sub_path}"
    run_name = run_name.replace('/', '_')
    
    run_dir = f"results/{mode}/{sub_path}"
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
                project=self.cfg.conductor.wandb.wandb_prj,
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
    
    def watch(self, model, log_freq):
        if self.run:
            self.run.watch(
                models=model, log='all', log_graph=True,
                log_freq=log_freq
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
    
    
    
    
    