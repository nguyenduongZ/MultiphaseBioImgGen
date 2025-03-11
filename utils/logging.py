import os
import time
import yaml
import wandb
import json
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from .build_opt import Opt

class Logging:
    def __init__(self, opt: Opt):
        self.__opt = opt
        self.__args = opt.args
        self.__log =  {}
        self.__epoch = 0
        self.__run = None
        self.__writer = None
        
        if self.__args.wandb:
            self.__args.run_name = f"{self.__args.ds}_{self.__args.model_type}_unet{self.__args.unet_number}_condscale{self.__args.cond_scale}__{int(time.time())}"
            self.__run = wandb.init(
                project=self.__args.wandb_prj,
                # entity=self.__args.wandb_entity,
                name=self.__args.run_name,
                config=self.__args,
                force=True
            )
            
            self.__run.define_metric("*", step_metric="global_step", step_sync=True)
            
        if self.__args.log:
            self.__writer = SummaryWriter(self.__args.exp_dir)
            self.__writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.__args).items()])),
            )
            
    def __call__(self, key, value, step=None, metadata=None):
        if step is not None:
            step = max(self.__epoch, step)  
            self.__epoch = step 

        log_data = {key: value}
        if metadata:
            log_data.update(metadata)

        if self.__run:
            log_data["global_step"] = self.__epoch 
            self.__run.log(log_data, step=self.__epoch, commit=False)

        if self.__writer:
            self.__writer.add_scalar(key, value, global_step=self.__epoch)
            
    def log_iteration(self, iteration):
        self.__epoch = max(self.__epoch, iteration) 
        if self.__run:
            self.__run.log({"global_step": self.__epoch}, step=self.__epoch, commit=True)
        
    def watch(self, model):
        if self.__run:
            self.__run.watch(models=model, log="all", log_freq=self.__args.num_train_batch, log_graph=True) 
            
    def log_images(self, images_dict, iteration):
        if not self.__run:
            return
        
        iteration = max(self.__epoch, iteration) 
        log_dict = {"global_step": iteration}

        for prompt, data in images_dict.items():
            generated_image = data.get("Generated Image", None)
            if generated_image is not None:
                log_dict[f"Generated/{prompt}"] = wandb.Image(generated_image, caption=f"Step {iteration}")

        self.__run.log(log_dict, step=iteration)         
        
    def close(self):
        if self.__run:
            self.__run.finish() 
            
        if self.__run:
            self.__writer.close() 