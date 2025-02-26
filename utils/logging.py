import os
import time
import yaml
import torch
import wandb
import numpy as np

from torch.utils.tensorboard import SummaryWriter

class Logging:
    def __init__(self, args):
        self.__log = {}
        self.__epoch = 0
        
        if args.wandb:
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)
            
            args.run_name = f"{args.ds}_u{args.unet_number}_{args.model_type}__{int(time.time())}"
            
            self.__run = wandb.init(
                project=args.wandb_prj,
                entity=args.wandb_entity,
                name=args.run_name,
                config=config,
                force=True
            )
            
        if args.log:
            self.__writer = SummaryWriter(args.exp_dir)
            self.__writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
            )
            
        self.__args = args
        
    def __call__(self, key, value):
        if key in self.__log:
            self.__log[key] += value
        
        else:
            self.__log[key] = value
            
    def __update_wandb(self):
        for log_key in self.__log_avg:
            self.__run.log({log_key: self.__log_avg[log_key]}, step=self.__epoch)
    
    def __update_board(self):
        for log_key in self.__log_avg:
            self.__writer.add_scalar(log_key, self.__log_avg[log_key], self.__epoch)
    
    def __reset_epoch(self):
        self.__log = {}
        
    def reset(self):
        self.__reset_epoch()
        self.__epoch = 0
        
    def step(self, epoch):
        self.__epoch = epoch
        
        self.__log_avg = {}
        for log_key in self.__log:
            if log_key.split("/")[-1] in ["loss", "FID", "KID_mean", "KID_std", "Clean_FID", "FCD", "LPIPS"]:
                if "train" in log_key:
                    self.__log_avg[log_key] = self.__log[log_key] / self.__args.num_train_batch
                    
                elif "valid" in log_key:
                    self.__log_avg[log_key] = self.__log[log_key] / self.__args.num_valid_batch

                elif "test" in log_key:
                    self.__log_avg[log_key] = self.__log[log_key] / self.__args.num_batch
                    
                else:
                    raise ValueError(f"key: {log_key} wrong format")
                
            else:
                if "train" in log_key:
                    self.__log_avg[log_key] = self.__log[log_key] / self.__args.num_train_sample
                    
                elif "valid" in log_key:
                    self.__log_avg[log_key] = self.__log[log_key] / self.__args.num_valid_sample
                    
                elif "test" in log_key:
                    self.__log_avg[log_key] = self.__log[log_key] / self.__args.num_sample
                
                else:
                    raise ValueError(f"key: {log_key} wrong format")
        
        if not (epoch % self.__args.valid_loss):                    
            if self.__args.wandb:
                self.__update_wandb()
                
            if self.__args.log:
                self.__update_board()
            
        self.__reset_epoch()
            
    def watch(self, model):
        self.__run.watch(models=model, log="all", log_freq=self.__args.num_train_batch, log_graph=True)
        
        
    def log_model(self):
        best_path = os.path.join(self.__args.exp_dir,  "best.pt")
        if os.path.exists(best_path):
            self.__run.log_model(path=best_path, model_name=f"{self.__args.run_name}-best-model")
        
        last_path = os.path.join(self.__args.exp_dir,  "last.pt")
        if os.path.exists(last_path):
            self.__run.log_model(path=last_path, model_name=f"{self.__args.run_name}-last-model")

    def log_images(self, table_data, epoch, validate_model):
        columns = ["Cond Scale", "Prompt", "Original Image"] + [
            f"Generated Image (Epoch {e})" for e in range(validate_model, epoch + 1, validate_model)
        ]
        
        table = wandb.Table(columns=columns)
        
        for prompt, data in table_data.items():
            row = [data["Cond Scale"], data["Prompt"], data["Original Image"]]      
            for col in columns[3:]:
                row.append(data.get(col, None))
            table.add_data(*row)
            
        self.__run.log({"Generated Samples": table})
            
    def close(self):
        if self.__writer is not None:
            self.__writer.close()
        if self.__run is not None:
            self.__run.finish()

    @property
    def log(self):
        return self._Logging__log
    
    @property
    def log_avg(self):
        return self._Logging__log_avg
    
    @property
    def epoch(self):
        return self._Logging__epoch
    
    @property
    def args(self):
        return self._Logging_args