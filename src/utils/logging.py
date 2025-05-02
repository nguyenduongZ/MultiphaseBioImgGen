import wandb

from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

class Logging:
    def __init__(self, cfg):
        self.__cfg = cfg
        self.__epoch = 0 
        self.__run = None
        self.__writer = None

        if self.__cfg.utils["wandb"]["usage"]:
            if self.__cfg.models["name"] in ["imagen", "elucidated_imagen"]:
                if self.__cfg.conductor["mode"] == "training":
                    self.__cfg.utils["wandb"]["run_name"] = f"training_{self.__cfg.datasets['name']}_{self.__cfg.models['name']}_unet{self.__cfg.conductor['unet_number']}_condscale{self.__cfg.conductor['cond_scale']}"
                elif self.__cfg.conductor["mode"] == "testing":
                    self.__cfg.utils["wandb"]["run_name"] = f"testing_{self.__cfg.datasets['name']}_{self.__cfg.models['name']}_unet{self.__cfg.conductor['unet_number']}_condscale{self.__cfg.conductor['cond_scale']}"
            else:
                if self.__cfg.conductor["mode"] == "training":
                    self.__cfg.utils["wandb"]["run_name"] = f"training_{self.__cfg.datasets['name']}_{self.__cfg.models['name']}"
                elif self.__cfg.conductor["mode"] == "testing":
                    self.__cfg.utils["wandb"]["run_name"] = f"testing_{self.__cfg.datasets['name']}_{self.__cfg.models['name']}"
            
            self.__run = wandb.init(
                project=self.__cfg.utils["wandb"]["wandb_prj"],
                name=self.__cfg.utils["wandb"]["run_name"],
                entity=self.__cfg.utils["wandb"]["entity"],
                config=OmegaConf.to_container(self.__cfg, resolve=True),
                resume="allow"
            )
            self.__run.define_metric("*", step_metric="global_step")

            # config = self.__run.config
            # if "cond_scale" in config:
            #     self.__cfg.conductor["cond_scale"] = config["cond_scale"]
            # if "unet_number" in config:
            #     self.__cfg.conductor["unet_number"] = config["unet_number"]
            # if "batch_size" in config:
            #     self.__cfg.conductor["trainer"]["batch_size"] = config["batch_size"]
            # if "idx" in config:
            #     self.__cfg.conductor["trainer"]["idx"] = config["idx"]
            # if "PATH_MODEL_LOAD" in config:
            #     self.__cfg.conductor["trainer"]["PATH_MODEL_LOAD"] = config["PATH_MODEL_LOAD"]
            # if "wandb_usage" in config:
            #     self.__cfg.utils["wandb"]["usage"] = config["wandb_usage"]
            # if "log_usage" in config:
            #     self.__cfg.utils["log"]["usage"] = config["log_usage"]
                    
        if self.__cfg.utils["log"]["usage"]:
            self.__writer = SummaryWriter(self.__cfg.utils["log"]["exp_dir"])
            cfg_dict = OmegaConf.to_container(self.__cfg, resolve=True)
            table_lines = ["| Param | Value |", "|-|-|"]
            for key, value in cfg_dict.items():
                if isinstance(value, (str, int, float, bool)):
                    table_lines.append(f"| `{key}` | `{value}` |")
                elif isinstance(value, list):
                    short = ", ".join(str(v) for v in value[:3])
                    if len(value) > 3:
                        short += "..."
                    table_lines.append(f"| `{key}` | `[ {short} ]` |")
                else:
                    table_lines.append(f"| `{key}` | *(section)* |")
            self.__writer.add_text("Config Summary (Top Level)", "\n".join(table_lines))

            self.__writer.add_text("Full Config (YAML)", f"```yaml\n{OmegaConf.to_yaml(self.__cfg)}\n```")

    def __call__(self, key, value, step=None, metadata=None):
        if self.__run is None and self.__writer is None:
            return  

        if step is not None:
            self.__epoch = max(self.__epoch, step) 

        log_data = {key: value}
        if metadata:
            log_data.update(metadata)

        if self.__run:  
            log_data["global_step"] = self.__epoch
            self.__run.log(log_data)

        if self.__writer: 
            self.__writer.add_scalar(key, value, global_step=self.__epoch)

    def log_iteration(self, iteration):
        self.__epoch = max(self.__epoch, iteration)

    def watch(self, model):
        if self.__run:
            self.__run.watch(models=model, log="all", log_freq=self.__cfg.datasets["num_train_batch"], log_graph=True)

    def log_images(self, grid_image, prompt, iteration, index):
        step = max(self.__epoch, iteration)
        self.global_step = step

        if self.__run:
            self.__run.log(
                {
                    f"Generated/{prompt}": wandb.Image(grid_image, caption=f"Iter {iteration}, Index {index}"),
                    "global_step": step
                }
            )

    def close(self):
        if self.__run:
            self.__run.finish()

        if self.__writer:
            self.__writer.close()