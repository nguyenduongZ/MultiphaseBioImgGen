import os, sys
import yaml
import pathlib
import logging
import logging.config

def get_project_root(project_name: str, marker_dirs=("asset", "dataset", ".git")):
    path = pathlib.Path.cwd()
    
    while path.name != project_name:
        if path.parent == path:
            break
        path = path.parent
        
    if path.name == project_name:
        return str(path)
    
    path = pathlib.Path.cwd()
    while path != path.parent:
        if any((path / marker).is_dir() for marker in marker_dirs):
            return str(path)
        path = path.parent
        
    return None

def _get_config_(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find config file: {path}")
    
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
            
    except yaml.YAMLError as e:
        raise RuntimeError(f"Error parsing YAML file '{path}': {e}")
    
    return config

def _get_logger_(log_file_path: str, log_level=logging.INFO):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    logger = logging.getLogger("Logger")
    
    if logger.hasHandlers():
        return logger
    
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_format = "%d/%m/%Y %I:%M:%S %p"
    
    logging.basicConfig(level=log_level, format=log_format, datefmt=date_format, filename=log_file_path, filemode="w")
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(console_handler)
    
    return logger

class Opt:
    def __init__(self, project_name="MultiphaseBioImgGen", args=None):
        #
        self.project_root = get_project_root(project_name=project_name)
        
        #
        log_file = os.path.join(self.project_root, "asset", "logs", "setup.log")
        self.logger = _get_logger_(log_file_path=log_file)
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug(f"Logger started")
        
        # Model
        self.imagen = _get_config_(os.path.join(self.project_root, "config", "model", "imagen.yaml"))
        self.elucidated_imagen = _get_config_(os.path.join(self.project_root, "config", "model", "elucidated_imagen.yaml"))
        self.logger.debug(f"Model configs loaded")
        
        # Dataset
        self.dataset = _get_config_(os.path.join(self.project_root, "config", "dataset.yaml"))
        self.logger.debug(f"Dataset configs loaded")
        
        # Deploying
        self.conductor = _get_config_(os.path.join(self.project_root, "config", "conductor.yaml"))
        
        if args:
            args_dict = vars(args).copy()
            args_dict.pop("config", None)

            mapping = {
                "batch_size": ("trainer", "batch_size"),
                "num_workers": ("trainer", "num_workers"),
                "pin_memory": ("trainer", "pin_memory"),
                "multi_gpu": ("trainer", "multi_gpu"),
                "idx": ("trainer", "idx"),
                "iterations": ("trainer", "iterations"),
                "unet_number": ("trainer", "unet_number"),
                
                "validate_model": ("validation", "interval", "validate_model"),
                "valid_loss": ("validation", "interval", "valid_loss"),
                "cond_scale_valid": ("validation", "cond_scale"),
                "cond_scale_test": ("testing", "cond_scale"),
                
                "ds": ("dataset", "ds"),
                
                "wandb_prj": ("wandb", "wandb_prj"),
                "wandb_entity": ("wandb", "wandb_entity"),
            }

            for arg_key, yaml_path in mapping.items():
                arg_value = args_dict.get(arg_key, None)
                
                if arg_value is not None and arg_value != -1:
                    if not yaml_path: 
                        raise ValueError(f"Invalid YAML path for argument {arg_key}: {yaml_path}")

                    if yaml_path[0] == "dataset":
                        config_target = self.dataset
                    else:
                        config_target = self.conductor
                    
                    yaml_path = yaml_path[1:]

                    if not yaml_path: 
                        raise ValueError(f"Invalid YAML path for argument {arg_key}: {yaml_path}")            
                    
                    config_section = config_target
                    for key in yaml_path[:-1]:  # Traverse except the last key
                        config_section = config_section.setdefault(key, {})
                    
                    # Update the final key
                    config_section[yaml_path[-1]] = arg_value
                    self.logger.info(f"Overriding {'.'.join(yaml_path)}: {arg_value}")

            # Log final config after merging
            self.logger.info("Config successfully overwritten with args!")
            
        self.args = args
        