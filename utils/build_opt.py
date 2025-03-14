import os, sys
import yaml
import pathlib
import logging
import logging.config

from omegaconf import OmegaConf

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

def update_config_recursively(config, updates, parent_key=""):
    for key, value in updates.items():
        full_key = f"{parent_key}.{key}" if parent_key else key

        if isinstance(value, dict) and key in config:
            update_config_recursively(config[key], value, full_key)
        else:
            if key not in config:
                print(f"Adding new key {full_key}: {value}")  
            config[key] = value

class Opt:
    def __init__(self, project_name="MultiphaseBioImgGen", args=None):
        #
        self.project_root = get_project_root(project_name=project_name)
        
        #
        log_file = os.path.join(self.project_root, "asset", "logs", "setup.log")
        self.logger = _get_logger_(log_file_path=log_file)
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug(f"Logger started")
        
        # 
        config_paths = [
            os.path.join(self.project_root, "config", "model", "imagen.yaml"),
            os.path.join(self.project_root, "config", "model", "elucidated_imagen.yaml"),
            os.path.join(self.project_root, "config", "dataset.yaml"),
            os.path.join(self.project_root, "config", "conductor.yaml")
        ]
        
        configs = [OmegaConf.load(path) for path in config_paths]
        self.config = OmegaConf.merge(*configs)
        self.logger.debug(f"All configs loaded and merged")
        
        if args:
            args_dict = {k: v for k, v in vars(args).items() if v is not None}
            
            update_config_recursively(self.config, args_dict)
                
            self.logger.debug("Config successfully overwritten with args!")  
            
        self.args = args  