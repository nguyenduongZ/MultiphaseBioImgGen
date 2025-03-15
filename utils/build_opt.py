import os
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

        if OmegaConf.select(config, full_key) is not None:
            OmegaConf.update(config, full_key, value, merge=True)
            print(f"Updated {full_key}: {value}")  
        else:
            if hasattr(config, "trainer") and key in config.trainer:
                OmegaConf.update(config.trainer, key, value, merge=True)
                print(f"Updated trainer.{key}: {value}")
            else:
                OmegaConf.update(config, key, value, merge=True)
                print(f"Added new key {key}: {value}")

class Opt:
    def __init__(self, project_name="MultiphaseBioImgGen", args=None):
        self.project_root = get_project_root(project_name=project_name)
        
        #
        log_file = os.path.join(self.project_root, "asset", "logs", "setup.log")
        self.logger = _get_logger_(log_file_path=log_file)
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug(f"Logger started")
        
        # Load YAML
        config_paths = [
            os.path.join(self.project_root, "config", "model", "imagen.yaml"),
            os.path.join(self.project_root, "config", "model", "elucidated_imagen.yaml"),
            os.path.join(self.project_root, "config", "dataset.yaml"),
            os.path.join(self.project_root, "config", "conductor.yaml")
        ]
        
        configs = [OmegaConf.load(path) for path in config_paths]
        self.config = OmegaConf.merge(*configs)  # Merge YAML
        self.logger.debug(f"All configs loaded and merged")

        # Override args
        if args:
            args_dict = {k: v for k, v in vars(args).items() if v is not None}
            
            update_config_recursively(self.config, args_dict)

            self.logger.debug(f"Config updated with args: {args_dict}")
        
        self.args = args  