import os, sys
sys.path.append(os.path.abspath(os.curdir))

import hydra
import logging
import configparser
import logging.config

from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from logging.handlers import RotatingFileHandler

class MasterLogger:
    def __init__(self, cfg: DictConfig):
        utils_cfg = cfg.utils
        logger_name = utils_cfg.name
        level = getattr(logging, utils_cfg.level.upper(), logging.INFO)

        run_dir = os.path.dirname(HydraConfig.get().run.dir)
        log_dir = os.path.join(run_dir, "logs")
        utils_cfg["log"]["exp_dir"] = log_dir
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, "training.log")
        
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S"
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        file_handler = RotatingFileHandler(
            log_file_path, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)
        self.logger.handlers = []
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
    def get_logger(self) -> logging.Logger:
        return self.logger