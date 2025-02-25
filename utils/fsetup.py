import os, sys
import json
import shutil

from datetime import datetime

def folder_setup(args):
    run_dir = os.getcwd() + f"/asset/results/{args.mode}"
    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)
    
    data_name = args.ds
    unet_number = f"u{args.unet_number}"
    model_name = args.model_type        
    run_name = f"{data_name}_{unet_number}_{model_name}"
    
    task_dir = os.path.join(run_dir, run_name)
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)
        
    exp_dirs = [d for d in os.listdir(task_dir) if d.startswith("exp_")]
    exp_indices = [int(d.split("_")[1]) for d in exp_dirs if d.split("_")[1].isdigit()]

    if not exp_indices:
        exp_cnt = 0
    else:
        exp_cnt = max(exp_indices) + 1
    
    exp_dir = os.path.join(task_dir, f"exp_{exp_cnt}")

    if exp_indices:
        last_exp_dir = os.path.join(task_dir, f"exp_{max(exp_indices)}")
        cfg_path = os.path.join(last_exp_dir, "config.json")

        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                data = json.load(f)
            if data.get("test", False):
                shutil.rmtree(last_exp_dir)
                exp_dir = last_exp_dir

    os.makedirs(exp_dir, exist_ok=True)

    return exp_dir

def save_cfg(args, exp_dir = None):
    config_dict = vars(args)

    if not exp_dir:
        raise ValueError('exp_dir cannot be None')
    else:
        path = exp_dir + "/config.json"

        with open(path, "w") as outfile: 
            json.dump(config_dict, outfile)

def save_json(dct, path):
    with open(path, "w") as outfile: 
        json.dump(dct, outfile)