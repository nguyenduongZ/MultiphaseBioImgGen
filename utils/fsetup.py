import os, sys
import json
import shutil
import pathlib

from .build_opt import Opt, get_project_root

def folder_setup(opt: Opt):
    args = opt.args
    
    project_root = get_project_root(args.wandb_prj)
    if project_root is None:
        raise RuntimeError("Project root directory not found!")
    
    run_dir = os.path.join(project_root, "asset", "results", args.mode)
    os.makedirs(run_dir, exist_ok=True)
    
    if args.model_type == "Imagen" or args.model_type == "ElucidatedImagen":
        run_name = f"{args.ds}_{args.model_type}_unet{args.unet_number}"
    else:
        run_name = f"{args.ds}_{args.model_type}"
        
    task_dir = os.path.join(run_dir, run_name)
    os.makedirs(task_dir, exist_ok=True)
    
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

def save_config(opt: Opt, exp_dir):
    args = opt.args
    
    if not exp_dir:
        raise ValueError("exp_dir cannot be None. Please provide a valid directory.")
    
    config_dict = vars(args)
    path = os.path.join(exp_dir, "config.json")
    
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=4)