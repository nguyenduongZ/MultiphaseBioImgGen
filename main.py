import os, sys
import yaml
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="MultiphaseBioImgGen")
    
    # Config file
    parser.add_argument("--config", type=str, default="./config/config.yaml", 
                        help="Path config.")
    
    parser.add_argument("--ds", type=str, default="VinDrMultiphase", choices=["VinDrMultiphase"], 
                        help="Data used in training.")
    parser.add_argument("--model_type", type=str, default="Imagen", choices=["Imagen", "ElucidatedImagen"], 
                        help="Model used in training.")

    # Training parameters
    parser.add_argument("--mode", type=str, required=True, choices=["training", "testing"],
                        help="Pipeline traning or testing.")
    parser.add_argument("--batch_size", type=int, default=-1, 
                        help="Batch size used for dataset.")
    parser.add_argument("--pin_memory", action="store_true", 
                        help="Toggle to pin memory in dataloader.")
    parser.add_argument("--num_workers", type=int, default=-1, 
                        help="Number of worker processor.")
    parser.add_argument("--idx", type=int, default=-1, 
                        help="Device index used in training.")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Seed used in training.")
    parser.add_argument("--unet_number", type=int, required=True, choices=[1, 2], 
                        help="Traning at unet number.")
    parser.add_argument('--epochs', type=int, default=-1, 
                        help='number of epochs used in training')

    # Test code 
    parser.add_argument('--test', action='store_true', 
                        help='toggle to say that this experiment is just flow testing')
    
    # Testing
    parser.add_argument("--validate_model", type=int, default=-1, 
                        help="validate_model")
    parser.add_argument("--cond_scale", type=int, default=-1, choices=range(1, 11), 
                        help="cond_scale")
    parser.add_argument("--valid_loss", type=int, default=-1, 
                        help="valid_loss")
    
    # LOGGING
    parser.add_argument("--wandb", action="store_true", 
                        help="Toggle to use wandb for online saving.")
    parser.add_argument("--log", action="store_true", 
                        help="Toggle to use tensorboard for offline saving.")
    parser.add_argument("--wandb_prj", type=str, default="MultiphaseBioImgGen", 
                        help="Toggle to use wandb for online saving.")
    parser.add_argument("--wandb_entity", type=str, default="scalemind", 
                        help="Toggle to use wandb for online saving.")

    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Function to update config
    def update_config(config, args):
        """ Update config with command-line arguments if they are set """
        mapping = {
            "batch_size": ("trainer", "batch_size"),
            "num_workers": ("trainer", "num_workers"),
            "pin_memory": ("trainer", "pin_memory"),
            "idx": ("trainer", "idx"),
            "epochs": ("trainer", "epochs"),
            "unet_number": ("trainer", "unet_number"),
            "validate_model": ("validation", "interval", "validate_model"),
            "cond_scale": [("validation", "cond_scale"), ("testing", "cond_scale")],
            "seed": ("trainer", "seed")
        }
        
        for key, value in vars(args).items():
            if value not in [None, -1]:
                if key in mapping:
                    paths = mapping[key]
                    
                    if isinstance(paths[0], tuple): 
                        for path in paths:
                            ref = config
                            for k in path[:-1]:
                                if k not in ref:
                                    ref[k] = {}
                                ref = ref[k]
                            ref[path[-1]] = value
                    
                    else:
                        ref = config
                        for k in paths[:-1]:
                            if k not in ref:
                                ref[k] = {}
                            ref = ref[k]
                        ref[paths[-1]] = value
                
    update_config(config, args)

    # Training
    if args.mode == "training":
        from train_imagen import train_imagen
        train_imagen(config, args)
        
    # Testing
    if args.mode == "testing":
        from testing_imagen import testing_imagen
        testing_imagen(config, args)