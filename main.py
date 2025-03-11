import os, sys
import yaml
import argparse

from utils import Opt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="MultiphaseBioImgGen")
    
    # 
    parser.add_argument("--ds", type=str, default="VinDrMultiphase", choices=["VinDrMultiphase"], 
                        help="Data used in training.")
    
    #
    parser.add_argument("--model_type", type=str, default="Imagen", choices=["Imagen", "ElucidatedImagen"], 
                        help="Model used in training.")

    # Training parameters
    parser.add_argument("--mode", type=str, required=True, choices=["training", "testing"],
                        help="Pipeline traning or testing.")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size used for dataset.")
    parser.add_argument("--pin_memory", action="store_true", 
                        help="Toggle to pin memory in dataloader.")
    parser.add_argument("--num_workers", type=int, default=-1, 
                        help="Number of worker processor.")
    parser.add_argument("--idx", type=int, default=-1, 
                        help="Device index used in training.")
    parser.add_argument("--multi_gpu", action="store_true", 
                        help="Multi GPU.")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Seed used in training.")
    parser.add_argument("--unet_number", type=int, required=True, choices=[1, 2], 
                        help="Traning at unet number.")
    parser.add_argument('--iterations', type=int, default=-1, 
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
    
    opt = Opt(args=args)

    # Training
    if args.mode == "training":
        from train_imagen import train_imagen
        train_imagen(opt=opt)
        
    # Testing
    if args.mode == "testing":
        from testing_imagen import testing_imagen
        testing_imagen(opt=opt)