import argparse
import torch
import numpy as np
import random
import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(prog="Multiphase Bio Image Generation")

    # Dataset
    parser.add_argument('--ds', type=str, required=True, choices=['vindr_multiphase'], 
                        help='Name of the dataset to use. Currently supports only "vindr_multiphase".')
    parser.add_argument('--metadata_path', type=str, default='/media/mountHDD1/quby/abdomen_phases_png/metadata.csv', 
                        help='Path to the metadata CSV file containing image paths, prompts, and labels.')
    parser.add_argument('--image_size', type=int, default=256, 
                        help='Size to which input images will be resized (height and width).')
    parser.add_argument('--bs', type=int, default=4, 
                        help='Batch size for data loading during training and validation.')
    parser.add_argument('--wk', type=int, default=0, 
                        help='Number of worker threads to use for data loading.')
    parser.add_argument('--pm', action='store_true', 
                        help='Enable pinned memory for DataLoader to speed up data transfer to GPU.')

    # Tokenizer
    parser.add_argument('--max_length', type=int, default=512, 
                        help='Maximum number of tokens to consider for text inputs.')
    parser.add_argument('--cache_dir', type=str, default='/media/mountHDD2/duong/MultiphaseBioImgGen/data/cache', 
                        help='Directory to store cached text embeddings for reuse.')
    parser.add_argument('--t5_model', type=str, default='t5-large', 
                        help='Name of the pre-trained T5 model to use for text embedding.')
    
    # TRAINING
    parser.add_argument('--idx', type=int, default=0, 
                        help='device index used in training')
    parser.add_argument('--seed', type=int, default=0, 
                        help='seed used in training')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='number of epochs used in training')
    parser.add_argument('--cosine-decay-max-steps', type=int, default = 500000, 
                        help='The max steps in cosine learning rate scheduler')
    parser.add_argument('--num-channels', type=int, default = 3, 
                        help='The total number of channels in the data file')
    parser.add_argument('--train-unet', type=int, default = 1, 
                        help='When using cascaded diffusion, control which unet is currently training.')
    parser.add_argument('--task', type=str, default='generation', choices=['generation'],
                        help='training task')
    parser.add_argument('--model', type=str, default='imagen', choices=['imagen'], 
                        help='backbone used in training')
    parser.add_argument('--loss', type=str, default='', choices=[], 
                        help='loss function used in training')
    parser.add_argument('--test', action='store_true', 
                        help='toggle to say that this experiment is just flow testing')
    parser.add_argument('--valid_per_step_num', type=int, default = 3000, 
                        help='Run validation for every this number of train steps')
    
    # LOGGING
    parser.add_argument('--wandb', action='store_true',
        help='toggle to use wandb for online saving')
    parser.add_argument('--log', action='store_true',
        help='toggle to use tensorboard for offline saving')
    parser.add_argument('--wandb_prj', type=str, default="ISIC2018 Segmentation",
        help='toggle to use wandb for online saving')
    parser.add_argument('--wandb_entity', type=str, default="scalemind",
        help='toggle to use wandb for online saving')
    parser.add_argument('--neptune', action='store_true', 
        help='toggle to use Neptune.ai for logging')
    parser.add_argument('--neptune_prj', type=str, default="ISIC2018 Segmentation", 
        help='Neptune.ai project name')
    parser.add_argument('--neptune_api_token', type=str, required=False, 
        help='API token for Neptune.ai')

    from train_imagen import train_imagen
    args = parser.parse_args()

    train_imagen(args)