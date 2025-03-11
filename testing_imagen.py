import os, sys
import json
import random
import torch
import numpy as np
import pandas as pd

from PIL import Image
from glob import glob
from datetime import datetime
from rich.progress import track
from cleanfid import fid as clean_fid
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from dataset import get_ds
from model import get_model
from utils import folder_setup, save_config, Logging, Opt

def save_image(images, text_batch: list, i: int, batch_size: int, unet_number: int, sample_folder: str):
    for j, image in enumerate(track(images)):
        image = image.cpu().permute(1, 2, 0)
        image = image.numpy() * 255
        image = Image.fromarray(image.astype(np.uint8))
        
        image_index = i * batch_size + j
        image_save_path = sample_folder + f"images/{image_index:05d}_u{unet_number}-{text_batch[j]}.png"
        image.save(image_save_path)
        
def rescale_tensor(x):
    y = 2 * (x - torch.min(x)) / (torch.max(x) - torch.min(x)) - 1
    
    return y

def check_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def gen_text(template: str, sample_quantity, patient_sex=None, patient_age=None, instance_number=None, slice_location=None, label=None):
    text_list = []
    
    for _ in range(sample_quantity):
        text = template
        
        replacements = {
            "{PatientSex}": patient_sex if patient_sex else "",
            "{PatientAge}": f"{patient_age} years old" if patient_age else "",
            "{InstanceNumber}": f"instance number {instance_number}" if instance_number else "",
            "{SliceLocation}": f"with slice location at {slice_location}mm" if slice_location else "",
            "{Label}": label if label else ""
        }
        
        for key, value in replacements.items():
            text = text.replace(key, value)
            
        text = text.replace(" ,", ",").replace("  ", " ").strip()
        text_list.append(text)
        
    return text_list

def text_2_image(
    opt: Opt, 
    sample_dl, 
    model, 
    device: torch.device,
    unet_number: int, 
    sample_quantity: int,
    save_samples: bool,
    save_image_tensors: bool,
    sample_folder: str,
    embed_shape: tuple,
    text_list=None
):
    now = f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        
    cond_scale = opt.conductor["testing"]["cond_scale"]
    sample_seed = opt.conductor["seed"]
    loss_weighting = opt.conductor["testing"]["loss_weighting"]
    ds = opt.dataset["data"]["ds"]
    model_type = opt.conductor["model"]["model_type"]
    lower_batch = opt.conductor['testing']['lower_batch']
    upper_batch = opt.conductor['testing']['upper_batch']
    
    real_image_save_path = os.path.join(sample_folder, f"{ds}_{unet_number}_{model_type}/real_images/cond_scale_{cond_scale}_{loss_weighting}/{sample_seed:02d}")
    check_exists(real_image_save_path)
    sample_image_save_path = os.path.join(sample_folder, f"{ds}_{unet_number}_{model_type}/sample_images/cond_scale_{cond_scale}_{loss_weighting}/{sample_seed:02d}")
    check_exists(sample_image_save_path)
    
    if text_list is None:
        if save_image_tensors or save_samples:
            for i, (image_batch, embed_batch, text_batch) in enumerate(track(sample_dl, description="Testing Progress")):
                if i < lower_batch or i > upper_batch:
                    continue
                
                if i >= int(sample_quantity / opt.conductor["trainer"]["batch_size"]):
                    break
                
                sample_image_bacth = model.trainer.sample(
                    text_embeds=embed_batch.cuda(),
                    return_pil_images=False,
                    stop_at_unet_number=unet_number,
                    cond_scale=cond_scale
                )
                
                print("Clamp Tensors")
                sample_image_bacth = torch.clamp(sample_image_bacth.to(device), min=0, max=1)
                real_image_batch = torch.clamp(image_batch.to(device), min=0, max=1)
                
                if save_image_tensors:
                    print("Save Tensors")
                    torch.save(sample_image_bacth, sample_image_save_path + f"{i:05d}_sample_image_batch.pt")
                    torch.save(real_image_batch, real_image_save_path + f"{i:05d}_real_image_batch.pt")
                    
                    with open(real_image_save_path + f"{i:05d}_text_batch.json", "w") as f:
                        json.dump(text_batch, f)
                        
                if save_samples:
                    save_image(
                        images=real_image_batch,
                        text_batch=text_batch,
                        batch_size=embed_shape[0],
                        unet_number=unet_number,
                        sample_folder=real_image_save_path,
                        i=i
                    )
                    
                    save_image(
                        images=sample_image_bacth,
                        text_batch=text_batch,
                        batch_size=embed_shape[0],
                        unet_number=unet_number,
                        sample_folder=sample_image_save_path,
                        i=i
                    )
        
        if opt.conductor["testing"]["FrechetInceptionDistance"]["usage"]:         
            fid = FrechetInceptionDistance(**opt.conductor["testing"]["FrechetInceptionDistance"]["params"])
            print("FID initialized")
            
        if opt.conductor["testing"]["KernelInceptionDistance"]["usage"]: 
            kid = KernelInceptionDistance(**opt.conductor["testing"]["KernelInceptionDistance"]["params"])
            print("KID initialized")
        
        if opt.conductor["testing"]["LearnedPerceptualImagePatchSimilarity"]["usage"]:
            lpips = LearnedPerceptualImagePatchSimilarity(**opt.conductor["testing"]["LearnedPerceptualImagePatchSimilarity"]["params"])
            lpips_batch_score_list = []
            print("lpips initialized")
        
        if opt.conductor["testing"]["FrechetInceptionDistance"]["usage"] or opt.conductor["testing"]["KernelInceptionDistance"]["usage"] or opt.conductor["testing"]["LearnedPerceptualImagePatchSimilarity"]["usage"]:
            print(f"Start updating FID / KID")
            
            real_image_path_list = glob(real_image_save_path + "*_real_image_batch.pt")
            sample_image_path_list = glob(sample_image_save_path + "_sample_image_batch.pt")
            
            for real_path, sample_path in track(zip(real_image_path_list, sample_image_path_list), total=len(real_image_path_list)):
                real_image_batch = torch.load(real_path).to(device)
                sample_image_bacth = torch.load(sample_path).to(device)
                
                # Update FID
                if opt.conductor["testing"]["FrechetInceptionDistance"]["usage"]:
                    fid.update(real_image_batch, real=True)
                    fid.update(sample_image_bacth, real=False)
                
                # Update KID
                if opt.conductor["testing"]["KernelInceptionDistance"]["usage"]:
                    kid.update(real_image_batch, real=True)
                    kid.update(sample_image_bacth, real=False)
                    
                # Update lpips
                if opt.conductor["testing"]["LearnedPerceptualImagePatchSimilarity"]["usage"]:
                    real_image_batch = rescale_tensor(real_image_batch)
                    sample_image_bacth = rescale_tensor(sample_image_bacth)
                    lpips_batch_score_list.append(lpips(real_image_batch, sample_image_bacth).tolist())
        
        # Compute FID
        if opt.conductor["testing"]["FrechetInceptionDistance"]["usage"]:
            fid_result = fid.compute().cpu().item()
            print(f"fid_result: {fid_result}")
        
        else:
            fid_result = None
            
        # Compute KID
        if opt.conductor["testing"]["KernelInceptionDistance"]["usage"]:
            kid_mean, kid_std = kid.compute()
            kid_mean, kid_std = (kid_mean.cpu().item(), kid_std.cpu().item())
            print(f"kid_result: {kid_mean}, {kid_std}")
            
        else:
            kid_mean, kid_std = [None, None]
            
        #
        if opt.conductor["testing"]["CleanFID"]["usage"]:
            fdir1 = os.path.join(real_image_save_path, "images")
            fdir2 = os.path.join(sample_image_save_path, "images")
            clean_fid_score = clean_fid.compute_fid(fdir1=fdir1, fdir2=fdir2, **opt.conductor["testing"]["CleanFID"]["params"])
            print(f"clean_fid_score: {clean_fid_score}")
            
        else:
            clean_fid_score = None
            
        # FCD
        if opt.conductor["testing"]["FrechetCLIPDistance"]["usage"]:
            fdir1 = os.path.join(real_image_save_path, "images")
            fdir2 = os.path.join(sample_image_save_path, "images")
            fcd_score = clean_fid.compute_fid(fdir1=fdir1, fdir2=fdir2, **opt.conductor["testing"]["FrechetCLIPDistance"]["params"])
            print(f"clean_fid_score: {fcd_score}")
            
        else:
            fcd_score = None
            
        # LPIPS
        if opt.conductor["testing"]["LearnedPerceptualImagePatchSimilarity"]["usage"]:
            lpips_mean = np.mean(lpips_batch_score_list)
            print(f"lpips_mean: {lpips_mean}")
            
            pd.DataFrame({
                "cond_scale": cond_scale,
                "Dataset": ds,
                "model_type": model_type,
                "lpips": lpips_batch_score_list
            }).to_json(sample_image_save_path + now + "_lpips.json")
            
        else:
            lpips_mean = None
            
        results = pd.DataFrame({
            "cond_scale": cond_scale,
            "Dataset": ds,
            "model_type": model_type,
            "FID": [fid_result],
            "KID_mean": [kid_mean],
            "KID_std": [kid_std],
            "Clean_FID": [clean_fid_score],
            "FCD": [fcd_score],
            "LPIPS": [lpips_mean]
        })
        
        results.to_csv(sample_image_save_path + now + "_result.csv")
        
        return results

    else:
        batch_size = opt.conductor['trainer']['batch_size']
        
        for i in track(range(int(np.ceil(len(text_list) / 100)))):
            print(f'Iterator: {i}')
            
            if i < lower_batch or i > upper_batch:
                continue
            
            if (i + 1) * batch_size >= len(text_list):
                text_list_batch = text_list[i * batch_size :]
                
            else:
                text_list_batch = text_list[i * batch_size : (i + 1) * batch_size]
                
            sample_images = model.trainer.sample(
                texts=text_list_batch,
                return_pil_images=True,
                stop_at_unet_number=unet_number,
                cond_scale=cond_scale
            )
            
            for j, sample_image in enumerate(sample_images):
                image_index = batch_size * i + j
                image_save_path_dir = os.path.join(sample_image_save_path, f"evaluation/{text_list[image_index]}")
                
                if not os.path.exists(image_save_path_dir):
                    os.makedirs(image_save_path_dir, exist_ok=True)
                    
                image_save_path = os.path.join(image_save_path_dir, f"{image_index:05d}-{model_type}-{text_list[image_index]}.png")
                sample_image.save(image_save_path)
        
        return None

def testing_imagen(opt: Opt):   
    sample_quantity = opt.conductor["testing"]["sample_quantity"]
    save_samples = opt.conductor["testing"]["save_samples"]
    save_image_tensors = opt.conductor["testing"]["save_image_tensors"]
    
    if opt.conductor["testing"]["text"] != "":
        text = opt.conductor["testing"]["text"]
        text_list=[text] * sample_quantity
    
    else: 
        text_list = gen_text(
            template="CT scan of a male, 56 years old, instance number 1224, with slice location at 366.9mm, in the non-contrast phase.",
            sample_quantity=sample_quantity,
            patient_sex=["male", "female"],
            patient_age=[59],
            instance_number=[1139], # 1 to 2000
            slice_location=[333.4], 
            label=["non-contrast", "arterial", "venous"]
        )
        
    # Data setup  
    sample_ds, sample_dl = get_ds(opt, testing=True)
    _, sample_text_embed, _ = sample_ds.__getitem__(idx=0)
    
    # Device setup
    idx = opt.conductor["trainer"]["idx"]
    torch.cuda.set_device(idx)
    device = torch.device(f"cuda:{idx}" if torch.cuda.is_available() else "cpu")
    
    # Model setup
    model_type = opt.conductor["model"]["model_type"]
    model = get_model(opt=opt, device=device, testing=True)
    unet_number = opt.conductor["testing"]["unet_number"]
    
    # Folder setup and save setting
    opt.args.exp_dir = folder_setup(opt)
    save_config(opt, opt.args.exp_dir)

    results =  text_2_image(
        opt=opt,
        sample_dl=sample_dl,
        model=model,
        device=device,
        unet_number=unet_number,
        sample_quantity=sample_quantity,
        save_samples=save_samples,
        save_image_tensors=save_image_tensors,
        sample_folder=opt.args.exp_dir,
        embed_shape=sample_text_embed.shape,
        text_list=text_list
    )
    
    if results is not None:
        print(f'FRECHET INCEPTION DISTANCE (FID): {results["FID"]}')
        print(f'KERNEL INCEPTION DISTANCE (KID): mean {results["KID_mean"]} | std {results["KID_std"]}')
        print(f'CLEAN - FRECHET INCEPTION DISTANCE (clean-fid): {results["Clean_FID"]}')
        print(f'FRECHET CLIP DISTANCE (FCD): {results["FCD"]}') 