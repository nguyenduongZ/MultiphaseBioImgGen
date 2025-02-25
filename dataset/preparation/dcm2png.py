import os, sys
import pydicom
import numpy as np

from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from utils import get_main_directory

project_root = get_main_directory(__file__)
if project_root is None:
    raise RuntimeError("Project root directory not found. Please check the directory structure again!")


def convert_pixel_to_hu(dcm):
    pixel_array = dcm.pixel_array.astype(np.float32)
    slope = getattr(dcm, "RescaleSlope", 1)
    intercept = getattr(dcm, "RescaleIntercept", 0)
    return slope * pixel_array + intercept

def apply_window(img, window_center, window_width):
    min_val = window_center - (window_width / 2)
    max_val = window_center + (window_width / 2)
    img = np.clip(img, min_val, max_val)
    return ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

def process_file(args):
    dcm_path, output_path, window_center, window_width = args
    dcm = pydicom.dcmread(dcm_path)
    hu_img = convert_pixel_to_hu(dcm)
    windowed_img = apply_window(hu_img, window_center, window_width)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(windowed_img).save(output_path)

def dcm_2_png(input_folder, output_folder, window_center, window_width):
    dcm_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith((".dicom", ".dcm")):
                dcm_path = os.path.join(root, file)
                output_path = os.path.join(
                    output_folder,
                    os.path.relpath(root, input_folder),
                    f"{os.path.splitext(file)[0]}.png"
                )
                dcm_files.append((dcm_path, output_path, window_center, window_width))

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(
            executor.map(process_file, dcm_files),
            total=len(dcm_files),
            desc="Converting DICOM to PNG",
            unit="file"
        ))

if __name__ == "__main__":
    input_folder = os.path.join(project_root, "assets/data/vindr_multiphase/abdomen_phases")
    output_folder = os.path.join(project_root, "assets/data/vindr_multiphase/abdomen_phases_png")
    window_center = 50
    window_width = 400

    dcm_2_png(input_folder, output_folder, window_center, window_width)