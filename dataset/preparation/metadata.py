import os, sys
sys.path.append(os.path.abspath(os.curdir))
import csv
import pandas as pd

from tqdm import tqdm
from pydicom import dcmread
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from utils import get_main_directory

project_root = get_main_directory(__file__)
if project_root is None:
    raise RuntimeError("Project root directory not found. Please check the directory structure again!")

FIELDS = [
    "image_path", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "PatientSex",
    "PatientAge", "PatientWeight", "BodyPartExamined", "Modality", "ScanOptions",
    "InstanceNumber", "SliceLocation", "ContrastBolusAgent", "ContrastBolusStartTime",
    "ImageOrientationPatient", "PixelSpacing", "SliceThickness"
]

def extract_metadata(dcm_file, dcm_dir):
    try:
        dataset = dcmread(dcm_file, stop_before_pixels=True)
        image_path = "./" + os.path.relpath(dcm_file, project_root)
        
        metadata = {field: getattr(dataset, field, None) for field in FIELDS if field != "image_path"}
        metadata["image_path"] = image_path

        return metadata
    
    except Exception as e:
        print(f"Error processing file {dcm_file}: {e}")
        return None

def get_dcm_files(dcm_dir):
    dcm_files = []
    print("Scanning for DICOM files...")

    for root, _, files in os.walk(dcm_dir):
        for file in files:
            if file.lower().endswith((".dicom", ".dcm")):
                dcm_files.append(os.path.join(root, file))
                if len(dcm_files) % 10000 == 0:
                    print(f"Found {len(dcm_files)} DICOM files...")

    return dcm_files

def process_files(dcm_files, dcm_dir, output_csv, workers):
    if os.path.exists(output_csv):
        os.remove(output_csv)

    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDS)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=workers) as executor:
            process_func = partial(extract_metadata, dcm_dir=dcm_dir)
            for result in tqdm(executor.map(process_func, dcm_files), total=len(dcm_files), desc="Processing DICOM files"):
                if result:
                    writer.writerow(result)

if __name__ == "__main__":
    dcm_dir = os.path.join(project_root, "asset/data/vindr_multiphase/abdomen_phases")
    output_csv = os.path.join(project_root, "asset/data/vindr_multiphase/metadata1.csv")

    dcm_files = get_dcm_files(dcm_dir)

    if not dcm_files:
        print("No DICOM file found in the specified directory.")
        sys.exit(1)    

    print(f"Processing {len(dcm_files)} DICOM files...")

    process_files(dcm_files, dcm_dir, output_csv, workers=4) 

    df = pd.read_csv(output_csv, low_memory=False)
    print(f"Metadata shape: {df.shape}")