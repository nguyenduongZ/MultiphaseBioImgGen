import os, sys
sys.path.append(os.path.abspath(os.curdir))

import csv
import pandas as pd

from tqdm import tqdm
from pydicom import dcmread
from functools import partial
from omegaconf import DictConfig
from concurrent.futures import ProcessPoolExecutor

from src.utils import MasterLogger

FIELDS = [
    "image_path", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "PatientSex",
    "PatientAge", "PatientWeight", "BodyPartExamined", "Modality", "ScanOptions",
    "InstanceNumber", "SliceLocation", "ContrastBolusAgent", "ContrastBolusStartTime",
    "ImageOrientationPatient", "PixelSpacing", "SliceThickness"
]

def extract_metadata(dcm_file, project_root, logger):
    try:
        dataset = dcmread(dcm_file, stop_before_pixels=True)

        image_path = "./" + os.path.relpath(dcm_file, project_root)
        
        metadata = {field: getattr(dataset, field, None) for field in FIELDS if field != "image_path"}
        metadata["image_path"] = image_path
        
        return metadata
    
    except Exception as e:
        logger.error(f"Error processing file {dcm_file}: {e}")
        return None
    
def get_dcm_files(dcm_dir, logger):
    dcm_files = []
    logger.info("Scanning for DICOM files...")

    for root, _, files in os.walk(dcm_dir):
        for file in files:
            if file.lower().endswith((".dicom", ".dcm")):
                dcm_files.append(os.path.join(root, file))
                if len(dcm_files) % 10000 == 0:
                    logger.info(f"Found {len(dcm_files)} DICOM files...")

    return dcm_files

def process_files(dcm_files, output_csv, workers, project_root, logger):
    if os.path.exists(output_csv):
        logger.info(f"Removing existing CSV file: {output_csv}")
        os.remove(output_csv)

    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDS)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=workers) as executor:
            process_func = partial(extract_metadata, project_root=project_root, logger=logger)
            for result in tqdm(executor.map(process_func, dcm_files), total=len(dcm_files), desc="Processing DICOM files"):
                if result:
                    writer.writerow(result)

def main():
    project_root = "/".join(__file__.split("/")[:-4])
    
    _root = project_root + "/data/vindr_multiphase"
    dcm_dir = os.path.join(_root, "abdomen_phases")
    output_csv = os.path.join(_root, "metadata.csv")

    logger_config = {
        "cfg_file": os.path.join(project_root, "configs/utils/logger.conf"),
        "name": "EXTRACT METADATA",
        "level": "INFO"
    }
    
    master_logger = MasterLogger(DictConfig(logger_config))
    logger = master_logger.get_logger()
    logger.info("Starting DICOM metadata extraction")

    if not os.path.exists(dcm_dir):
        msg = f"DICOM directory not found: {dcm_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    dcm_files = get_dcm_files(dcm_dir, logger)
    
    if not dcm_files:
        msg = "No DICOM files found."
        logger.info(msg)
        raise RuntimeError(msg)

    logger.debug(f"Processing {len(dcm_files)} DICOM files...")

    process_files(dcm_files, output_csv, workers=4, project_root=project_root, logger=logger)

    df = pd.read_csv(output_csv, low_memory=False)
    logger.info(f"Metadata shape: {df.shape}")
    logger.info(f"Metadata extraction completed. Output saved to {output_csv}")

if __name__ == "__main__":
    main()