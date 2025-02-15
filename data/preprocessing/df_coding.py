import os, sys
sys.path.append(os.path.abspath(os.curdir))

import pandas as pd

from utils.built_opt import Opt

def get_df_coding(opt: Opt, dataset_name: str):
    
    csv_file = opt.data["data"][dataset_name]["PATH_METADATA_PROMPT_FILE"]
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file, low_memory=False)
    
    except Exception as e:
        raise Exception(f"Error loading CSV file from {csv_file}: {str(e)}")
    
    opt.logger.debug(f"CSV file loaded \n {df.head(5)}")
    
    required_columns = {
         "image_path",
         "StudyInstanceUID",
         "InstanceNumber",
         "SliceLocation",
         "prompt",
         "Label"
    }
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV file must contain the following columns: {', '.join(required_columns)}")
    
    new_df = pd.DataFrame({
        "image_path": df["image_path"],
        "StudyInstanceUID": df["StudyInstanceUID"],
        "InstanceNumber": df["InstanceNumber"],
        "SliceLocation": df["SliceLocation"],
        "prompt": df["prompt"],
        "Label": df["Label"]
    }).reset_index(drop=True)
    
    return new_df