import os, sys
import pandas as pd

from torch.utils.data import DataLoader
from .vindr_multiphase import VindrMultiphase
from sklearn.model_selection import train_test_split

def get_ds_vindr_multiphase(args):
    metadata_path = args.metadata_path

    df = pd.read_csv(metadata_path, low_memory=False)

    study_ids = df['StudyInstanceUID'].unique()
    train_study_ids, temp_study_ids = train_test_split(study_ids, test_size=0.2, random_state=args.seed)
    valid_study_ids, test_study_ids = train_test_split(temp_study_ids, test_size=0.5, random_state=args.seed)

    train_df = df[df["StudyInstanceUID"].isin(train_study_ids)]
    valid_df = df[df["StudyInstanceUID"].isin(valid_study_ids)]
    test_df = df[df["StudyInstanceUID"].isin(test_study_ids)]

    train_ds = VindrMultiphase(args=args, df=train_df, split="train")
    valid_ds = VindrMultiphase(args=args, df=valid_df, split="valid")
    test_ds = VindrMultiphase(args=args, df=test_df, split="test")

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pm, num_workers=args.wk)
    valid_dl = DataLoader(valid_ds, batch_size=args.bs, shuffle=False, pin_memory=args.pm, num_workers=args.wk)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=False, pin_memory=args.pm, num_workers=args.wk)

    args.num_train_sample = len(train_ds)
    args.num_valid_sample = len(valid_ds)
    args.num_test_sample = len(test_ds)
    args.num_train_batch = len(train_dl)
    args.num_valid_batch = len(valid_dl)
    args.num_test_batch = len(test_dl)

    return (train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl), args

def get_ds(args):
    ds_mapping = {
        "vindr_multiphase" : get_ds_vindr_multiphase
    }

    data, args = ds_mapping[args.ds](args)

    return data, args