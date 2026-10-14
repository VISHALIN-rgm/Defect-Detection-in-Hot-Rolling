"""Shared data-loading helpers for the hot rolling defect detection project."""

import os
from zipfile import ZipFile

import pandas as pd

ZIP_NAME = "Datasets.zip"
REQUIRED_FILES = ("train.csv", "test.csv")


def _find_file(base_dir, name):
    """Return the path of `name` in base_dir or its 'dataset/' subfolder (the
    layout used inside Datasets.zip), or None if it is not found."""
    for folder in (base_dir, os.path.join(base_dir, "dataset")):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return None


def locate_dataset(base_dir="."):
    """Return (train_path, test_path), extracting Datasets.zip if needed."""
    paths = [_find_file(base_dir, name) for name in REQUIRED_FILES]
    if all(paths):
        return tuple(paths)

    zip_path = os.path.join(base_dir, ZIP_NAME)
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(
            f"Could not find train.csv/test.csv or {ZIP_NAME} in "
            f"'{os.path.abspath(base_dir)}'. Place {ZIP_NAME} next to the scripts."
        )

    print(f"Extracting {ZIP_NAME}...")
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(base_dir)

    paths = [_find_file(base_dir, name) for name in REQUIRED_FILES]
    if not all(paths):
        raise FileNotFoundError(f"{ZIP_NAME} does not contain train.csv and test.csv.")
    return tuple(paths)


def load_train_test(base_dir="."):
    """Load and validate train/test DataFrames."""
    train_path, test_path = locate_dataset(base_dir)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    for name, df, required in (
        ("train.csv", train_df, ("CoilID", "Y")),
        ("test.csv", test_df, ("CoilID",)),
    ):
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing required column(s): {missing}")
    return train_df, test_df
