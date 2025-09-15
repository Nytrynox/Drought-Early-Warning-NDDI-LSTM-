import os
import io
from typing import List
import pandas as pd


def load_csv_from_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))


def kaggle_download_dataset(dataset_slug: str, dest_dir: str) -> List[str]:
    """Download a Kaggle dataset into dest_dir using kaggle API CLI.
    Returns list of extracted file paths.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    os.makedirs(dest_dir, exist_ok=True)

    # Download and unzip
    api.dataset_download_files(dataset_slug, path=dest_dir, unzip=True)

    # Collect all files
    results: List[str] = []
    for root, _, files in os.walk(dest_dir):
        for f in files:
            results.append(os.path.join(root, f))
    return results
