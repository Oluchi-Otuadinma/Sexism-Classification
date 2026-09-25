"""
Load the EDOS sexism datasets via kagglehub — ported from
notebooks/06_full_pipeline.ipynb (X3a7e2118_1_066.ipynb).

The dataset ("aadyasingh55/sexism-detection-in-english-texts") carries the
EDOS 2022 splits used across the project:
    dev.csv, test (1).csv, train (2).csv
"""

from pathlib import Path
import pandas as pd


def load_edos_splits(dataset_id: str = "aadyasingh55/sexism-detection-in-english-texts") -> dict:
    """
    Download (and cache) the dataset via kagglehub and load all three splits.

    Returns:
        {'train': DataFrame, 'dev': DataFrame, 'test': DataFrame}
    """
    import kagglehub

    path = Path(kagglehub.dataset_download(dataset_id))
    print("Path to dataset files:", path)

    files = {
        "dev": "dev.csv",
        "test": "test (1).csv",
        "train": "train (2).csv",
    }
    splits = {}
    for name, filename in files.items():
        f = path / filename
        if not f.exists():
            # Fall back to a plain "<name>.csv" if the parenthesised variant is absent
            alt = path / f"{name}.csv"
            if alt.exists():
                f = alt
            else:
                raise FileNotFoundError(f"{filename} not found in {path}")
        splits[name] = pd.read_csv(f)
    return splits
