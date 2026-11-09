import os, json, random
import numpy as np
import pandas as pd
from typing import Any, List

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_json(obj: Any, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def write_csv(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)
