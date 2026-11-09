"""Offline stand-in for sentence_transformers used ONLY by tests/smoke_e2e.py.
Deterministic hashed bag-of-words vectors (dim 768), unit-normalised."""
import hashlib
import re

import numpy as np

DIM = 768


def _vec(text):
    v = np.zeros(DIM, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", str(text).lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % DIM] += 1.0 if (h >> 20) & 1 else -1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class SentenceTransformer:
    def __init__(self, name, *a, **kw):
        self.name = name

    def encode(self, texts, convert_to_numpy=True, convert_to_tensor=False, **kw):
        single = isinstance(texts, str)
        arr = np.stack([_vec(t) for t in ([texts] if single else list(texts))]) if texts else np.zeros((0, DIM), np.float32)
        if convert_to_tensor:
            import torch
            arr = torch.from_numpy(arr)
        return arr[0] if single else arr
