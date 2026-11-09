import os
import sys
from typing import List, Sequence
from sentence_transformers import SentenceTransformer
import numpy as np
from .utils import save_json, load_json

# Dynamic fallback to support both Annoy (original) and sklearn NearestNeighbors.
# NOTE: On Windows the Annoy native extension reliably segfaults during
# AnnoyIndex.build() for large indexes (observed at 53K x 768, 50 trees) and
# kills the process WITHOUT a Python traceback. We therefore default to the
# sklearn brute-force NearestNeighbors backend on win32, which is fully
# reliable here. Set IKGR_FORCE_ANNOY=1 to force Annoy regardless of platform.
_FORCE_ANNOY = os.environ.get("IKGR_FORCE_ANNOY", "").lower() in ("1", "true", "yes")
_USE_ANNOY = _FORCE_ANNOY or sys.platform != "win32"
try:
    if not _USE_ANNOY:
        raise ImportError(
            "Annoy disabled on Windows due to native build crash; "
            "set IKGR_FORCE_ANNOY=1 to override."
        )
    from annoy import AnnoyIndex
    HAS_ANNOY = True
except ImportError:
    from sklearn.neighbors import NearestNeighbors
    HAS_ANNOY = False

if not HAS_ANNOY:
    class NearestNeighborsIndex:
        def __init__(self, dim: int = 768):
            self.dim = dim
            self.items = []
            self.nn = NearestNeighbors(n_neighbors=100, metric="cosine", algorithm="brute")
            self.emb = None

        def add_item(self, i: int, v: List[float]):
            self.items.append(v)

        def build(self, trees: int):
            self.emb = np.array(self.items, dtype=np.float32)
            self.nn.fit(self.emb)

        def save(self, out_path: str):
            with open(out_path, "wb") as f:
                np.save(f, self.emb)

        def load(self, path: str):
            with open(path, "rb") as f:
                self.emb = np.load(f)
            self.nn.fit(self.emb)

        def get_nns_by_vector(self, query: np.ndarray, k: int) -> List[int]:
            query = query.reshape(1, -1)
            _, indices = self.nn.kneighbors(query, n_neighbors=k)
            return indices[0].tolist()

class IntentEncoderIndex:
    def __init__(self, encoder_name: str, dim: int = 768):
        self.encoder = SentenceTransformer(encoder_name)
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        return self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=len(texts) > 2000)

    @staticmethod
    def build_ann(emb: np.ndarray, dim: int, trees: int, out_path: str):
        if HAS_ANNOY:
            ann = AnnoyIndex(dim, 'angular')
            for i, v in enumerate(emb):
                ann.add_item(i, v.tolist())
            ann.build(trees)
            ann.save(out_path)
        else:
            # Same index as add_item()+build(), without materialising one Python
            # float per vector element (~3.6 GB for 200k x 768 intents on Windows).
            ann = NearestNeighborsIndex(dim)
            ann.emb = np.ascontiguousarray(emb, dtype=np.float32)
            ann.nn.fit(ann.emb)
            ann.save(out_path)

    @staticmethod
    def load_ann(dim: int, path: str):
        if HAS_ANNOY:
            ann = AnnoyIndex(dim, 'angular')
            ann.load(path)
            return ann
        else:
            ann = NearestNeighborsIndex(dim)
            ann.load(path)
            return ann

def knn_strings(ann, emb_query: np.ndarray, vocab: List[str], k: int) -> List[str]:
    idxs = ann.get_nns_by_vector(emb_query, k)
    return [vocab[i] for i in idxs]
