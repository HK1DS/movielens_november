"""
RAG inputs of step2: intent vocabulary, sentence-encoder embeddings and each profile's top-k candidate
intents, built ONCE per run directory and reused on every resume ("frozen").

  vocab_json      sorted unique step1 exact intents (intent id = position)       [D: build_kg.py]
  encoding_npy    [n_vocab, dim] encoder embeddings aligned with the vocab        [D: build_kg.py]
  rag_candidates.npz   per kind ("user", "item"): int32 [n_profiles, k] vocab ids of the k most similar
                  intents (exact cosine, most similar first); rows follow the SORTED unique profile texts
  rag_manifest.json    encoder, k, vocab hash, per-kind profile-list hashes (+ device, timing); written
                  last, so it only exists for a complete build. step2 reuses the files while this
                  fingerprint matches its inputs.

Exact cosine top-k replaces the earlier Annoy / sklearn index: same similarity, but deterministic,
identical on every machine (GPU or CPU, Windows or Linux) and computed once instead of on every run.
The encoder runs on the GPU when torch sees one: IKGR_ENCODER_DEVICE = auto (default) | cuda | cpu;
IKGR_ENCODER_BATCH sets the encode batch size (default 256 on GPU, 64 on CPU).
"""
import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FORMAT = 1
NOT_BUILT = "not built yet"
_PROFILE_CHUNK = 16384          # profiles encoded + ranked per block (bounds host memory)


# ---- device ---------------------------------------------------------------------------------------

def pick_device(pref: Optional[str] = None) -> str:
    pref = (pref or os.environ.get("IKGR_ENCODER_DEVICE", "auto")).strip().lower()
    if pref not in ("auto", "cpu", "cuda") and not pref.startswith("cuda:"):
        raise ValueError(f"IKGR_ENCODER_DEVICE must be auto, cuda, cuda:N or cpu (got {pref!r})")
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    if pref == "auto":
        return "cuda" if has_cuda else "cpu"
    if pref.startswith("cuda") and not has_cuda:
        raise RuntimeError(f"IKGR_ENCODER_DEVICE={pref} but torch sees no CUDA GPU "
                           "(CPU-only torch build, or CUDA_VISIBLE_DEVICES hides the GPU)")
    return pref


def device_label(device: str) -> str:
    if device.startswith("cuda"):
        try:
            import torch
            return f"{device} ({torch.cuda.get_device_name(torch.device(device))})"
        except Exception:
            return device
    return device


# ---- encoder + exact cosine top-k -----------------------------------------------------------------

class Encoder:
    def __init__(self, name: str, device: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device=device)
        self.device = device

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        default_bs = "256" if self.device.startswith("cuda") else "64"
        bs = int(os.environ.get("IKGR_ENCODER_BATCH", default_bs))
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        out = self.model.encode(texts, batch_size=bs, convert_to_numpy=True, show_progress_bar=len(texts) > 2000)
        return np.ascontiguousarray(out, dtype=np.float32)


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def topk_cosine(queries: np.ndarray, keys: np.ndarray, k: int, device: str = "cpu",
                max_bytes: int = 256 << 20) -> np.ndarray:
    """int32 [n_queries, min(k, n_keys)]: key ids by cosine similarity, most similar first.
    Scores are computed block-wise so one block of the score matrix stays under max_bytes."""
    nq, nk = len(queries), len(keys)
    k = min(int(k), nk)
    out = np.zeros((nq, k), dtype=np.int32)
    if nq == 0 or k == 0:
        return out
    q_all, kk = _unit(queries), _unit(keys)
    block = max(1, int(max_bytes // (4 * nk)))
    if device.startswith("cuda"):
        import torch
        dev = torch.device(device)
        kt = torch.from_numpy(kk).to(dev)
        for s in range(0, nq, block):
            qt = torch.from_numpy(q_all[s:s + block]).to(dev)
            idx = torch.topk(qt @ kt.T, k, dim=1, largest=True, sorted=True).indices
            out[s:s + block] = idx.cpu().numpy().astype(np.int32)
        del kt
        torch.cuda.empty_cache()
        return out
    for s in range(0, nq, block):
        sc = q_all[s:s + block] @ kk.T
        part = np.argpartition(-sc, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sc, part, axis=1), axis=1, kind="stable")
        out[s:s + block] = np.take_along_axis(part, order, axis=1)
    return out


# ---- frozen inputs --------------------------------------------------------------------------------

def sha256_lines(items: Sequence[str]) -> str:
    h = hashlib.sha256()
    for s in items:
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def fingerprint(encoder: str, k: int, vocab: Sequence[str], profiles: Dict[str, List[str]]) -> dict:
    return {"format": FORMAT, "encoder": encoder, "knn_k": int(k), "n_vocab": len(vocab),
            "vocab_sha256": sha256_lines(vocab),
            "profiles": {kind: {"n": len(p), "sha256": sha256_lines(p)} for kind, p in sorted(profiles.items())}}


def _atomic(path: str, write):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def load_frozen(paths: dict, want: dict) -> Tuple[Optional[Dict[str, np.ndarray]], str]:
    """(candidates, "") when the frozen files exist and match `want`, else (None, reason)."""
    files = [paths["manifest"], paths["candidates"], paths["vocab_json"], paths["encoding_npy"]]
    if not all(os.path.exists(p) for p in files):
        return None, NOT_BUILT
    with open(paths["manifest"], encoding="utf-8") as f:
        got = json.load(f)
    diff = [key for key in want if got.get(key) != want[key]]
    if diff:
        return None, "inputs changed since the RAG files were built: " + ", ".join(diff)
    with np.load(paths["candidates"]) as npz:
        cands = {kind: npz[kind] for kind in want["profiles"]}
    k = min(want["knn_k"], want["n_vocab"])
    for kind, meta in want["profiles"].items():
        if cands[kind].shape != (meta["n"], k):
            return None, f"{os.path.basename(paths['candidates'])} has the wrong shape for {kind}"
    return cands, ""


def build_frozen(paths: dict, want: dict, encoder_name: str, vocab: List[str],
                 profiles: Dict[str, List[str]], device: str, log=print) -> Dict[str, np.ndarray]:
    t0 = time.time()
    if os.path.exists(paths["manifest"]):
        os.remove(paths["manifest"])               # an interrupted rebuild must not look complete
    enc = Encoder(encoder_name, device)
    log(f"[rag] encoding {len(vocab):,} intents on {device_label(device)} ...")
    emb = enc.encode(vocab)
    _atomic(paths["vocab_json"], lambda f: f.write(json.dumps(vocab, ensure_ascii=False).encode("utf-8")))
    _atomic(paths["encoding_npy"], lambda f: np.save(f, emb))
    cands = {}
    for kind in sorted(profiles):
        plist, parts = profiles[kind], []
        log(f"[rag] encoding {len(plist):,} {kind} profiles + top-{want['knn_k']} candidates ...")
        for s in range(0, len(plist), _PROFILE_CHUNK):
            parts.append(topk_cosine(enc.encode(plist[s:s + _PROFILE_CHUNK]), emb, want["knn_k"], device))
        cands[kind] = (np.concatenate(parts) if parts
                       else np.zeros((0, min(want["knn_k"], len(vocab))), dtype=np.int32))
    _atomic(paths["candidates"], lambda f: np.savez(f, **cands))
    manifest = dict(want, device=device_label(device), seconds=round(time.time() - t0, 1),
                    built=time.strftime("%Y-%m-%d %H:%M:%S"))
    _atomic(paths["manifest"], lambda f: f.write(json.dumps(manifest, indent=1).encode("utf-8")))
    log(f"[rag] frozen RAG inputs written in {manifest['seconds']:.0f}s -> {paths['candidates']}")
    return cands
