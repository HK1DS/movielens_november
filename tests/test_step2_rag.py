#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2's frozen RAG inputs (ikgr_core/rag.py): vocabulary, embeddings and every profile's top-k
candidates are built once (GPU when available) and reused on resume without re-encoding; the exact
cosine top-k matches brute force; a changed step1 output with cached step2 results stops (exit 2)
unless IKGR_STEP2_REBUILD=1; --prepare_only makes no API call.

Run:  python -m pytest -q tests/test_step2_rag.py
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import fake_llm_server  # noqa: E402
from ikgr_core.rag import pick_device, topk_cosine  # noqa: E402


def _brute(q, k_, k):
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    kn = k_ / np.linalg.norm(k_, axis=1, keepdims=True)
    return np.argsort(-(qn @ kn.T), axis=1, kind="stable")[:, :k]


def test_topk_cosine_matches_brute_force_in_blocks():
    rng = np.random.default_rng(0)
    q, keys = rng.normal(size=(57, 16)).astype(np.float32), rng.normal(size=(300, 16)).astype(np.float32)
    want = _brute(q, keys, 10)
    for max_bytes in (4 * 300, 4 * 300 * 7, 256 << 20):          # block of 1 row, 7 rows, everything
        got = topk_cosine(q, keys, 10, "cpu", max_bytes=max_bytes)
        assert got.dtype == np.int32 and got.shape == (57, 10) and (got == want).all()
    assert topk_cosine(q, keys[:4], 10).shape == (57, 4)           # k capped at the vocabulary size
    assert topk_cosine(q[:0], keys, 10).shape == (0, 10)
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    if cuda:
        assert (topk_cosine(q, keys, 10, "cuda", max_bytes=4 * 300 * 7) == want).all()


def test_pick_device(monkeypatch):
    monkeypatch.setenv("IKGR_ENCODER_DEVICE", "cpu")
    assert pick_device() == "cpu"
    monkeypatch.setenv("IKGR_ENCODER_DEVICE", "tpu")
    with pytest.raises(ValueError):
        pick_device()
    monkeypatch.setenv("IKGR_ENCODER_DEVICE", "auto")
    assert pick_device() in ("cpu", "cuda")


# ---- step2 end to end (offline fake LLM + fake encoder) ------------------------------------------

@pytest.fixture()
def server():
    from ikgr_core import key_pool
    key_pool._POOLS.clear()
    fake_llm_server.reset()
    srv = fake_llm_server.start()
    yield srv
    srv.shutdown()
    os.environ.pop("FAKE_LLM_DELAY", None)


def _setup(tmp_path, server, n_users=30, n_items=10):
    words = ["space", "dragons", "pirates", "romance", "heist", "robots", "jazz", "war", "comedy", "ghosts"]
    rows = []
    for k in range(n_users):
        i = k % n_items
        u_int = [f"{words[k % 10]} adventure", f"{words[(k + 3) % 10]} story", f"user taste {k}"]
        i_int = [f"{words[i]} adventure", f"{words[(i + 5) % 10]} movie"]
        rows.append((f"u{k}", f"Favorite movies: {words[k % 10]} film {k} | Frequent genres: {words[(k + 1) % 10]}",
                     f"i{i}", f"{words[i]} movie {i} | {words[(i + 2) % 10]}", str(u_int), str(i_int)))
    df = pd.DataFrame(rows, columns=["user_id", "user_profile", "item_id", "item_profile",
                                     "user_intents_exact", "item_intents_exact"])
    run = tmp_path / "run"
    run.mkdir()
    df.to_csv(run / "step1_intents.csv", index=False)
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
    cfg["paths"].update({"workdir": str(run), "step1_output": str(run / "step1_intents.csv"),
                         "step2_output": str(run / "step2_related_intents.csv")})
    cfg["rag"].update({"knn_k": 10, "encoding_npy": str(run / "intents_emb.npy"),
                       "vocab_json": str(run / "intent_vocab.json")})
    cfg["llm"].update({"base_url": f"http://127.0.0.1:{server.server_address[1]}/v1", "provider": "openai",
                       "api_key": "offline-test", "model": "fake"})
    yaml.safe_dump(cfg, open(tmp_path / "config.yaml", "w", encoding="utf-8"), sort_keys=False)
    env = dict(os.environ, IKGR_CONFIG=str(tmp_path / "config.yaml"), PYTHONPATH=os.path.join(ROOT, "tests", "fakes"),
               IKGR_LLM_WORKERS_PER_KEY="2", IKGR_PROGRESS_EVERY_S="0.3", IKGR_ENCODER_DEVICE="cpu")
    env.pop("FAKE_ST_FORBID", None)
    return run, df, env


def _step2(env, *args):
    r = subprocess.run([sys.executable, "step2.py", *args], cwd=ROOT, env=env, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_step2_freezes_rag_inputs_once_and_resumes_without_reencoding(server, tmp_path):
    run, df, env = _setup(tmp_path, server)

    rc, out = _step2(env, "--prepare_only")
    assert rc == 0, out[-3000:]
    assert "building RAG inputs (not built yet)" in out and "no API call made" in out
    assert fake_llm_server.CALLS["n"] == 0
    man = json.load(open(run / "rag_manifest.json", encoding="utf-8"))
    vocab = json.load(open(run / "intent_vocab.json", encoding="utf-8"))
    assert vocab == sorted(vocab) and man["n_vocab"] == len(vocab) and man["device"] == "cpu"
    assert man["profiles"]["user"]["n"] == 30 and man["profiles"]["item"]["n"] == 10
    with np.load(run / "rag_candidates.npz") as z:
        assert z["user"].shape == (30, 10) and z["item"].shape == (10, 10)
    assert np.load(run / "intents_emb.npy").shape[0] == len(vocab)                # build_kg.py contract
    frozen = {p: _sha(run / p) for p in ("rag_candidates.npz", "intent_vocab.json", "intents_emb.npy")}

    # paid run under a short time budget: pauses; the encoder is never constructed again
    os.environ["FAKE_LLM_DELAY"] = "0.1"
    env2 = dict(env, FAKE_ST_FORBID="1", IKGR_LLM_DEADLINE=str(time.time() + 1.5))
    rc, out = _step2(env2)
    assert rc == 5, out[-3000:]
    assert "reusing frozen RAG inputs" in out
    cache = json.load(open(run / "step2_cache.json", encoding="utf-8"))
    assert 0 < len(cache["user"]) + len(cache["item"]) < 40

    env2.pop("IKGR_LLM_DEADLINE")
    rc, out = _step2(env2)
    assert rc == 0, out[-3000:]
    cache = json.load(open(run / "step2_cache.json", encoding="utf-8"))
    assert len(cache["user"]) == 30 and len(cache["item"]) == 10
    assert {p: _sha(run / p) for p in frozen} == frozen                          # untouched by the resumes
    exact = {r.user_profile.strip(): set(eval(r.user_intents_exact)) for r in df.itertuples()}
    assert all(not (set(rel) & exact[p]) for p, rel in cache["user"].items())     # exact intents never offered
    assert any(cache["user"].values())
    out_df = pd.read_csv(run / "step2_related_intents.csv", dtype=str, keep_default_na=False)
    assert {"user_intents_related", "item_intents_related"} <= set(out_df.columns)


def test_changed_step1_output_stops_instead_of_mixing_vocabularies(server, tmp_path):
    run, df, env = _setup(tmp_path, server)
    rc, out = _step2(env)
    assert rc == 0, out[-3000:]
    df.loc[0, "user_intents_exact"] = str(["a brand new intent"] + eval(df.loc[0, "user_intents_exact"]))
    df.to_csv(run / "step1_intents.csv", index=False)
    rc, out = _step2(env)
    assert rc == 2, out[-3000:]
    assert "STOP: inputs changed since the RAG files were built: n_vocab, vocab_sha256" in out
    rc, out = _step2(dict(env, IKGR_STEP2_REBUILD="1"))
    assert rc == 0, out[-3000:]
    assert "building RAG inputs (inputs changed" in out
    assert "a brand new intent" in json.load(open(run / "intent_vocab.json", encoding="utf-8"))
