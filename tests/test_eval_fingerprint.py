#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation signatures (eval_slices._experiment_context) must let teammates run seeds on different machines
and merge them: a spec's signature covers only the data files it reads, hashes their CONTENT (not gzip
timestamps or torch.save bytes), normalises source line endings and stores repo-relative paths.

Run:  python -m pytest -q tests/test_eval_fingerprint.py
"""
import gzip
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
pytest.importorskip("recbole")
import torch  # noqa: E402

import eval_slices as ev  # noqa: E402


@pytest.fixture()
def at_root(monkeypatch):
    monkeypatch.chdir(ROOT)                       # code paths in the signature are repo-relative
    ev._SHA_MEMO.clear()


def _write_gz(path, text, mtime):
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=mtime) as f:
        f.write(text.encode())


def _sig(name, paths):
    rb = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))["recbole"]
    model_arg, extra = ev.build_specs(paths)[name]
    return ev._experiment_context(name, model_arg, extra, rb, paths, "TO")[0]


def test_unused_kg_files_do_not_touch_a_spec_signature(tmp_path, at_root):
    paths = {"profile_provenance": str(tmp_path / "prov.csv.gz"), "kg_pack": str(tmp_path / "kg_pack.pt"),
             "meta_kg_pack": str(tmp_path / "meta_kg_pack.pt"), "workdir": str(tmp_path),
             "inter_file": "data/kc_k30/interactions_k30.csv"}
    torch.save({"genre": torch.arange(6)}, paths["meta_kg_pack"])
    before = {s: _sig(s, paths) for s in ("BPR", "LightGCN", "IKGR_dyn_nointent", "IKGR_dyn")}

    # the TO bundle arrives: intent KG + provenance appear -> no-intent results stay valid
    torch.save({"edges": torch.ones(3, 2)}, paths["kg_pack"])
    _write_gz(paths["profile_provenance"], "user_id,item_id\n1,2\n", mtime=1)
    after = {s: _sig(s, paths) for s in before}
    assert after["BPR"] == before["BPR"] and after["LightGCN"] == before["LightGCN"]
    assert after["IKGR_dyn_nointent"] == before["IKGR_dyn_nointent"]
    assert after["IKGR_dyn"] != before["IKGR_dyn"]

    # a genuinely different metadata KG invalidates exactly the specs that read it
    torch.save({"genre": torch.arange(7)}, paths["meta_kg_pack"])
    changed = {s: _sig(s, paths) for s in before}
    assert changed["BPR"] == after["BPR"]
    assert changed["IKGR_dyn_nointent"] != after["IKGR_dyn_nointent"]
    assert changed["IKGR_dyn"] != after["IKGR_dyn"]


def test_content_hash_ignores_byte_noise(tmp_path, at_root):
    a, b = tmp_path / "a.csv.gz", tmp_path / "b.csv.gz"
    _write_gz(a, "user_id,item_id\n1,2\n3,4\n", mtime=1_700_000_000)
    _write_gz(b, "user_id,item_id\n1,2\n3,4\n", mtime=1_800_000_000)          # rebuilt later: other header
    assert a.read_bytes() != b.read_bytes() and ev._file_sha256(str(a)) == ev._file_sha256(str(b))
    _write_gz(tmp_path / "c.csv.gz", "user_id,item_id\n1,2\n3,5\n", mtime=1)
    assert ev._file_sha256(str(tmp_path / "c.csv.gz")) != ev._file_sha256(str(a))

    torch.save({"x": torch.tensor([1.0, 2.0]), "rel": {"genre": [1, 2]}}, tmp_path / "p.pt")
    torch.save({"rel": {"genre": [1, 2]}, "x": torch.tensor([1.0, 2.0])}, tmp_path / "q.pt")   # other key order
    torch.save({"x": torch.tensor([1.0, 2.5]), "rel": {"genre": [1, 2]}}, tmp_path / "r.pt")
    assert ev._file_sha256(str(tmp_path / "p.pt")) == ev._file_sha256(str(tmp_path / "q.pt"))
    assert ev._file_sha256(str(tmp_path / "p.pt")) != ev._file_sha256(str(tmp_path / "r.pt"))


def test_signature_is_machine_independent(tmp_path, at_root):
    lf, crlf = tmp_path / "lf.py", tmp_path / "crlf.py"
    lf.write_bytes(b"a = 1\nb = 2\n")
    crlf.write_bytes(b"a = 1\r\nb = 2\r\n")
    assert ev._code_sha256(str(lf)) == ev._code_sha256(str(crlf))
    assert ev._portable(os.path.join(ROOT, "run_k30_TO", "kg_pack.pt")) == "run_k30_TO/kg_pack.pt"
    assert ev._portable("run_k30_TO/kg_pack.pt") == "run_k30_TO/kg_pack.pt" and ev._portable(3) == 3
    paths = {"kg_pack": "run_k30_TO/kg_pack.pt", "meta_kg_pack": "run_k30_TO/meta_kg_pack.pt",
             "profile_provenance": "data/kc_k30/p.csv.gz", "workdir": "run_k30_TO/",
             "inter_file": "data/kc_k30/interactions_k30.csv"}
    rb = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))["recbole"]
    model_arg, extra = ev.build_specs(paths)["IKGR_dyn"]
    _, ctx = ev._experiment_context("IKGR_dyn", model_arg, extra, rb, paths, "TO")
    flat = repr(ctx)
    assert ROOT not in flat                                                      # no machine-specific prefix
    assert ctx["extra"]["kg_pack_path"] == "run_k30_TO/kg_pack.pt"
    assert set(ctx["code"]) == {"eval_slices.py", "ikgr_core/model_ikgr.py", "ikgr_core/corona_retriever.py"}
