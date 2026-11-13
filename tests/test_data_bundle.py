#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Team hand-off tools: scripts/pack_data_bundle.py cuts a split's evaluation inputs into <100 MB parts,
scripts/unpack_data_bundle.py verifies and restores them byte for byte, scripts/merge_results.py
combines teammates' seeds (folder or zip) and refuses results evaluated differently.

Run:  python -m pytest -q tests/test_data_bundle.py
"""
import json
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = ("pack_data_bundle.py", "unpack_data_bundle.py", "merge_results.py")


def _repo(tmp_path, name):
    r = tmp_path / name
    (r / "scripts").mkdir(parents=True)
    for s in SCRIPTS:
        shutil.copy(os.path.join(ROOT, "scripts", s), r / "scripts" / s)
    return r


def _run(repo, script, *args, env=None):
    p = subprocess.run([sys.executable, os.path.join("scripts", script), *args], cwd=repo,
                       capture_output=True, text=True, env=dict(os.environ, **(env or {})))
    return p.returncode, p.stdout + p.stderr


def test_pack_unpack_roundtrip_in_parts(tmp_path):
    src = _repo(tmp_path, "src")
    files = {"data/kc_k30/ikgr-custom/ikgr-custom.inter": os.urandom(250_000),
             "data/interactions.csv.stats.json": b"{}", "data/kc_k30/interactions_k30.csv.stats.json": b"{}",
             "data/kc_k30/profiles_k30_TO.csv": b"user_id,user_profile\n1,x\n" * 2000,
             "data/kc_k30/profile_provenance_k30_TO.csv.gz": os.urandom(40_000),
             "data/kc_k30/profile_provenance_k30_TO.csv.gz.meta.json": b'{"split": "TO"}',
             "run_k30_TO/kg_pack.pt": os.urandom(30_000), "run_k30_TO/meta_kg_pack.pt": os.urandom(5_000)}
    for rel, data in files.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_bytes(data)
    rc, out = _run(src, "pack_data_bundle.py", "--split", "TO", env={"IKGR_BUNDLE_PART_BYTES": "100000"})
    assert rc == 0, out
    parts = sorted((src / "data_bundle").glob("movielens_k30_TO_bundle.tgz.part*"))
    assert len(parts) >= 3 and all(p.stat().st_size <= 100_000 for p in parts)
    man = json.loads((src / "data_bundle" / "movielens_k30_TO_bundle.files.json").read_text())
    assert {f["path"] for f in man["files"]} == set(files)                     # no interactions_k30.csv
    rc, out = _run(src, "pack_data_bundle.py", "--split", "TO_GLOBAL")
    assert rc != 0 and "missing" in out                                         # TO_GLOBAL not finished

    dst = _repo(tmp_path, "dst")
    shutil.copytree(src / "data_bundle", dst / "data_bundle")
    rc, out = _run(dst, "unpack_data_bundle.py", "--check")
    assert rc == 0 and "parts OK" in out and not (dst / "run_k30_TO").exists()
    rc, out = _run(dst, "unpack_data_bundle.py")
    assert rc == 0, out
    for rel, data in files.items():
        assert (dst / rel).read_bytes() == data

    bad = dst / "data_bundle" / parts[1].name
    bad.write_bytes(b"x" + bad.read_bytes()[1:])
    rc, out = _run(dst, "unpack_data_bundle.py", "--check")
    assert rc != 0 and "sha256 mismatch" in out


def _teammate(base, split, results, files):
    d = base / f"run_k30_{split}"
    (d / f"per_user_{split}").mkdir(parents=True)
    (d / f"slice_eval_{split}_result.json").write_text(json.dumps(results))
    for name in files:
        (d / f"per_user_{split}" / name).write_bytes(name.encode())
    return d


def _entry(sig, seeds):
    return {"run_signature": sig, "run_context": {"spec": "LightGCN", "code": sig},
            "seeds": {s: {"ndcg@10": v, "train_sec": 1.0} for s, v in seeds.items()}}


def test_merge_seeds_from_folders_and_zips(tmp_path):
    repo = _repo(tmp_path, "repo")
    a = _teammate(tmp_path / "A", "TO", {"LightGCN": _entry("s1", {"2020": 0.1}), "BPR": _entry("b1", {"2020": 0.05})},
                  ["LightGCN__s2020__test.csv.gz", "BPR__s2020__test.csv.gz"])
    b = _teammate(tmp_path / "B", "TO", {"LightGCN": _entry("s1", {"2021": 0.2})}, ["LightGCN__s2021__test.csv.gz"])
    c = _teammate(tmp_path / "C", "TO", {"LightGCN": _entry("OTHER", {"2022": 0.3})}, ["LightGCN__s2022__test.csv.gz"])
    zb = tmp_path / "B.zip"
    with zipfile.ZipFile(zb, "w") as z:
        for p in b.rglob("*"):
            if p.is_file():
                z.write(p, arcname=os.path.join("results_B", os.path.relpath(p, b.parent)))
    rc, out = _run(repo, "merge_results.py", "--split", "TO", "--inputs", str(a), str(zb), str(c))
    assert rc == 1, out                                                          # C was evaluated differently
    assert "LightGCN was evaluated differently (code)" in out
    merged = json.loads((repo / "run_k30_TO" / "slice_eval_TO_result.json").read_text())
    assert sorted(merged["LightGCN"]["seeds"]) == ["2020", "2021"] and sorted(merged["BPR"]["seeds"]) == ["2020"]
    pu = sorted(p.name for p in (repo / "run_k30_TO" / "per_user_TO").iterdir())
    assert pu == ["BPR__s2020__test.csv.gz", "LightGCN__s2020__test.csv.gz", "LightGCN__s2021__test.csv.gz"]
    rc, out = _run(repo, "merge_results.py", "--split", "TO", "--inputs", str(a))       # idempotent re-run
    assert rc == 0 and ": 0 new result x seed entries" in out
