"""Shared test fixture: synthetic raw data -> stages 0 / A / T (no LLM, no GPU)."""
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from make_synthetic_data import dataset_section  # noqa: E402


def _run(cmd, cwd=ROOT):
    r = subprocess.run([sys.executable] + cmd, cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def prepare(d, k=5):
    """Build d/kc/ikgr-custom/ikgr-custom.inter from synthetic data; return the config path."""
    d = str(d)
    _run([os.path.join(ROOT, "tests", "make_synthetic_data.py"), "--out_dir", d])
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
    cfg["dataset"] = dataset_section(d)
    cfg_path = os.path.join(d, "config.yaml")
    yaml.safe_dump(cfg, open(cfg_path, "w", encoding="utf-8"), sort_keys=False)
    _run([os.path.join(ROOT, "prepare_interactions.py"), "--config", cfg_path,
          "--out", os.path.join(d, "interactions.csv")])
    _run([os.path.join(ROOT, "apply_k_core.py"), "--interactions_in", os.path.join(d, "interactions.csv"),
          "--k", str(k), "--out_dir", os.path.join(d, "kc")])
    lo, hi = cfg["dataset"]["options"]["ts_clip"]
    _run([os.path.join(ROOT, "make_temporal_inter.py"), "--inter_ts", os.path.join(d, "kc", f"interactions_k{k}.csv"),
          "--out", os.path.join(d, "kc", "ikgr-custom", "ikgr-custom.inter"), "--clip_lo", lo, "--clip_hi", hi])
    return cfg_path
