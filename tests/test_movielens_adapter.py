#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MovieLens (ml-25m format) adapter: reads ratings.csv / movies.csv exactly as GroupLens
ships them (quoted titles with commas, "(no genres listed)"), applies the earlier repo's
choices, and feeds stages 0 -> A -> T -> P with train-only profiles in the expected shape.

Run:  python -m pytest -q tests/test_movielens_adapter.py
"""
import csv
import os
import random
import subprocess
import sys

import pandas as pd
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data_adapters import get_adapter  # noqa: E402

MOVIES = [
    ("1", "Toy Story (1995)", "Adventure|Animation|Children|Comedy|Fantasy"),
    ("2", "American President, The (1995)", "Comedy|Drama|Romance"),
    ("3", 'Shawshank Redemption, The (1994)', "Crime|Drama"),
    ("4", "Heat (1995)", "Action|Crime|Thriller"),
    ("5", "Some Home Video (2018)", "(no genres listed)"),
    ("6", "Alien (1979)", "Horror|Sci-Fi"),
    ("7", '"Great Train Robbery, The (1903)"', "Crime|Western"),
    ("8", "Up (2009)", "Adventure|Animation|Children|Drama"),
]


def write_ml25m(d, n_users=40, seed=3):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "movies.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["movieId", "title", "genres"])
        w.writerows(MOVIES)
    rng = random.Random(seed)
    with open(d / "ratings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["userId", "movieId", "rating", "timestamp"])
        for u in range(1, n_users + 1):
            t = 1_100_000_000 + u * 1000
            for m, _, _ in MOVIES:
                if rng.random() < 0.9:
                    t += rng.choice([0, 0, 60, 3600])          # bulk-rating ties
                    w.writerow([u, m, rng.choice([0.5, 3.0, 4.0, 5.0]), t])


def test_adapter_reads_ml25m_format(tmp_path):
    write_ml25m(tmp_path / "ml-25m")
    a = get_adapter({"adapter": "movielens", "options": {"raw_dir": str(tmp_path / "ml-25m")}})
    rows = list(a.iter_interactions())
    assert rows and all(m != "5" for _, m, _, _ in rows)          # no-genre movie dropped
    u, m, r, ts = rows[0]
    assert isinstance(u, str) and isinstance(r, float) and ts > 1e9
    items = a.load_items({"1", "2", "5", "7", "999"})
    assert items["2"].title == "American President, The (1995)"
    assert items["2"].profile == "American President, The (1995) | Comedy|Drama|Romance"
    assert items["2"].relations == {"genre": ["Comedy", "Drama", "Romance"]}
    assert items["5"].relations == {"genre": []}
    assert items["7"].title.startswith('"Great Train Robbery')     # quotes preserved as in movies.csv
    assert "999" not in items
    keep = get_adapter({"adapter": "movielens", "options": {"raw_dir": str(tmp_path / "ml-25m"),
                                                            "drop_no_genres": False}})
    assert any(m == "5" for _, m, _, _ in keep.iter_interactions())
    assert [str(p) for p in a.source_files()] == [str(tmp_path / "ml-25m" / "movies.csv")]


def test_missing_raw_files_point_to_download(tmp_path):
    a = get_adapter({"adapter": "movielens", "options": {"raw_dir": str(tmp_path / "nope")}})
    with pytest.raises(FileNotFoundError, match="ml-25m.zip"):
        list(a.iter_interactions())


def test_stages_0_to_P_on_ml25m_format(tmp_path):
    write_ml25m(tmp_path / "raw" / "ml-25m")
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
    cfg["dataset"]["options"]["raw_dir"] = str(tmp_path / "raw" / "ml-25m")
    cfg_path = tmp_path / "config.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w", encoding="utf-8"), sort_keys=False)

    def run(*cmd):
        r = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout

    run("prepare_interactions.py", "--config", str(cfg_path), "--out", str(tmp_path / "interactions.csv"))
    run("apply_k_core.py", "--interactions_in", str(tmp_path / "interactions.csv"), "--k", "3",
        "--out_dir", str(tmp_path / "kc"))
    inter = tmp_path / "kc" / "ikgr-custom" / "ikgr-custom.inter"
    run("make_temporal_inter.py", "--inter_ts", str(tmp_path / "kc" / "interactions_k3.csv"), "--out", str(inter))
    out = run("build_trainonly_profiles.py", "--split", "TO", "--config", str(cfg_path), "--inter", str(inter),
              "--out_profiles", str(tmp_path / "prof.csv"), "--out_provenance", str(tmp_path / "prov.csv.gz"))
    assert "adapter=movielens" in out
    p = pd.read_csv(tmp_path / "prof.csv", dtype=str, keep_default_na=False)
    texts = [t for t in p.user_profile if t]
    assert texts and all(t.startswith("Recently watched (newest first): ") for t in texts)
    assert all(" | Frequent genres: " in t for t in texts)
    assert any("American President, The (1995)" in t for t in texts)
    assert set(p.item_profile) <= {f"{t} | {g}" for _, t, g in MOVIES}


def _repo_copy(tmp_path):
    import shutil
    wd = tmp_path / "repo"
    wd.mkdir()
    for f in ("prepare_interactions.py", "apply_k_core.py", "make_temporal_inter.py", "build_trainonly_profiles.py",
              "estimate_llm_cost.py", "run_pipeline.py", "config.yaml"):
        shutil.copy(os.path.join(ROOT, f), wd)
    for d in ("ikgr_core", "data_adapters", "prompts"):
        shutil.copytree(os.path.join(ROOT, d), wd / d)
    (wd / "ref_adapter.py").write_text(
        "import json, os\n"
        "from data_adapters.movielens import MovieLensAdapter\n"
        "class RefAdapter(MovieLensAdapter):\n"
        "    REFERENCE = json.loads(os.environ['TEST_REF'], object_hook=lambda d: {int(k) if k.isdigit() else k: v"
        " for k, v in d.items()})\n", encoding="utf-8")
    return wd


def test_scale_check_matches_and_blocks_paid_stages(tmp_path):
    import json as _json
    wd = _repo_copy(tmp_path)
    write_ml25m(wd / "raw" / "ml-25m")
    cfg = yaml.safe_load(open(wd / "config.yaml", encoding="utf-8"))
    cfg["dataset"]["adapter"] = "ref_adapter:RefAdapter"
    yaml.safe_dump(cfg, open(wd / "config.yaml", "w", encoding="utf-8"), sort_keys=False)

    def run(ref, *steps):
        env = dict(os.environ, TEST_REF=_json.dumps(ref), PYTHONPATH=str(wd))
        r = subprocess.run([sys.executable, "run_pipeline.py", "--k", "3", "--split", "TO", "--yes", *steps],
                           cwd=wd, env=env, capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    # 1) build stage 0/A once with a placeholder reference, read the real sizes
    placeholder = {"prepared": {"interactions": 0, "users": 0, "items": 0},
                   "kcore": {"3": {"interactions": 0, "users": 0, "items": 0}}}
    rc, out = run(placeholder, "--steps", "0A")
    assert rc == 0, out[-2000:]
    assert "[scale] MISMATCH" in out
    prep = _json.load(open(wd / "data" / "interactions.csv.stats.json"))
    kc = _json.load(open(wd / "data" / "kc_k3" / "interactions_k3.csv.stats.json"))
    assert prep["items"] == 7                                         # 8 movies minus the no-genre one
    # 2) wrong reference -> paid stage refused before any LLM call
    rc, out = run(placeholder, "--steps", "B")
    assert rc != 0 and "aborting before any paid LLM call" in out and "[preflight]" not in out
    # 3) the real sizes as reference -> MATCH, guard passes (stop before B: no LLM here)
    good = {"prepared": {k: prep[k] for k in ("interactions", "users", "items")},
            "kcore": {"3": {k: kc[k] for k in ("interactions", "users", "items")}}}
    rc, out = run(good, "--steps", "0A")
    assert rc == 0 and "[scale] MATCH" in out, out[-2000:]
    assert _json.load(open(wd / "run_k3_TO" / "scale_check.json"))["ok"] is True


def test_movielens_reference_is_the_earlier_repo_k30():
    a = get_adapter({"adapter": "movielens", "options": {}})
    ref = a.reference_stats(30)
    assert ref["kcore"] == {"interactions": 24_046_865, "users": 133_684, "items": 15_785}
    assert ref["prepared"] == {"interactions": 24_973_468, "users": 162_541, "items": 54_479}
    assert a.reference_stats(20) is None
    assert get_adapter({"adapter": "movielens", "options": {"drop_no_genres": False}}).reference_stats(30) is None
