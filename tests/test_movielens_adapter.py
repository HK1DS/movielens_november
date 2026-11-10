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
