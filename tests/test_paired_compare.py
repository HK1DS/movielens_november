#!/usr/bin/env python3
"""paired_compare.py --min_activity: only users with enough train history enter the pairs and the summary
(TO_GLOBAL: most test users are cold, so warm users are reported separately).

Run:  python -m pytest -q tests/test_paired_compare.py
"""
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write(pu, name, seed, ndcg):
    df = pd.DataFrame({"user_id": ["1", "2", "3", "4"], "activity_train": [0, 0, 3, 7],
                       "ndcg@10": ndcg, "tail_recall@10": [0.0, 0.0, 0.0, 0.1]})
    df.to_csv(pu / f"{name}__s{seed}__test.csv.gz", index=False, compression="gzip")


def test_min_activity_keeps_warm_users_only(tmp_path):
    pu = tmp_path / "per_user_TO_GLOBAL"
    pu.mkdir()
    for seed in (2020, 2021):
        _write(pu, "IKGR_dyn", seed, [0.9, 0.9, 0.2, 0.4])          # cold users high, warm users low
        _write(pu, "BPR", seed, [0.0, 0.0, 0.3, 0.5])
        _write(pu, "BPR_pop0p5", seed, [0.0, 0.0, 0.3, 0.5])
    p = subprocess.run([sys.executable, os.path.join(ROOT, "paired_compare.py"), "--run_dir", str(tmp_path),
                        "--split", "TO_GLOBAL", "--a", "IKGR_dyn", "--b", "BPR", "--metrics", "ndcg@10",
                        "--report", "--min_activity", "1", "--n_boot", "200"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    rep = (tmp_path / "report_TO_GLOBAL_warm.md").read_text(encoding="utf-8")
    assert "| IKGR_dyn vs BPR | ndcg@10 | 2 | 0.3000 | 0.4000 | -0.1000" in rep    # 2 warm users, B wins
    assert "| BPR | 2 | 2 | 0.4000 |" in rep and "BPR_pop0p5" not in rep.split("Per-result means")[1]
