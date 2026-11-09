#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User-level paired comparisons + accuracy/exposure Pareto summary.

Reads the per-user tables written by eval_slices.py
(<run_dir>/per_user_<SPLIT>/<result>__s<seed>__<phase>.csv.gz), averages each
user's metric over seeds (repeated seeds of the same user stay together), and
compares two configurations on the SAME users:

  mean A, mean B, mean diff, relative diff, 95% bootstrap CI over users,
  Wilcoxon signed-rank p-value, win/tie/loss rates, optional activity buckets.

The Pareto section lists every result in slice_eval_<SPLIT>_result.json
(including pop-penalty rerank variants such as BPR_pop0p5) and marks the
configurations that are not dominated on (NDCG@10, tail Recall@10) and on
(NDCG@10, Coverage@10). A KG model only "buys" long-tail exposure if it sits
above the curve that a popularity penalty on plain CF already traces.

Usage
  python paired_compare.py --run_dir run_k30_TO --split TO --a IKGR_dyn --b BPR LightGCN IKGR_dyn_nointent
  python paired_compare.py --run_dir run_k30_TO --split TO --report     # default pairs + Pareto -> markdown
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

DEFAULT_PAIRS = [("IKGR_dyn", "BPR"), ("IKGR_dyn", "LightGCN"), ("IKGR_dyn", "IKGR_kgoff"),
                 ("IKGR_dyn", "IKGR_dyn_nointent"), ("IKGR_full_hetero", "IKGR_meta_only"),
                 ("IKGR_lgcn_dyn_nointent", "LightGCN"), ("IKGR_lgcn_dyn", "IKGR_lgcn_dyn_nointent"),
                 ("IKGR_lgcn_dyn_nointent", "IKGR_dyn_nointent")]


def load_user_means(pu_dir, result, metric, phase="test", seeds=None):
    files = sorted(glob.glob(os.path.join(pu_dir, f"{glob.escape(result)}__s*__{phase}.csv.gz")))
    frames = []
    for f in files:
        m = re.search(r"__s(\d+)__", os.path.basename(f))
        if seeds and m and int(m.group(1)) not in seeds:
            continue
        d = pd.read_csv(f, dtype={"user_id": str})
        if metric not in d.columns:
            raise KeyError(f"{metric} not in {f}")
        frames.append(d[["user_id", "activity_train", metric]])
    if not frames:
        return None, 0
    allf = pd.concat(frames)
    g = allf.groupby("user_id").agg(activity_train=("activity_train", "first"), value=(metric, "mean"))
    return g.dropna(subset=["value"]), len(frames)


def paired(a, b, n_boot=10000, seed=0):
    j = a.join(b[["value"]], how="inner", lsuffix="_a", rsuffix="_b")
    d = (j["value_a"] - j["value_b"]).to_numpy()
    n = len(d)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n)) if n * n_boot <= 5e7 else None
    if idx is not None:
        boots = d[idx].mean(axis=1)
    else:
        boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    try:
        from scipy.stats import wilcoxon
        p = float(wilcoxon(d, zero_method="zsplit").pvalue) if np.any(d != 0) else 1.0
    except Exception:
        p = float("nan")
    ma, mb = float(j["value_a"].mean()), float(j["value_b"].mean())
    return {"n_users": n, "mean_a": ma, "mean_b": mb, "diff": float(d.mean()),
            "rel_diff_pct": (100.0 * d.mean() / mb) if mb else float("nan"),
            "ci95": [float(lo), float(hi)], "wilcoxon_p": p,
            "win": float((d > 1e-12).mean()), "tie": float((np.abs(d) <= 1e-12).mean()),
            "loss": float((d < -1e-12).mean()), "joined": j}


def by_activity(j, q=(0.2, 0.4, 0.6, 0.8)):
    cuts = j["activity_train"].quantile(list(q)).to_numpy()
    lab = np.digitize(j["activity_train"].to_numpy(), cuts, right=True)
    rows = []
    for b in range(len(q) + 1):
        s = j[lab == b]
        if len(s) == 0:
            continue
        d = s["value_a"] - s["value_b"]
        rows.append({"bucket": f"Q{b + 1}", "activity<=": int(s["activity_train"].max()), "n": len(s),
                     "diff": float(d.mean()), "win": float((d > 1e-12).mean()), "loss": float((d < -1e-12).mean())})
    return rows


def pareto_rows(result_json):
    res = json.load(open(result_json, encoding="utf-8"))
    rows = []
    for name, r in res.items():
        agg = r.get("agg")
        if not agg:
            continue
        rows.append({"result": name, "n_seeds": len(r.get("seeds", {})),
                     "ndcg@10": agg["overall.ndcg@10"]["mean"], "ndcg_std": agg["overall.ndcg@10"]["std"],
                     "tail_recall@10": agg["long_tail.recall@10"]["mean"],
                     "coverage@10": agg["coverage@10"]["mean"],
                     "best_epoch": agg.get("best_epoch", {}).get("mean")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for other, col in (("tail_recall@10", "pareto_tail"), ("coverage@10", "pareto_cov")):
        flags = []
        for _, r in df.iterrows():
            dom = ((df["ndcg@10"] >= r["ndcg@10"]) & (df[other] >= r[other]) &
                   ((df["ndcg@10"] > r["ndcg@10"]) | (df[other] > r[other]))).any()
            flags.append(not dom)
        df[col] = flags
    return df.sort_values("ndcg@10", ascending=False)


def fmt_pair(a, b, metric, r):
    lo, hi = r["ci95"]
    return (f"| {a} vs {b} | {metric} | {r['n_users']:,} | {r['mean_a']:.4f} | {r['mean_b']:.4f} | "
            f"{r['diff']:+.4f} ({r['rel_diff_pct']:+.1f}%) | [{lo:+.4f}, {hi:+.4f}] | {r['wilcoxon_p']:.2g} | "
            f"{100 * r['win']:.0f}/{100 * r['tie']:.0f}/{100 * r['loss']:.0f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--split", default="TO")
    ap.add_argument("--a")
    ap.add_argument("--b", nargs="*", default=[])
    ap.add_argument("--metrics", nargs="*", default=["ndcg@10", "tail_recall@10"])
    ap.add_argument("--phase", default="test", choices=["test", "valid"])
    ap.add_argument("--seeds", default="")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--by_activity", action="store_true")
    ap.add_argument("--report", action="store_true", help="default pairs + Pareto -> <run_dir>/report_<split>.md")
    args = ap.parse_args()

    seeds = {int(s) for s in args.seeds.split(",") if s.strip()} or None
    pu_dir = os.path.join(args.run_dir, f"per_user_{args.split}")
    pairs = [(args.a, b) for b in args.b] if args.a else []
    if args.report and not pairs:
        pairs = DEFAULT_PAIRS

    lines = [f"# Paired comparisons ({args.split}, {args.phase})", "",
             "Unit = user (metric averaged over seeds first). CI = 95% bootstrap over users; "
             "p = Wilcoxon signed-rank; W/T/L = % users where A wins/ties/loses.", "",
             "| A vs B | metric | users | A | B | diff (rel) | 95% CI | p | W/T/L % |",
             "|---|---|---:|---:|---:|---:|---|---:|---|"]
    activity_blocks = []
    for a, b in pairs:
        for metric in args.metrics:
            ua, na = load_user_means(pu_dir, a, metric, args.phase, seeds)
            ub, nb = load_user_means(pu_dir, b, metric, args.phase, seeds)
            if ua is None or ub is None:
                print(f"[skip] {a} vs {b} ({metric}): per-user files missing")
                continue
            r = paired(ua, ub, args.n_boot)
            if r is None:
                continue
            lines.append(fmt_pair(a, b, metric, r))
            print(fmt_pair(a, b, metric, r))
            if args.by_activity or args.report:
                rows = by_activity(r["joined"])
                block = [f"", f"**{a} vs {b} — {metric} by train-activity quintile**", "",
                         "| bucket | activity ≤ | users | diff | win % | loss % |", "|---|---:|---:|---:|---:|---:|"]
                block += [f"| {x['bucket']} | {x['activity<=']} | {x['n']:,} | {x['diff']:+.4f} | "
                          f"{100 * x['win']:.0f} | {100 * x['loss']:.0f} |" for x in rows]
                activity_blocks += block
    lines += activity_blocks

    res_json = os.path.join(args.run_dir, "slice_eval_result.json" if args.split == "RS"
                            else f"slice_eval_{args.split}_result.json")
    if os.path.exists(res_json):
        df = pareto_rows(res_json)
        if not df.empty:
            lines += ["", f"# Accuracy / exposure Pareto ({args.split}, seed means)", "",
                      "★ = not dominated on (NDCG@10, tail Recall@10); ◆ = not dominated on (NDCG@10, Coverage@10).", "",
                      "| result | seeds | NDCG@10 | ±sd | tail R@10 | Cov@10 | best epoch | frontier |",
                      "|---|---:|---:|---:|---:|---:|---:|---|"]
            for _, r in df.iterrows():
                mark = ("★" if r["pareto_tail"] else "") + ("◆" if r["pareto_cov"] else "")
                be = "" if pd.isna(r["best_epoch"]) or r["best_epoch"] is None else f"{r['best_epoch']:.0f}"
                lines.append(f"| {r['result']} | {r['n_seeds']} | {r['ndcg@10']:.4f} | {r['ndcg_std']:.4f} | "
                             f"{r['tail_recall@10']:.4f} | {r['coverage@10']:.4f} | {be} | {mark} |")
    if args.report:
        out = os.path.join(args.run_dir, f"report_{args.split}.md")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[report] -> {out}")


if __name__ == "__main__":
    main()
