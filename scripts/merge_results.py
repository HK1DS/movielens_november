#!/usr/bin/env python3
"""Merge evaluation results that teammates produced on different machines (seed-per-person split).

Each input is a teammate's run folder (run_k30_<SPLIT>/) or a .zip of it, holding
slice_eval_<SPLIT>_result.json and per_user_<SPLIT>/<result>__s<seed>__<phase>.csv.gz.
The merged JSON + per-user files go to --out (default run_k<K>_<SPLIT>/), ready for
`python paired_compare.py --run_dir run_k30_<SPLIT> --split <SPLIT> --report`.

A result is only merged when its run_signature (eval_slices.py: protocol, code, data the spec reads,
repo-relative paths) equals the one already collected, i.e. both machines evaluated the same thing.
A mismatch stops the merge and names the differing parts. A seed present in several inputs keeps the
first copy; a copy with different metrics is reported.

Usage (from the repository root):
    python scripts/merge_results.py --split TO --inputs from_A.zip from_B.zip run_k30_TO_C
    python scripts/merge_results.py --split TO --inputs ... --dry_run      # report only
"""
import argparse
import io
import json
import os
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Source:
    """Read-only view of one teammate's results (folder or zip)."""

    def __init__(self, path, split):
        self.path, self.split = path, split
        self.zip = zipfile.ZipFile(path) if path.lower().endswith(".zip") else None
        result_name = f"slice_eval_{split}_result.json"
        if self.zip:
            hits = [n for n in self.zip.namelist() if n.replace("\\", "/").endswith(result_name)]
            if len(hits) != 1:
                raise SystemExit(f"{path}: expected one {result_name}, found {len(hits)}")
            self.prefix = hits[0][: -len(result_name)]
            self.results = json.loads(self.zip.read(hits[0]).decode("utf-8"))
        else:
            fp = os.path.join(path, result_name)
            if not os.path.exists(fp):
                raise SystemExit(f"{path}: {result_name} not found")
            self.results = json.load(open(fp, encoding="utf-8"))

    def per_user_files(self, result, seed):
        stem = f"{result}__s{seed}__"
        if self.zip:
            pre = f"{self.prefix}per_user_{self.split}/"
            return {n[len(pre):]: n for n in self.zip.namelist()
                    if n.startswith(pre) and n[len(pre):].startswith(stem)}
        d = os.path.join(self.path, f"per_user_{self.split}")
        if not os.path.isdir(d):
            return {}
        return {f: os.path.join(d, f) for f in os.listdir(d) if f.startswith(stem)}

    def read(self, ref):
        return self.zip.read(ref) if self.zip else open(ref, "rb").read()


def _diff_context(a, b):
    keys = sorted(set(a) | set(b))
    return [k for k in keys if a.get(k) != b.get(k)]


def _metrics(rep):
    return {k: v for k, v in rep.items() if k not in ("train_sec", "fit", "leakage_guard")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["TO", "TO_GLOBAL", "RS"])
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--inputs", nargs="+", required=True, help="teammates' run folders or .zip files")
    ap.add_argument("--out", default=None, help="target run folder (default run_k<K>_<SPLIT>)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, f"run_k{args.k}_{args.split}")
    out_json = os.path.join(out, f"slice_eval_{args.split}_result.json")
    out_pu = os.path.join(out, f"per_user_{args.split}")

    merged = json.load(open(out_json, encoding="utf-8")) if os.path.exists(out_json) else {}
    if merged:
        print(f"[merge] starting from {out_json} ({len(merged)} results)")
    copies, problems = [], []
    for path in args.inputs:
        src = Source(path, args.split)
        added = 0
        for rname, entry in src.results.items():
            have = merged.get(rname)
            if have is None:
                merged[rname] = {"run_signature": entry["run_signature"], "run_context": entry["run_context"],
                                 "seeds": {}}
                have = merged[rname]
            elif have["run_signature"] != entry["run_signature"]:
                diff = _diff_context(have.get("run_context", {}), entry.get("run_context", {}))
                problems.append(f"{path}: {rname} was evaluated differently ({', '.join(diff) or 'signature'}) "
                                f"-> not merged; rerun it with the same branch/data")
                continue
            for seed, rep in entry["seeds"].items():
                if seed in have["seeds"]:
                    if _metrics(have["seeds"][seed]) != _metrics(rep):
                        problems.append(f"{path}: {rname} seed {seed} already merged from another input with "
                                        f"different metrics -> kept the first copy")
                    continue
                have["seeds"][seed] = rep
                added += 1
                copies += [(src, ref, name) for name, ref in src.per_user_files(rname, seed).items()]
        print(f"[merge] {path}: {added} new result x seed entries")

    for rname in sorted(merged):
        print(f"  {rname}: seeds {sorted(merged[rname]['seeds'])}")
    for p in problems:
        print(f"[merge] WARNING {p}")
    if args.dry_run:
        print("[merge] --dry_run: nothing written")
        return 1 if problems else 0
    os.makedirs(out_pu, exist_ok=True)
    for src, ref, name in copies:
        dst = os.path.join(out_pu, name)
        data = src.read(ref)
        if os.path.exists(dst) and open(dst, "rb").read() != data:
            print(f"[merge] WARNING {dst} exists with other content -> kept")
            continue
        with open(dst, "wb") as f:
            f.write(data)
    if os.path.exists(out_json):
        shutil.copyfile(out_json, out_json + ".bak")
    tmp = out_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=1)
    os.replace(tmp, out_json)
    print(f"[merge] wrote {out_json} and {len(copies)} per-user files -> next: "
          f"python paired_compare.py --run_dir {os.path.relpath(out)} --split {args.split} --report")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
