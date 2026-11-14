#!/usr/bin/env python3
"""Check (or rebuild) the metadata KG packs so every teammate evaluates with the SAME file content.

build_meta_kg.py is deterministic since 2026-11-14 (before, node ids followed Python's per-process set order,
so each build differed and invalidated the eval signatures of the specs reading it). TO and TO_GLOBAL share
the same .inter, so both packs have the same content.

Usage (repository root):
    python scripts/check_meta_kg.py              # print content hashes, compare with the expected one
    python scripts/check_meta_kg.py --rebuild    # rebuild both packs with build_meta_kg.py, then check
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# content hash (eval_slices._file_sha256: tensors/lists, not file bytes) of the MovieLens-25M k=30 pack
EXPECTED = {30: "eb64bb38a9656e81"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    inter = f"data/kc_k{args.k}/ikgr-custom/ikgr-custom.inter"
    packs = [f"run_k{args.k}_{s}/meta_kg_pack.pt" for s in ("TO", "TO_GLOBAL")]
    if args.rebuild:
        for p in packs:
            subprocess.run([sys.executable, "build_meta_kg.py", "--inter", inter, "--out", p], check=True)
    import eval_slices as ev
    want = EXPECTED.get(args.k)
    bad = 0
    for p in packs:
        if not os.path.exists(p):
            print(f"[meta] {p}: missing (run with --rebuild)")
            bad += 1
            continue
        got = ev._file_sha256(p)
        ok = want is None or got == want
        bad += not ok
        print(f"[meta] {p}: {got} " + ("OK" if ok else f"MISMATCH (expected {want}) -> run with --rebuild"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
