#!/usr/bin/env python3
"""Reassemble and extract the data bundles shipped in data_bundle/ (made by scripts/pack_data_bundle.py).

Each bundle holds one split's evaluation inputs: train-only profiles + provenance and the intent / metadata
KG packs (TO also carries the shared RecBole .inter). With them the evaluation stage (E) runs without any
LLM/API call. Parts are <100 MB because GitHub rejects larger files.

Usage (from the repository root):
    python scripts/unpack_data_bundle.py                 # every bundle present: verify sha256, then extract
    python scripts/unpack_data_bundle.py --split TO      # one split
    python scripts/unpack_data_bundle.py --check         # verify only
"""
import argparse
import glob
import hashlib
import json
import os
import re
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_DIR = os.path.join(ROOT, "data_bundle")


class _Concat:
    """Read-only file object over the ordered parts (no temporary full archive)."""

    def __init__(self, paths):
        self._files = [open(p, "rb") for p in paths]
        self._i = 0

    def read(self, n=-1):
        out = bytearray()
        while self._i < len(self._files) and (n < 0 or len(out) < n):
            chunk = self._files[self._i].read(-1 if n < 0 else n - len(out))
            if chunk:
                out += chunk
            else:
                self._i += 1
        return bytes(out)

    def close(self):
        for f in self._files:
            f.close()


def bundles():
    names = sorted({re.sub(r"\.tgz\.part\d+$", "", os.path.basename(p))
                    for p in glob.glob(os.path.join(BUNDLE_DIR, "*_bundle.tgz.part*"))})
    if not names:
        raise SystemExit(f"no bundle parts found in {BUNDLE_DIR} (git pull?)")
    return names


def parts(name):
    return sorted(glob.glob(os.path.join(BUNDLE_DIR, f"{name}.tgz.part*")))


def verify(name):
    ps = parts(name)
    want = open(os.path.join(BUNDLE_DIR, f"{name}.sha256"), encoding="ascii").read().split()[0]
    h = hashlib.sha256()
    for p in ps:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    if h.hexdigest() != want:
        raise SystemExit(f"[bundle] {name}: sha256 mismatch (re-download data_bundle/ with git pull)")
    print(f"[bundle] {name}: {len(ps)} parts OK (sha256 {want[:16]}...)")
    return ps


def extract(name, ps):
    src = _Concat(ps)
    try:
        with tarfile.open(fileobj=src, mode="r|gz") as tar:
            for member in tar:
                path = os.path.normpath(member.name)
                if path.startswith("..") or os.path.isabs(path) or not (member.isfile() or member.isdir()):
                    raise SystemExit(f"unsafe entry in bundle: {member.name}")
                if hasattr(tarfile, "data_filter"):
                    tar.extract(member, ROOT, filter="data")
                else:
                    tar.extract(member, ROOT)
                if member.isfile():
                    print(f"[bundle] {member.name} ({member.size / 1e6:.1f} MB)")
    finally:
        src.close()
    manifest = os.path.join(BUNDLE_DIR, f"{name}.files.json")
    if os.path.exists(manifest):
        want = json.load(open(manifest, encoding="utf-8"))["files"]
        missing = [f["path"] for f in want if not os.path.exists(os.path.join(ROOT, f["path"]))]
        if missing:
            raise SystemExit(f"[bundle] extraction incomplete, missing: {missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", help="TO, TO_GLOBAL or all")
    ap.add_argument("--check", action="store_true", help="verify only, do not extract")
    args = ap.parse_args()
    names = [n for n in bundles() if args.split == "all" or n.endswith(f"_{args.split}_bundle")]
    if not names:
        raise SystemExit(f"no bundle for split {args.split} in {BUNDLE_DIR}")
    for name in names:
        ps = verify(name)
        if not args.check:
            extract(name, ps)
    if not args.check:
        print("[bundle] done. e.g.: python run_pipeline.py --k 30 --split TO --steps E --seeds 2020 "
              "--specs IKGR_dyn,IKGR_lgcn_dyn")


if __name__ == "__main__":
    main()
