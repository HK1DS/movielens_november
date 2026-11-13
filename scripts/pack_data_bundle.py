#!/usr/bin/env python3
"""Pack one split's evaluation inputs into data_bundle/ as <100 MB parts (same scheme as the Goodreads repo).

Teammates restore them with scripts/unpack_data_bundle.py and run the evaluation stage (E) without any
LLM/API call. Run on the machine that holds the finished D stage, from the repository root:

    python scripts/pack_data_bundle.py --split TO
    python scripts/pack_data_bundle.py --split TO_GLOBAL

Writes data_bundle/movielens_k<K>_<SPLIT>_bundle.tgz.partNN, ...bundle.sha256 (whole archive) and
...bundle.files.json (file list with sizes and per-file sha256). Old parts of the same split are replaced.
interactions_k<K>.csv is left out: it holds the same rows as the RecBole .inter and E does not read it.
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import tarfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 99,614,720 bytes per part, like the Goodreads bundle (GitHub rejects files over 100 MB); env override for tests
PART_BYTES = int(os.environ.get("IKGR_BUNDLE_PART_BYTES", 95 * 1024 * 1024))


def bundle_files(k, split):
    d, r = f"data/kc_k{k}", f"run_k{k}_{split}"
    files = [f"{d}/profiles_k{k}_{split}.csv",
             f"{d}/profile_provenance_k{k}_{split}.csv.gz",
             f"{d}/profile_provenance_k{k}_{split}.csv.gz.meta.json",
             f"{r}/kg_pack.pt", f"{r}/meta_kg_pack.pt"]
    if split == "TO":                   # shared by both splits: shipped once, with TO
        files = [f"{d}/ikgr-custom/ikgr-custom.inter", "data/interactions.csv.stats.json",
                 f"{d}/interactions_k{k}.csv.stats.json"] + files
    return files


class PartWriter:
    """File-like sink that cuts the stream into numbered parts and hashes it."""

    def __init__(self, prefix):
        self.prefix, self.n, self.f, self.left = prefix, 0, None, 0
        self.sha = hashlib.sha256()
        self.paths = []

    def write(self, data):
        self.sha.update(data)
        view = memoryview(data)
        while view:
            if self.left == 0:
                if self.f:
                    self.f.close()
                path = f"{self.prefix}{self.n:02d}"
                self.f, self.left = open(path, "wb"), PART_BYTES
                self.paths.append(path)
                self.n += 1
            take = min(self.left, len(view))
            self.f.write(view[:take])
            self.left -= take
            view = view[take:]
        return len(data)

    def flush(self):
        pass

    def close(self):
        if self.f:
            self.f.close()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["TO", "TO_GLOBAL"])
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--level", type=int, default=6, help="gzip level")
    args = ap.parse_args()
    os.chdir(ROOT)
    files = bundle_files(args.k, args.split)
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"[pack] missing (finish the stages first): {missing}")
    name = f"movielens_k{args.k}_{args.split}_bundle"
    os.makedirs("data_bundle", exist_ok=True)
    for old in glob.glob(f"data_bundle/{name}.tgz.part*"):
        os.remove(old)
    t0 = time.time()
    sink = PartWriter(f"data_bundle/{name}.tgz.part")
    gz = gzip.GzipFile(fileobj=sink, mode="wb", compresslevel=args.level, mtime=0)
    with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as tar:
        for f in files:
            info = tar.gettarinfo(f, arcname=f)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with open(f, "rb") as fh:
                tar.addfile(info, fh)
            print(f"[pack] + {f} ({os.path.getsize(f) / 1e6:.1f} MB)", flush=True)
    gz.close()
    sink.close()
    digest = sink.sha.hexdigest()
    with open(f"data_bundle/{name}.sha256", "w", encoding="ascii", newline="\n") as f:
        f.write(f"{digest}  {name}.tgz (parts concatenated in order)\n")
    manifest = {"bundle": name, "split": args.split, "k": args.k, "parts": [os.path.basename(p) for p in sink.paths],
                "archive_sha256": digest, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "files": [{"path": f, "bytes": os.path.getsize(f), "sha256": sha256_file(f)} for f in files]}
    with open(f"data_bundle/{name}.files.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1)
    total = sum(os.path.getsize(p) for p in sink.paths)
    print(f"[pack] {len(sink.paths)} parts, {total / 1e6:.0f} MB, {time.time() - t0:.0f}s -> data_bundle/{name}.*")


if __name__ == "__main__":
    main()
