#!/usr/bin/env python3
"""Measure training speed per spec x batch size x sparse format on the REAL data, without training a model.

For each setting it times a few optimizer steps (forward + BPR loss + backward + Adam step) on uniformly
sampled (user, positive, negative) batches from the split's train interactions, i.e. the same work as one
RecBole training step, and projects the time of one epoch and of a full 40-epoch run. RecBole's own
batch sampling on the CPU and the validation passes are not included (both small next to the graph
propagation of the LightGCN-backbone specs). The dataset is loaded once (several minutes on MovieLens-25M).

Usage (from the repository root, after the data stages):
    python scripts/benchmark_train_speed.py --split TO
    python scripts/benchmark_train_speed.py --split TO --specs LightGCN --batches 2048,16384 --sparse coo,csr
Result: run_k<K>_<SPLIT>/speed_benchmark.json + a table on the console.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


_LOG = None


def log(msg):
    print(msg, flush=True)
    if _LOG:
        with open(_LOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def main():
    global _LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--split", default="TO", choices=["TO", "TO_GLOBAL"])
    ap.add_argument("--specs", default="LightGCN,IKGR_lgcn_dyn_nointent,BPR,IKGR_dyn_nointent")
    ap.add_argument("--batches", default="2048,8192,16384,32768")
    ap.add_argument("--sparse", default="coo,csr", help="formats tried for LightGCN-backbone specs")
    ap.add_argument("--steps", type=int, default=15, help="timed steps per setting")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40, help="epochs used for the full-run projection")
    ap.add_argument("--max_step_sec", type=float, default=30.0, help="skip larger batches once a step is slower")
    args = ap.parse_args()
    sys.argv = sys.argv[:1]                    # RecBole's Config also parses the command line
    os.chdir(ROOT)
    os.environ["IKGR_SPLIT"] = args.split

    import torch
    import yaml
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.data.interaction import Interaction
    from recbole.utils import get_model, init_seed
    import eval_slices as ev

    run_dir = f"run_k{args.k}_{args.split}"
    _LOG = os.path.join(run_dir, "speed_benchmark.log")
    open(_LOG, "w", encoding="utf-8").close()
    cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml"), encoding="utf-8"))
    paths, rb = cfg["paths"], cfg["recbole"]
    specs = ev.build_specs(paths)
    names = [s for s in args.specs.split(",") if s]
    unknown = [s for s in names if s not in specs]
    if unknown:
        raise SystemExit(f"unknown specs {unknown}; known: {sorted(specs)}")
    batches = [int(b) for b in args.batches.split(",")]
    formats = [f for f in args.sparse.split(",") if f]
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    log(f"[bench] {args.split} | device {dev_name} | specs {names} | batches {batches} | sparse {formats}",
          )

    def model_cfg(name, fmt):
        model_arg, extra = specs[name]
        extra = {k: v for k, v in extra.items() if not k.startswith(("corona_", "pop_rerank"))}
        os.environ["IKGR_SPARSE"] = fmt
        cd = ev._config(rb, paths, extra, 2020)
        cd["show_progress"] = False
        return model_arg, Config(model=model_arg, dataset=rb["dataset"], config_dict=cd)

    t0 = time.time()
    _, base_cfg = model_cfg("BPR", "coo")
    init_seed(2020, True)
    dataset = create_dataset(base_cfg)
    train_data, _, _ = data_preparation(base_cfg, dataset)
    inter = train_data.dataset.inter_feat
    uid, iid = base_cfg["USER_ID_FIELD"], base_cfg["ITEM_ID_FIELD"]
    users, items = inter[uid].long(), inter[iid].long()
    n_train, n_items = len(users), train_data.dataset.item_num
    load_sec = time.time() - t0
    log(f"[bench] data loaded in {load_sec:.0f}s: {n_train:,} train interactions, "
          f"{train_data.dataset.user_num - 1:,} users, {n_items - 1:,} items")

    rows = []
    for name in names:
        model_arg, _ = specs[name]
        uses_graph = name == "LightGCN" or "lgcn" in name
        for fmt in (formats if uses_graph else ["coo"]):
            klass, cfg_s = model_cfg(name, fmt)
            klass = klass if not isinstance(klass, str) else get_model(cfg_s["model"])
            init_seed(2020, True)
            model = klass(cfg_s, train_data.dataset).to(cfg_s["device"])
            ev._apply_sparse_format(model, cfg_s)
            if getattr(model, "use_dynamic", False):
                ev._build_recency(model, train_data, cfg_s)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=cfg_s["learning_rate"])
            for bs in batches:
                row = {"spec": name, "sparse": fmt if uses_graph else "-", "batch": bs}
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    times = []
                    for step in range(args.warmup + args.steps):
                        idx = torch.randint(0, n_train, (bs,))
                        neg = torch.randint(1, n_items, (bs,))
                        batch = Interaction({model.USER_ID: users[idx], model.ITEM_ID: items[idx],
                                             model.NEG_ITEM_ID: neg}).to(cfg_s["device"])
                        t = time.perf_counter()
                        opt.zero_grad()
                        loss = model.calculate_loss(batch)
                        if isinstance(loss, tuple):
                            loss = sum(loss)
                        loss.backward()
                        opt.step()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        if step >= args.warmup:
                            times.append(time.perf_counter() - t)
                        if step == args.warmup and times[-1] > args.max_step_sec * 3:
                            break
                    sec = sorted(times)[len(times) // 2]
                    epoch = (n_train / bs) * sec
                    row.update(step_sec=round(sec, 4), epoch_min=round(epoch / 60, 1),
                               run_hours=round(epoch * args.epochs / 3600, 1),
                               peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2)
                               if torch.cuda.is_available() else None)
                except torch.cuda.OutOfMemoryError:
                    row["error"] = "out of GPU memory"
                    torch.cuda.empty_cache()
                rows.append(row)
                log(f"[bench] {name:24s} {row['sparse']:>4s} batch {bs:>6d}: "
                      + (f"{row['step_sec']:.3f} s/step -> {row['epoch_min']:.1f} min/epoch, "
                         f"{row['run_hours']:.1f} h per {args.epochs} epochs, peak {row['peak_gb']} GB"
                         if "error" not in row else row["error"]))
                if "error" in row or row.get("step_sec", 0) > args.max_step_sec:
                    break
            del model, opt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    out = os.path.join(run_dir, "speed_benchmark.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "device": dev_name, "n_train": n_train, "load_sec": round(load_sec),
                   "epochs_projected": args.epochs, "rows": rows,
                   "note": "optimizer steps only; RecBole CPU batch sampling and validation passes excluded"},
                  f, indent=1)
    log(f"[bench] saved {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        log("[bench] FAILED:\n" + traceback.format_exc())
        raise
