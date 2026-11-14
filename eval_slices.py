#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-seed sliced evaluation: overall vs long-tail vs cold-start vs coverage/novelty,
with robustness across seeds.

For each spec x seed: train under one shared protocol (early stopping on the
validation split), compute per-user metrics on the test split, slice by
TRAIN-ONLY user activity (cold-start) and TRAIN-ONLY item popularity (long-tail),
plus catalog coverage@10 and novelty. Aggregates mean +/- std over seeds and
writes per-user metric tables for paired significance tests (paired_compare.py).

Leakage guard: if `paths.profile_provenance` is set (build_trainonly_profiles.py
writes it), every (user, item) that contributed to an LLM user profile must be
in RecBole's actual TRAIN split for this split protocol, and every user that has
intents in the KG pack must have such a profile. Otherwise the run aborts.

Output: <workdir>/slice_eval_<SPLIT>_result.json (RS: slice_eval_result.json)
        <workdir>/per_user_<SPLIT>/<result>__s<seed>__test.csv.gz (+ __valid with IKGR_DUMP_VALID=1)

Env overrides:
  IKGR_SPLIT=TO|TO_GLOBAL|RS      split protocol (default RS)
  IKGR_SEEDS=2020,2021,2022       seeds
  IKGR_SPECS=IKGR_kgoff,BPR       subset of specs
  IKGR_EPOCHS=40                  max epochs (early stopping may stop sooner)
  IKGR_EVAL_STEP=2                validate every N epochs
  IKGR_STOPPING_STEP=3            stop after N validations without improvement
  IKGR_VALID_METRIC=NDCG@10       model-selection metric on the validation split
  IKGR_LR / IKGR_REG              learning rate / reg weight (same for every spec)
  IKGR_POP_RERANK_GRID=0.25,0.5,1 popularity-penalty rerank lambdas (eval-only, no retraining)
  IKGR_POP_RERANK_SPECS=...       specs that get the pop-rerank variants
  IKGR_DUMP_VALID=1               also dump per-user VALIDATION metrics (for gate/selection work)
  IKGR_REQUIRE_PROVENANCE=1       refuse to run KG specs without a profile provenance file
"""
import os, json, time, math, yaml, hashlib, random, gzip
import scipy.sparse as _sp
if not hasattr(_sp.dok_matrix, "_update"):
    # RecBole 1.2.0 builds LightGCN's normalized adjacency with dok_matrix._update(),
    # which SciPy >= 1.13 removed. There the entries live in self._dict; the old shim
    # (`_update = dict.update`) wrote into the unused dict base instead, so the
    # adjacency was silently EMPTY and "LightGCN" never propagated (it trained as MF
    # with layer-mean scaling). Write into the real store.
    def _dok_update(self, data):
        store = getattr(self, "_dict", None)
        if store is None:
            dict.update(self, data)
        else:
            store.update(data)
    _sp.dok_matrix._update = _dok_update
import numpy as np
import pandas as pd
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_seed
from ikgr_core.model_ikgr import IKGR as IKGRModel

KS = [10, 30]
MAXK = max(KS)
COV_K = 10
EVAL_SCHEMA_VERSION = 4   # 3: train-only pop/activity, early stopping, leakage guard
                          # 4: machine-independent signature (only the data files a spec uses, content-based
                          #    hashes, EOL-normalised code, repo-relative paths) so teammates' seeds can be merged


def _set_determinism(seed):
    """Reset every RNG used by this custom eval path before each training run."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    init_seed(seed, reproducibility=True)


def _protocol(rb):
    """Training/model-selection protocol shared by EVERY spec (comparable budgets)."""
    return {
        "epochs": int(os.environ.get("IKGR_EPOCHS", rb.get("epochs", 40))),
        "eval_step": int(os.environ.get("IKGR_EVAL_STEP", rb.get("eval_step", 2))),
        "stopping_step": int(os.environ.get("IKGR_STOPPING_STEP", rb.get("stopping_step", 3))),
        "valid_metric": os.environ.get("IKGR_VALID_METRIC", rb.get("valid_metric", "NDCG@10")),
        "learning_rate": float(os.environ.get("IKGR_LR", rb.get("learning_rate", 1e-3))),
        "reg_weight": float(os.environ.get("IKGR_REG", rb.get("reg_weight", 1e-6))),
        # Training batch size: LightGCN-backbone specs propagate the WHOLE graph once per batch, so on
        # MovieLens-25M (~39M adjacency entries) the epoch time is ~ n_batches x graph size.
        "train_batch_size": int(os.environ.get("IKGR_BATCH", rb.get("train_batch_size", 2048))),
        # Sparse format of the LightGCN adjacency (coo | csr): same math, csr is usually faster on GPU.
        "sparse_format": os.environ.get("IKGR_SPARSE", rb.get("sparse_format", "coo")).lower(),
        # Users per full-sort evaluation batch (validation + test). RecBole's eval_batch_size counts
        # user x item pairs, so its default 4096 is ONE user per batch on MovieLens (15,786 items) and the
        # TO validation took ~10 min. Scores do not depend on the batch except for float summation order
        # (a few near-tied top-30 lists flip); it is part of the protocol so runs that differ never merge.
        "eval_batch_users": int(os.environ.get("IKGR_EVAL_USERS", rb.get("eval_batch_users", 1))),
    }


_SHA_MEMO = {}


def _canonical_update(h, obj):
    """Feed a loaded KG pack (dicts / lists / tensors / arrays / scalars) into h, order-independently for dicts."""
    import numpy as _np
    try:
        import torch as _torch
    except Exception:
        _torch = None
    if _torch is not None and isinstance(obj, _torch.Tensor):
        obj = obj.detach().cpu().contiguous().numpy()
    if isinstance(obj, _np.ndarray):
        a = _np.ascontiguousarray(obj)
        h.update(f"nd|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    elif isinstance(obj, dict):
        h.update(f"dict|{len(obj)}|".encode())
        for k in sorted(obj, key=lambda x: repr(x)):
            h.update(repr(k).encode("utf-8"))
            _canonical_update(h, obj[k])
    elif isinstance(obj, (list, tuple)):
        h.update(f"{type(obj).__name__}|{len(obj)}|".encode())
        for v in obj:
            _canonical_update(h, v)
    else:
        h.update(f"{type(obj).__name__}|{obj!r}|".encode("utf-8"))


def _file_sha256(path, chunk=1 << 22):
    """Content fingerprint that ignores byte-level noise between machines / rebuilds:
    .gz -> hash of the decompressed bytes (gzip headers carry a timestamp),
    .pt -> hash of the loaded tensors and values (torch.save bytes can differ across versions),
    anything else -> raw bytes."""
    key = (os.path.abspath(path), os.path.getsize(path), os.path.getmtime(path))
    if key not in _SHA_MEMO:
        h = hashlib.sha256()
        if path.endswith(".pt"):
            import torch
            try:
                obj = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:                                   # torch < 1.13
                obj = torch.load(path, map_location="cpu")
            _canonical_update(h, obj)
        else:
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rb") as f:
                for block in iter(lambda: f.read(chunk), b""):
                    h.update(block)
        _SHA_MEMO[key] = h.hexdigest()[:16]
    return _SHA_MEMO[key]


def _code_sha256(path):
    """Source hash with line endings normalised (a Windows checkout may turn LF into CRLF)."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def _portable(value):
    """Absolute paths inside a spec's options -> repo-relative, forward slashes (same on every machine)."""
    if isinstance(value, str) and os.path.isabs(value):
        try:
            return os.path.relpath(value).replace("\\", "/")
        except ValueError:                                      # other drive on Windows
            return os.path.basename(value)
    return value


def _spec_data_files(extra):
    """The data files a spec actually reads: only these enter its fingerprint."""
    uses_intents = bool(extra.get("use_kg")) or bool(extra.get("kg_pack_path") and (
        extra.get("corona_cand") or extra.get("corona_rerank_grid")))
    keys = []
    if uses_intents:
        keys += ["profile_provenance", "kg_pack"]
    if extra.get("use_meta_kg"):
        keys.append("meta_kg_pack")
    return keys


def _experiment_context(name, model_arg, extra, rb, paths, split):
    """Fingerprint evaluation inputs so stale seed results cannot be reused."""
    source_paths = [__file__, "ikgr_core/model_ikgr.py", "ikgr_core/corona_retriever.py"]
    code = {}
    for path in source_paths:
        rel = _portable(os.path.abspath(path))
        try:
            code[rel] = _code_sha256(path)
        except OSError:
            code[rel] = None
    # Content hashes (not mtimes) of only the files this spec reads: re-creating identical inputs on resume,
    # rebuilding them on another machine, or adding a KG pack a spec does not use must not invalidate
    # finished seeds; genuinely changed inputs must.
    data_fp = {}
    for key in _spec_data_files(extra):
        p = paths.get(key)
        if p and os.path.exists(p):
            data_fp[key] = _file_sha256(p)
    context = {
        "schema": EVAL_SCHEMA_VERSION,
        "spec": name,
        "model": model_arg if isinstance(model_arg, str) else model_arg.__name__,
        "split": split,
        "protocol": _protocol(rb),
        "extra": {k: _portable(v) for k, v in extra.items()},
        "paths": {k: _portable(v) for k, v in paths.items()},
        "data_fingerprint": data_fp,
        "recbole": rb,
        "code": code,
    }
    encoded = json.dumps(context, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16], context


def _float_grid_from_env(name, default):
    raw = os.environ.get(name)
    if not raw:
        return default
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(value < 0 for value in values):
        raise ValueError(f"{name} must contain one or more non-negative floats")
    return values


def _apply_eval_batch(config, dataset):
    """Full-sort evaluation batches of `eval_batch_users` users: RecBole takes eval_batch_size // item_num
    users per batch, so the size in user x item pairs is set once the item count is known."""
    users = max(1, int(config["eval_batch_users"] or 1))
    config["eval_batch_size"] = users * dataset.item_num
    return users


def _lam_tag(lam):
    return str(lam).replace(".", "p")


def _config(rb, paths, extra, seed):
    split = os.environ.get("IKGR_SPLIT", "RS").upper()
    is_temporal = split in ("TO", "TO_GLOBAL")
    load_inter = ["user_id", "item_id", "rating"] + (["timestamp"] if is_temporal else [])
    proto = _protocol(rb)
    cd = {
        "epochs": proto["epochs"], "eval_step": proto["eval_step"],
        "stopping_step": proto["stopping_step"], "valid_metric": proto["valid_metric"],
        "metrics": rb["metrics"], "topk": rb["topk"],
        "embedding_size": rb["embedding_size"], "learning_rate": proto["learning_rate"],
        "reg_weight": proto["reg_weight"],
        "train_batch_size": proto["train_batch_size"], "sparse_format": proto["sparse_format"],
        "eval_batch_users": proto["eval_batch_users"],
        "dropout_prob": rb.get("dropout", 0.1),
        "data_path": os.path.dirname(paths["inter_file"]),
        "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id", "LABEL_FIELD": "rating",
        "load_col": {"inter": load_inter},
        "train_neg_sample_args": {"distribution": "uniform"},
        "save_dataset": False, "save_dataloaders": False, "show_progress": False,
        "checkpoint_dir": os.path.abspath(os.path.join(paths.get("workdir", "run/"), "recbole_slice")),
        "seed": seed, "reproducibility": True,
    }
    if split == "TO":
        # temporal (per-user time-ordered) split: train=earliest, test=latest
        cd["TIME_FIELD"] = "timestamp"
        cd["eval_args"] = {"split": {"RS": [0.8, 0.1, 0.1]}, "order": "TO",
                           "group_by": "user", "mode": "full"}
    elif split == "TO_GLOBAL":
        # GLOBAL temporal split (single time cutoff over ALL interactions): users
        # active only after the cutoff are genuine cold-start users.
        # Keep in sync with ikgr_core/temporal_split.py SPLITS.
        cd["TIME_FIELD"] = "timestamp"
        cd["eval_args"] = {"split": {"RS": [0.7, 0.1, 0.2]}, "order": "TO",
                           "group_by": None, "mode": "full"}
    cd.update(extra)
    return cd


def _apply_sparse_format(model, config):
    """RecBole's LightGCN keeps its normalised adjacency as COO; switch it to CSR when the protocol says so
    (IKGR's own LightGCN backbone reads `sparse_format` itself)."""
    fmt = str(config["sparse_format"] or "coo").lower()
    if fmt not in ("coo", "csr"):
        raise ValueError(f"sparse_format must be coo or csr, not {fmt!r}")
    if fmt == "csr" and hasattr(model, "norm_adj_matrix") and model.norm_adj_matrix.layout == torch.sparse_coo:
        model.norm_adj_matrix = model.norm_adj_matrix.coalesce().to_sparse_csr()


def _build_recency(model, train_data, config):
    """Per-user recency profile from TRAIN interactions only (no leakage):
    top-N most recent items + exp-decay weights -> model.set_recency()."""
    from collections import defaultdict
    inter = train_data.dataset.inter_feat
    uf, itf = config["USER_ID_FIELD"], config["ITEM_ID_FIELD"]
    if "timestamp" not in inter.columns:
        raise RuntimeError("use_dynamic requires timestamp (run with IKGR_SPLIT=TO).")
    u = inter[uf].numpy(); it = inter[itf].numpy(); ts = inter["timestamp"].numpy()
    N = model.recency_topn
    tau = max(model.recency_tau_days * 86400.0, 1.0)
    byu = defaultdict(list)
    for a, b, c in zip(u, it, ts):
        byu[int(a)].append((float(c), int(b)))
    ids = np.zeros((model.n_users, N), dtype=np.int64)
    wts = np.zeros((model.n_users, N), dtype=np.float32)
    for uid, lst in byu.items():
        lst.sort(reverse=True)            # most recent first
        lst = lst[:N]
        tref = lst[0][0]
        w = np.exp(-np.array([tref - c for c, _ in lst], dtype=np.float64) / tau)
        s = w.sum()
        w = (w / s) if s > 0 else w
        for j, (c, b) in enumerate(lst):
            ids[uid, j] = b; wts[uid, j] = w[j]
    dev = next(model.parameters()).device
    model.set_recency(torch.from_numpy(ids).to(dev), torch.from_numpy(wts).to(dev))
    print(f"  [recency] built for {len(byu)} users from {len(u)} train inters (N={N})", flush=True)


def _train_stats(train_data, config):
    """TRAIN-split item popularity and user activity (slice definitions)."""
    ds = train_data.dataset
    inter = ds.inter_feat
    u = inter[config["USER_ID_FIELD"]].numpy()
    it = inter[config["ITEM_ID_FIELD"]].numpy()
    item_pop = np.bincount(it, minlength=int(ds.item_num)).astype(np.int64)
    user_act = np.bincount(u, minlength=int(ds.user_num)).astype(np.int64)
    item_pop[0] = 0  # padding id
    return item_pop, user_act


def _check_profile_leakage(paths, extra, train_data, config, split):
    """Abort if any LLM-profile evidence lies outside RecBole's train split."""
    prov_path = paths.get("profile_provenance")
    uses_intents = bool(extra.get("use_kg")) or bool(extra.get("kg_pack_path") and (
        extra.get("corona_cand") or extra.get("corona_rerank_grid")))
    if not prov_path:
        if uses_intents and os.environ.get("IKGR_REQUIRE_PROVENANCE") == "1":
            raise RuntimeError("[leakage-guard] KG spec without paths.profile_provenance "
                               "(IKGR_REQUIRE_PROVENANCE=1). Build profiles with build_trainonly_profiles.py.")
        if uses_intents:
            print("  [leakage-guard] WARNING: no profile_provenance configured -> user intents NOT verified "
                  "to be train-only.", flush=True)
        return {"verified": False}
    meta_path = prov_path + ".meta.json"
    meta = json.load(open(meta_path, encoding="utf-8"))
    if meta.get("split") != split:
        raise RuntimeError(f"[leakage-guard] profiles were built for split={meta.get('split')} "
                           f"but this run evaluates split={split}. Use the matching run directory.")
    prov = pd.read_csv(prov_path, dtype=str, keep_default_na=False)
    ds = train_data.dataset
    uf, itf = config["USER_ID_FIELD"], config["ITEM_ID_FIELD"]
    u2id, i2id = ds.field2token_id[uf], ds.field2token_id[itf]
    n_items = int(ds.item_num)
    inter = ds.inter_feat
    train_keys = np.unique(inter[uf].numpy().astype(np.int64) * n_items + inter[itf].numpy().astype(np.int64))
    pu = prov["user_id"].map(u2id)
    pi = prov["item_id"].map(i2id)
    unknown = int(pu.isna().sum() + pi.isna().sum())
    ok = ~(pu.isna() | pi.isna())
    keys = pu[ok].astype(np.int64).to_numpy() * n_items + pi[ok].astype(np.int64).to_numpy()
    outside = int((~np.isin(keys, train_keys)).sum())
    users_with_profile = set(prov["user_id"].unique())
    stale = 0
    kg_path = extra.get("kg_pack_path")
    if kg_path and os.path.exists(kg_path):
        pack = torch.load(kg_path, map_location="cpu")
        stale = len([u for u in pack["user_intents"].keys() if str(u) not in users_with_profile])
    if unknown or outside or stale:
        raise RuntimeError(f"[leakage-guard] FAILED: {outside} profile pairs outside the train split, "
                           f"{unknown} unknown tokens, {stale} KG users without a train-only profile. "
                           f"Rebuild profiles -> step1 -> step2 -> KG for split={split}.")
    print(f"  [leakage-guard] OK: {len(prov)} profile pairs ({len(users_with_profile)} users) all in the "
          f"{split} train split; KG user intents consistent.", flush=True)
    return {"verified": True, "pairs": int(len(prov)), "users_with_profile": len(users_with_profile)}


def _blend_rerank_scores(scores, graph_prior, score_scale, lam):
    """Apply a relative graph prior without altering the zero-lambda control."""
    if lam is None or lam == 0.0:
        return scores
    return scores + lam * score_scale * graph_prior


def _pop_penalty_scores(scores, pop_z, score_scale, lam):
    """Popularity-penalty rerank baseline: s - lam * sigma_u * z(log(1+pop_train))."""
    if lam is None or lam == 0.0:
        return scores
    return scores - lam * score_scale * pop_z


def _variants(rerank_lambdas, pop_lambdas):
    """(suffix, kind, lam) list. One trained model serves every variant."""
    if rerank_lambdas:
        return [(f"_l{_lam_tag(l)}", "graph", l) for l in rerank_lambdas]
    return [("", None, None)] + [(f"_pop{_lam_tag(l)}", "pop", l) for l in pop_lambdas if l > 0]


def _collect(model, eval_data, config, variants, retriever, cand_m, pop_z, n_items, want_cand_stats):
    dev = config["device"]
    uid_field = config["USER_ID_FIELD"]
    per_user = {v[0]: [] for v in variants}
    cand_rec_sum, cand_size_sum, cand_n = 0.0, 0, 0
    need_scale = any(kind in ("graph", "pop") for _, kind, _ in variants)
    with torch.no_grad():
        for interaction, history_index, positive_u, positive_i in eval_data:
            interaction = interaction.to(dev)
            users = interaction[uid_field]
            B = users.shape[0]
            scores = model.full_sort_predict(interaction).view(B, -1)
            scores[:, 0] = -np.inf
            users_np = users.cpu().numpy()
            cand_sets = None
            # Only explicit candidate retrieval masks scores (soft rerank keeps full-sort).
            if cand_m > 0:
                cand = retriever.candidates(users_np, cand_m)      # [B, M]
                cand_mask = torch.zeros((B, n_items), dtype=torch.bool, device=dev)
                cand_mask.scatter_(1, torch.from_numpy(cand).to(dev), True)
                cand_mask[:, 0] = False
                scores = scores.masked_fill(~cand_mask, -np.inf)
                cand_sets = [set(row.tolist()) for row in cand]
            if history_index is not None:
                scores[history_index] = -np.inf
            graph_prior = None
            score_scale = None
            if need_scale:
                # Express lambda in each user's finite full-sort score std, so a
                # bonus/penalty has the same meaning for every model/seed.
                finite = torch.isfinite(scores)
                n_finite = finite.sum(dim=1, keepdim=True).clamp_min(1)
                safe_scores = scores.masked_fill(~finite, 0.0)
                mean = safe_scores.sum(dim=1, keepdim=True) / n_finite
                var = ((safe_scores - mean).square() * finite).sum(dim=1, keepdim=True) / n_finite
                score_scale = var.sqrt().clamp_min(1e-6)
            if any(kind == "graph" for _, kind, _ in variants):
                graph_prior = torch.from_numpy(retriever.normalized_prior(users_np)).to(dev)
            pu = positive_u.cpu().numpy(); pi = positive_i.cpu().numpy()
            rel_by_row = {}
            for r, it in zip(pu, pi):
                rel_by_row.setdefault(int(r), []).append(int(it))
            for suffix, kind, lam in variants:
                if kind == "graph":
                    rank_scores = _blend_rerank_scores(scores, graph_prior, score_scale, lam)
                elif kind == "pop":
                    rank_scores = _pop_penalty_scores(scores, pop_z, score_scale, lam)
                else:
                    rank_scores = scores
                topk = torch.topk(rank_scores, MAXK, dim=-1)[1].cpu().numpy()
                for row in range(B):
                    rel = rel_by_row.get(row)
                    if not rel:
                        continue
                    if want_cand_stats and cand_sets is not None and suffix == variants[0][0]:
                        cs = cand_sets[row]
                        cand_rec_sum += sum(1 for it in rel if it in cs) / len(rel)
                        cand_size_sum += len(cs); cand_n += 1
                    per_user[suffix].append({"uid": int(users_np[row]), "topk": topk[row].tolist(), "rel": rel})
    cand_stats = {}
    if want_cand_stats and cand_m > 0 and cand_n:
        cand_stats = {"cand_recall": round(cand_rec_sum / cand_n, 4),
                      "cand_size": round(cand_size_sum / cand_n, 1), "cand_M": cand_m}
    return per_user, cand_stats


def _train_and_collect(model_arg, extra, rb, paths, seed, split):
    _set_determinism(seed)
    extra = dict(extra)
    # Eval-only knobs (candidate restriction / reranking): not model parameters.
    cand_m = int(extra.pop("corona_cand", 0))
    cand_cf = bool(extra.pop("corona_cf", True))
    cand_idf = bool(extra.pop("corona_idf", False))
    cand_popnorm = float(extra.pop("corona_popnorm", 0.0))
    cand_weights = extra.pop("corona_weights", None)
    rerank_lambdas = [float(x) for x in extra.pop("corona_rerank_grid", [])]
    pop_lambdas = [float(x) for x in extra.pop("pop_rerank_grid", [])]
    guard_extra = dict(extra, corona_cand=cand_m, corona_rerank_grid=rerank_lambdas)
    cfg_dict = _config(rb, paths, extra, seed)
    os.makedirs(cfg_dict["checkpoint_dir"], exist_ok=True)
    config = Config(model=model_arg, dataset=rb["dataset"], config_dict=cfg_dict)
    dataset = create_dataset(config)
    _apply_eval_batch(config, dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    guard = _check_profile_leakage(paths, guard_extra, train_data, config, split)
    item_pop, user_act = _train_stats(train_data, config)
    klass = model_arg if not isinstance(model_arg, str) else get_model(config["model"])
    model = klass(config, train_data.dataset).to(config["device"])
    _apply_sparse_format(model, config)
    if getattr(model, "use_dynamic", False):
        _build_recency(model, train_data, config)
    retriever = None
    if cand_m > 0 or rerank_lambdas:
        from ikgr_core.corona_retriever import CoronaRetriever
        retriever = CoronaRetriever(train_data, config,
                                    kg_pack_path=extra.get("kg_pack_path"),
                                    meta_kg_path=extra.get("meta_kg_path"),
                                    weights=cand_weights, use_cf=cand_cf,
                                    idf=cand_idf, pop_norm=cand_popnorm)
        mode = f"M={cand_m}" if cand_m > 0 else f"soft_rerank={rerank_lambdas}"
        print(f"  [corona] retriever ready ({mode}, cf={cand_cf}, "
              f"idf={cand_idf}, pop_norm={cand_popnorm})", flush=True)
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    fit_info = {"best_epoch": None, "last_epoch": None}
    _orig_save = trainer._save_checkpoint

    def _save_and_track(epoch, *a, **kw):
        fit_info["best_epoch"] = int(epoch)
        return _orig_save(epoch, *a, **kw)
    trainer._save_checkpoint = _save_and_track

    def _cb(epoch_idx, valid_score):
        fit_info["last_epoch"] = int(epoch_idx)
    t0 = time.time()
    best_score, best_result = trainer.fit(train_data, valid_data, saved=True, show_progress=False,
                                          callback_fn=_cb)
    train_sec = round(time.time() - t0, 1)
    fit_info["best_valid_score"] = None if best_score is None else float(best_score)
    fit_info["max_epochs"] = int(config["epochs"])
    smf = getattr(trainer, "saved_model_file", None)
    if smf and os.path.exists(smf):
        ck = torch.load(smf, map_location=config["device"])
        model.load_state_dict(ck["state_dict"])
        if ck.get("other_parameter"):
            model.load_other_parameter(ck["other_parameter"])
    model.eval()

    n_items = int(dataset.item_num)
    if hasattr(model, "eval_cache_items"):
        model.eval_cache_items()
    lp = np.log1p(item_pop[1:].astype(np.float64))
    z = np.zeros(n_items, dtype=np.float32)
    z[1:] = (lp - lp.mean()) / max(lp.std(), 1e-9)
    pop_z = torch.from_numpy(z).to(config["device"]).unsqueeze(0)
    variants = _variants(rerank_lambdas, pop_lambdas)
    test_pu, cand_stats = _collect(model, test_data, config, variants, retriever, cand_m, pop_z,
                                   n_items, want_cand_stats=True)
    valid_pu = None
    if os.environ.get("IKGR_DUMP_VALID") == "1":
        valid_pu, _ = _collect(model, valid_data, config, variants, retriever, cand_m, pop_z,
                               n_items, want_cand_stats=False)
    for pu_by in (test_pu, valid_pu or {}):
        for rows in pu_by.values():
            for r in rows:
                r["activity"] = int(user_act[r["uid"]])
    uid_tokens = dataset.field2id_token[config["USER_ID_FIELD"]]
    return {"test": test_pu, "valid": valid_pu, "item_pop": item_pop, "train_sec": train_sec,
            "n_items": n_items, "cand_stats": cand_stats, "fit": fit_info, "guard": guard,
            "uid_tokens": uid_tokens}


def _user_metrics(topk, rel_set, k):
    hits = [r for r, it in enumerate(topk[:k]) if it in rel_set]
    n_rel = len(rel_set)
    recall = len(hits) / n_rel if n_rel else 0.0
    dcg = sum(1.0 / math.log2(r + 2) for r in hits)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n_rel, k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return recall, ndcg, (1.0 if hits else 0.0)


def _avg(rows, is_tail=None):
    agg = {}
    for k in KS:
        agg[f"recall@{k}"] = 0.0; agg[f"ndcg@{k}"] = 0.0; agg[f"hit@{k}"] = 0.0
    n = 0
    for topk, rel in rows:
        if is_tail is not None:
            rel = [it for it in rel if is_tail[it]]
            if not rel:
                continue
        rel_set = set(rel); n += 1
        for k in KS:
            rc, nd, ht = _user_metrics(topk, rel_set, k)
            agg[f"recall@{k}"] += rc; agg[f"ndcg@{k}"] += nd; agg[f"hit@{k}"] += ht
    if n:
        for kk in agg:
            agg[kk] = round(agg[kk] / n, 4)
    return agg, n


def tail_mask(item_pop):
    """Long tail = items at or below the 80th percentile of TRAIN popularity
    (among items seen in train); unseen items are tail. Pad id 0 excluded."""
    pos = item_pop[item_pop > 0]
    head_cut = float(np.percentile(pos, 80)) if pos.size else 0.0
    is_tail = item_pop <= head_cut
    is_tail[0] = False
    return is_tail, head_cut


def slice_report(per_user, item_pop, n_items):
    """per_user rows: {uid, activity (TRAIN interactions), topk, rel}; item_pop = TRAIN popularity."""
    is_tail, head_cut = tail_mask(item_pop)

    rows_all = [(r["topk"], r["rel"]) for r in per_user]
    overall, n_all = _avg(rows_all)
    tail, n_tail = _avg(rows_all, is_tail=is_tail)

    # cold-start buckets by TRAIN-activity quantiles
    acts = np.array([r["activity"] for r in per_user])
    q = np.percentile(acts, [20, 40, 60, 80]) if acts.size else [0, 0, 0, 0]
    def b(a):
        return ("Q1_cold" if a <= q[0] else "Q2" if a <= q[1] else "Q3" if a <= q[2]
                else "Q4" if a <= q[3] else "Q5_warm")
    buckets = {}
    for r in per_user:
        buckets.setdefault(b(r["activity"]), []).append((r["topk"], r["rel"]))
    bucket_metrics = {}
    for name, rows in buckets.items():
        m, n = _avg(rows)
        bucket_metrics[name] = {"n_users": n, "ndcg@10": m["ndcg@10"], "recall@10": m["recall@10"]}

    # ABSOLUTE cold-start buckets by # TRAIN interactions (meaningful under TO_GLOBAL).
    abs_buckets = {}
    for name, lo, hi in [("cold0_train", 0, 0), ("cold_1_5", 1, 5),
                         ("cold_6_20", 6, 20), ("warm_gt20", 21, 10**9)]:
        rows = [(r["topk"], r["rel"]) for r in per_user if lo <= r["activity"] <= hi]
        m, n = _avg(rows)
        abs_buckets[name] = {"n_users": n, "ndcg@10": m["ndcg@10"],
                             "recall@10": m["recall@10"], "recall@30": m["recall@30"]}

    # coverage@10 and novelty (self-information under TRAIN popularity)
    rec_items = set()
    pop_sum, pop_cnt = 0.0, 0
    total = max(1, int(item_pop.sum()))
    novelty = 0.0
    for r in per_user:
        for it in r["topk"][:COV_K]:
            rec_items.add(it)
            p = item_pop[it] / total
            novelty += -math.log2(p) if p > 0 else -math.log2(1.0 / total)
            pop_sum += item_pop[it]; pop_cnt += 1
    coverage = round(len(rec_items) / max(1, n_items - 1), 4)  # exclude pad item
    novelty = round(novelty / max(1, pop_cnt), 4)
    avg_rec_pop = round(pop_sum / max(1, pop_cnt), 1)

    return {
        "overall": overall, "n_test_users": n_all,
        "long_tail": {"head_cut_pop": head_cut, "n_tail_items": int(is_tail.sum()),
                      "n_users_with_tail_rel": n_tail, **tail},
        "cold_start_buckets": dict(sorted(bucket_metrics.items())),
        "cold_abs_buckets": abs_buckets,
        "activity_q_20_40_60_80": [float(x) for x in q],
        "coverage@10": coverage, "novelty_bits": novelty, "avg_rec_popularity": avg_rec_pop,
        "definitions": {"activity": "train interactions", "popularity": "train interactions"},
    }


def per_user_table(per_user, item_pop, uid_tokens):
    """One row per evaluated user: the unit of the paired tests in paired_compare.py."""
    is_tail, _ = tail_mask(item_pop)
    out = []
    for r in per_user:
        rel_set = set(r["rel"])
        row = {"user_id": uid_tokens[r["uid"]], "activity_train": r["activity"], "n_rel": len(rel_set)}
        for k in KS:
            rc, nd, ht = _user_metrics(r["topk"], rel_set, k)
            row[f"recall@{k}"] = rc; row[f"ndcg@{k}"] = nd; row[f"hit@{k}"] = ht
        tail_rel = {it for it in rel_set if is_tail[it]}
        row["n_rel_tail"] = len(tail_rel)
        if tail_rel:
            rc, nd, _ = _user_metrics(r["topk"], tail_rel, 10)
            row["tail_recall@10"] = rc; row["tail_ndcg@10"] = nd
        else:
            row["tail_recall@10"] = np.nan; row["tail_ndcg@10"] = np.nan
        row["rec_pop_mean@10"] = float(np.mean([item_pop[it] for it in r["topk"][:COV_K]]))
        out.append(row)
    return pd.DataFrame(out)


def _aggregate(per_seed):
    """mean/std over seeds for headline metrics."""
    keys = [("overall", "ndcg@10"), ("overall", "recall@10"),
            ("long_tail", "recall@10"), ("long_tail", "recall@30"),
            ("coverage@10", None), ("novelty_bits", None)]
    out = {}
    for grp, sub in keys:
        vals = []
        for rep in per_seed.values():
            v = rep[grp] if sub is None else rep[grp][sub]
            vals.append(float(v))
        label = grp if sub is None else f"{grp}.{sub}"
        out[label] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4),
                      "seeds": vals}
    eps = [rep.get("fit", {}).get("best_epoch") for rep in per_seed.values()]
    eps = [e for e in eps if e is not None]
    if eps:
        out["best_epoch"] = {"mean": round(float(np.mean(eps)), 1), "seeds": eps}
    return out


def build_specs(paths):
    kg = os.path.abspath(paths.get("kg_pack", "run/kg_pack.pt"))
    meta = os.path.abspath(paths.get("meta_kg_pack", "run/meta_kg_pack.pt"))
    return {
        "IKGR_kgoff":   (IKGRModel, {"use_kg": False}),
        "IKGR_kgon_L1": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1, "kg_cap": 32}),
        "IKGR_kgon_L1_frozen": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1,
                                            "kg_cap": 32, "intent_learnable": False}),
        "IKGR_kgon_L2": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 2, "kg_cap": 32}),
        # heterogeneous metadata KG (adapter.meta_relations, e.g. genre/author/...), LLM-free
        "IKGR_meta_only": (IKGRModel, {"use_kg": False, "use_meta_kg": True, "meta_kg_path": meta, "kg_cap": 32}),
        "IKGR_full_hetero": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1, "kg_cap": 32,
                                         "intent_learnable": False, "use_meta_kg": True, "meta_kg_path": meta}),
        "IKGR_dyn": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1, "kg_cap": 32,
                                 "intent_learnable": False, "use_meta_kg": True, "meta_kg_path": meta,
                                 "use_dynamic": True}),
        # Same fusion structure as IKGR_dyn with the LLM-intent channel switched off:
        # isolates the contribution of intent (metadata + recency remain).
        "IKGR_dyn_nointent": (IKGRModel, {"use_kg": False, "kg_cap": 32,
                                          "use_meta_kg": True, "meta_kg_path": meta,
                                          "use_dynamic": True}),
        # Same facets (metadata KG + recency, +/- LLM intent) on a LightGCN backbone:
        # base user/item embeddings are propagated over the train graph (2 layers,
        # as RecBole LightGCN's default) before the facets are added.
        "IKGR_lgcn_dyn_nointent": (IKGRModel, {"use_kg": False, "kg_cap": 32,
                                               "use_meta_kg": True, "meta_kg_path": meta,
                                               "use_dynamic": True, "lgcn_layers": 2}),
        "IKGR_lgcn_dyn": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1, "kg_cap": 32,
                                      "intent_learnable": False, "use_meta_kg": True, "meta_kg_path": meta,
                                      "use_dynamic": True, "lgcn_layers": 2}),
        "IKGR_dyn_attn": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1, "kg_cap": 32,
                                      "intent_learnable": False, "use_meta_kg": True, "meta_kg_path": meta,
                                      "use_dynamic": True, "profile_attn": True}),
        # CORONA-inspired per-channel weighted-sum late fusion (CF / intent+meta / recency).
        "IKGR_full": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1, "kg_cap": 32,
                                  "intent_learnable": False, "use_meta_kg": True, "meta_kg_path": meta,
                                  "use_dynamic": True, "use_corona": True}),
        # Graph-neighbour candidate generation (eval-only top-M restriction).
        "IKGR_cand": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1, "kg_cap": 32,
                                  "intent_learnable": False, "use_meta_kg": True, "meta_kg_path": meta,
                                  "use_dynamic": True, "corona_cand": 500}),
        # De-biased candidates: CF channel off, IDF node weights, score / item_pop^0.5.
        "IKGR_cand_db": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1, "kg_cap": 32,
                                     "intent_learnable": False, "use_meta_kg": True, "meta_kg_path": meta,
                                     "use_dynamic": True, "corona_cand": 500,
                                     "corona_cf": False, "corona_idf": True, "corona_popnorm": 0.5}),
        # Soft graph-prior reranking over full-sort scores (one training run per seed).
        "IKGR_rerank_db_rel": (IKGRModel, {"use_kg": True, "kg_pack_path": kg, "kg_layers": 1, "kg_cap": 32,
                                           "intent_learnable": False, "use_meta_kg": True, "meta_kg_path": meta,
                                           "use_dynamic": True,
                                           "corona_rerank_grid": _float_grid_from_env(
                                               "IKGR_CORONA_RERANK_GRID", [0.0, 0.1, 0.25, 0.5]),
                                           "corona_cf": False, "corona_idf": True, "corona_popnorm": 0.5}),
        "BPR":          ("BPR", {}),
        "LightGCN":     ("LightGCN", {}),
    }


def main():
    cfg = yaml.safe_load(open(os.environ.get("IKGR_CONFIG", "config.yaml"), encoding="utf-8"))
    paths, rb = cfg["paths"], cfg["recbole"]
    all_specs = build_specs(paths)
    seeds = [int(s) for s in os.environ.get("IKGR_SEEDS", "2020,2021,2022").split(",")]
    spec_names = os.environ.get("IKGR_SPECS", ",".join(all_specs)).split(",")
    pop_grid = _float_grid_from_env("IKGR_POP_RERANK_GRID", [])
    pop_specs = set(os.environ.get("IKGR_POP_RERANK_SPECS",
                                   "IKGR_kgoff,BPR,LightGCN,IKGR_dyn,IKGR_dyn_nointent,"
                                   "IKGR_lgcn_dyn_nointent,IKGR_lgcn_dyn").split(","))

    split = os.environ.get("IKGR_SPLIT", "RS").upper()
    wd = paths.get("workdir", "run/")
    os.makedirs(wd, exist_ok=True)
    fname = "slice_eval_result.json" if split == "RS" else f"slice_eval_{split}_result.json"
    out_path = os.path.join(wd, fname)
    pu_dir = os.path.join(wd, f"per_user_{split}")
    os.makedirs(pu_dir, exist_ok=True)
    results = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}

    for name in spec_names:
        model_arg, extra = all_specs[name]
        signature, context = _experiment_context(name, model_arg, extra, rb, paths, split)
        rerank_lambdas = [float(x) for x in extra.get("corona_rerank_grid", [])]
        lambdas = pop_grid if (name in pop_specs and not rerank_lambdas) else []
        run_extra = dict(extra, pop_rerank_grid=lambdas) if lambdas else extra
        variants = _variants(rerank_lambdas, lambdas)
        result_names = {suffix: f"{name}{suffix}" for suffix, _, _ in variants}
        for suffix, kind, lam in variants:
            rname = result_names[suffix]
            sig = signature if kind != "pop" else hashlib.sha256(
                f"{signature}|pop|{lam}".encode()).hexdigest()[:16]
            ctx = context if kind != "pop" else dict(context, pop_rerank_lambda=lam)
            existing = results.get(rname)
            if not existing or existing.get("run_signature") != sig:
                if existing:
                    print(f"[invalidate] {rname}: evaluation inputs changed", flush=True)
                results[rname] = {"run_signature": sig, "run_context": ctx, "seeds": {}}
        for seed in seeds:
            if all(str(seed) in results[r]["seeds"] for r in result_names.values()):
                print(f"[skip] {name} seed={seed}", flush=True); continue
            print(f"\n===== {name} seed={seed} ({split}) =====", flush=True)
            out = _train_and_collect(model_arg, run_extra, rb, paths, seed, split)
            for suffix, rname in result_names.items():
                rep = slice_report(out["test"][suffix], out["item_pop"], out["n_items"])
                rep["train_sec"] = out["train_sec"]
                rep["fit"] = out["fit"]
                rep["leakage_guard"] = out["guard"]
                if out["cand_stats"] and suffix == variants[0][0]:
                    rep.update(out["cand_stats"])
                results[rname]["seeds"][str(seed)] = rep
                results[rname]["agg"] = _aggregate(results[rname]["seeds"])
                per_user_table(out["test"][suffix], out["item_pop"], out["uid_tokens"]).to_csv(
                    os.path.join(pu_dir, f"{rname}__s{seed}__test.csv.gz"), index=False, compression="gzip")
                if out["valid"] is not None:
                    per_user_table(out["valid"][suffix], out["item_pop"], out["uid_tokens"]).to_csv(
                        os.path.join(pu_dir, f"{rname}__s{seed}__valid.csv.gz"), index=False, compression="gzip")
                print(f"[{rname} s{seed}] ndcg@10={rep['overall']['ndcg@10']} "
                      f"tail_recall@10={rep['long_tail']['recall@10']} cov@10={rep['coverage@10']} "
                      f"best_epoch={out['fit']['best_epoch']} ({out['train_sec']}s)", flush=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nALL DONE")


if __name__ == "__main__":
    main()
