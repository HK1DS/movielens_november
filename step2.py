"""
step2 (stage C): expand each profile's exact intents with related intents from the frozen vocabulary.

1. RAG inputs, built ONCE per run directory and reused on every resume (ikgr_core/rag.py):
   vocabulary = sorted unique step1 exact intents, their encoder embeddings, and for every user / item
   profile its knn_k most similar intents (exact cosine). The encoder runs on the GPU when available
   (IKGR_ENCODER_DEVICE). A resumed run loads these files and never re-encodes.
2. One LLM call per profile: pick the related intents among its candidates (minus its exact intents).
   Calls are cached per profile text, resumable, and paced per API key like step1.

  python step2.py                 # build/reuse RAG inputs, then the LLM calls
  python step2.py --prepare_only  # only build/check the RAG inputs (no API call)
If step1's output changes after step2 has cached results, step2 stops (exit 2) instead of mixing
candidates from two vocabularies; IKGR_STEP2_REBUILD=1 rebuilds the RAG inputs anyway.
"""

import os
import yaml
from ikgr_core.utils import read_csv, write_csv, ensure_dir
from ikgr_core.rag import NOT_BUILT, build_frozen, device_label, fingerprint, load_frozen, pick_device
from ikgr_core.llm_client import AllKeysExhausted, LocalLLM
from ikgr_core.key_lanes import KeyLanes, chat_with_retry as lanes_chat
from ikgr_core.llm_run import EXIT_PAUSED, Progress, keep_awake, time_up
import re, ast, json, time

# Worker threads: by default 8 per API key, all keys in parallel (ikgr_core/key_lanes.py);
# IKGR_STEP2_WORKERS fixes the total instead.
WORKERS_ENV = os.environ.get("IKGR_STEP2_WORKERS")
# Retry policy for transient API errors (429 rate limits, 5xx). Overridable for tests.
LLM_MAX_RETRIES = int(os.environ.get("IKGR_LLM_MAX_RETRIES", "10"))
LLM_RETRY_DELAY = float(os.environ.get("IKGR_LLM_RETRY_DELAY", "5"))
# Full-cache rewrite interval (the cache reaches 100+ MB at ~150k profiles; every rewrite holds
# the worker lock). Ctrl+C always saves immediately; a hard kill loses at most this many calls.
CACHE_SAVE_EVERY = int(os.environ.get("IKGR_CACHE_SAVE_EVERY", "500"))

def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _safe_eval_list(s):
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not s:
        return []
    
    # Strip markdown block quotes
    s = re.sub(r"^```(?:python|json)?", "", s, flags=re.MULTILINE)
    s = re.sub(r"```$", "", s, flags=re.MULTILINE)
    s = s.strip()
    
    # Extract everything inside [ and ]
    match = re.search(r"\[.*\]", s, re.DOTALL)
    if match:
        s_content = match.group(0)
    else:
        s_content = s
        
    try:
        val = ast.literal_eval(s_content)
        if isinstance(val, list):
            return val
    except:
        pass
        
    try:
        val = json.loads(s_content)
        if isinstance(val, list):
            return val
    except:
        pass
        
    # Regex fallback for integers or single/double quoted items
    try:
        # Check if they are indices (integers)
        nums = [int(x) for x in re.findall(r"\d+", s_content)]
        if nums:
            return nums
    except:
        pass
        
    return []

def main():
    import argparse
    ap = argparse.ArgumentParser(description="step2: RAG related-intent expansion")
    ap.add_argument("--prepare_only", action="store_true", help="build/check the frozen RAG inputs, no API call")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.environ.get("IKGR_CONFIG", "config.yaml"), encoding="utf-8"))
    paths = cfg["paths"]
    rag_cfg = cfg["rag"]
    llm_cfg = cfg["llm"]

    ensure_dir(paths["workdir"])
    p_rel = load_prompt("prompts/step2_related.txt")
    df = read_csv(paths["step1_output"]).fillna("")

    def _parse_exact_list(s):
        val = _safe_eval_list(s)
        return [str(x) for x in val if isinstance(x, str)]

    # Unique profiles (first occurrence wins) with their exact intents, items before users.
    exact_maps = {}
    for kind in ("item", "user"):
        sub = df[[f"{kind}_profile", f"{kind}_intents_exact"]].copy()
        sub[f"{kind}_profile"] = sub[f"{kind}_profile"].astype(str).str.strip()
        sub = sub[sub[f"{kind}_profile"] != ""].drop_duplicates(f"{kind}_profile")
        exact_maps[kind] = {p: _parse_exact_list(v) for p, v in zip(sub[f"{kind}_profile"], sub[f"{kind}_intents_exact"])}
    vocab = sorted({x for m in exact_maps.values() for lst in m.values() for x in lst})
    profiles = {kind: sorted(m) for kind, m in exact_maps.items()}
    print(f"[step2] profiles: {len(profiles['user']):,} users, {len(profiles['item']):,} items | "
          f"intent vocabulary: {len(vocab):,}", flush=True)

    cache_path = os.path.join(paths["workdir"], "step2_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Loaded cache from {cache_path} (users={len(cache.get('user', {}))}, items={len(cache.get('item', {}))})")
    else:
        cache = {"user": {}, "item": {}}

    # 1) Frozen RAG inputs: reuse when they match step1's output, else build once (GPU if available)
    rag_paths = {"vocab_json": rag_cfg["vocab_json"], "encoding_npy": rag_cfg["encoding_npy"],
                 "candidates": os.path.join(paths["workdir"], "rag_candidates.npz"),
                 "manifest": os.path.join(paths["workdir"], "rag_manifest.json")}
    want = fingerprint(rag_cfg["encoder"], rag_cfg["knn_k"], vocab, profiles)
    cands, reason = load_frozen(rag_paths, want)
    if cands is not None:
        print(f"[step2] reusing frozen RAG inputs ({rag_paths['manifest']}): no re-encoding", flush=True)
    else:
        if reason != NOT_BUILT and (cache["user"] or cache["item"]) and os.environ.get("IKGR_STEP2_REBUILD") != "1":
            print(f"[step2] STOP: {reason}.\n"
                  f"  {len(cache['user']) + len(cache['item']):,} step2 results in {cache_path} were made with the old\n"
                  f"  candidates; rebuilding would mix two vocabularies. If step1 really changed, delete step2_cache.json\n"
                  f"  (start step2 over) or set IKGR_STEP2_REBUILD=1 to rebuild and keep the old results.", flush=True)
            raise SystemExit(2)
        if time_up():
            print("[step2] PAUSED: time budget reached or STOP file found before the RAG inputs were built.", flush=True)
            raise SystemExit(EXIT_PAUSED)
        device = pick_device()
        print(f"[step2] building RAG inputs ({reason}) | encoder {rag_cfg['encoder']} on {device_label(device)}",
              flush=True)
        if device == "cpu":
            print("[step2] WARNING: no GPU visible -> encoding on the CPU (much slower; done once per split)", flush=True)
        cands = build_frozen(rag_paths, want, rag_cfg["encoder"], vocab, profiles, device,
                             log=lambda m: print(m, flush=True))
    if args.prepare_only:
        print("[step2] --prepare_only: RAG inputs ready, no API call made.", flush=True)
        return
    row_of = {kind: {p: i for i, p in enumerate(profiles[kind])} for kind in profiles}

    llm = LocalLLM(**llm_cfg)
    print(f"[step2] LLM keys: {len(llm.keys)} configured, {len(llm.keys.usable())} usable", flush=True)
    if keep_awake():
        print("[step2] Windows sleep blocked while this step runs", flush=True)
    sys_prompt = "You are a helpful assistant that returns ONLY JSON lists of integers."

    def save_cache():
        # Atomic write to avoid corrupting the resumable cache on interrupt.
        tmp = cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        os.replace(tmp, cache_path)

    lanes = KeyLanes(llm.keys, total_workers=int(WORKERS_ENV) if WORKERS_ENV else None)
    MAX_WORKERS = lanes.workers
    print(f"[step2] API keys: {lanes.describe()}", flush=True)
    progress = Progress()

    def chat_with_retry(sys_prompt, prompt, max_retries=LLM_MAX_RETRIES, initial_delay=LLM_RETRY_DELAY):
        # Per-key pacing (each key its own 429 cooldown / interval) and retries: ikgr_core/key_lanes.py
        return lanes_chat(llm, lanes, progress, sys_prompt, prompt, max_retries, initial_delay)

    # ---- Pending jobs: every unique profile not cached yet, items first ----
    jobs = [(kind, key) for kind in ("item", "user") for key in exact_maps[kind] if key not in cache[kind]]
    print(f"Pending RAG+LLM calls: {len(jobs)} | workers={MAX_WORKERS} "
          f"(cached users={len(cache['user'])}, items={len(cache['item'])})")

    # ---- Concurrent expansion (thread-safe cache + periodic atomic save) ----
    import threading

    cache_lock = threading.Lock()
    failed = {"user": 0, "item": 0}
    save_state = {"since_save": 0, "done": 0}
    keys_exhausted = threading.Event()
    deferred = {"user": 0, "item": 0}           # not started because the time budget ran out

    def process(kind, key):
        if keys_exhausted.is_set():            # all API keys gone: skip, re-run resumes these
            with cache_lock:
                failed[kind] += 1
            return
        if time_up():                          # time budget over: leave it for the next run
            with cache_lock:
                deferred[kind] += 1
            return
        exact = set(exact_maps[kind].get(key, []))
        options_filtered = [vocab[i] for i in cands[kind][row_of[kind][key]] if vocab[i] not in exact]
        if not options_filtered:               # nothing left to choose from: no call needed
            with cache_lock:
                cache[kind][key] = []
                progress.success(kind)
            return
        options_text = "\n".join([f"{idx}: {opt}" for idx, opt in enumerate(options_filtered)])
        prompt = p_rel.replace("{PROFILE}", key).replace("{OPTIONS}", options_text)
        try:
            ans = chat_with_retry(sys_prompt, prompt)
        except AllKeysExhausted as e:
            if not keys_exhausted.is_set():
                keys_exhausted.set()
                print(f"\n[step2] {e}", flush=True)
            with cache_lock:
                failed[kind] += 1
            return
        except Exception as e:
            print(f"\n[skip {kind}] permanent failure, will retry next run: {e}")
            with cache_lock:
                failed[kind] += 1
            return
        selected_indices = _safe_eval_list(ans)
        rel = []
        for idx in selected_indices:
            try:
                idx_int = int(idx)
                if 0 <= idx_int < len(options_filtered):
                    rel.append(options_filtered[idx_int])
            except Exception:
                pass
        with cache_lock:
            cache[kind][key] = rel
            save_state["since_save"] += 1
            save_state["done"] += 1
            progress.success(kind)
            if save_state["since_save"] >= CACHE_SAVE_EVERY:
                save_cache()
                save_state["since_save"] = 0

    if jobs:
        from ikgr_core.jobs import run_interruptible
        _t0 = time.time()
        progress.start(os.path.join(paths["workdir"], "llm_progress_step2.csv"), len(jobs), lanes, lanes)
        try:
            run_interruptible(process, jobs, MAX_WORKERS, "Expanding Intents", save_cache)
        finally:
            progress.stop()
        _dt = time.time() - _t0
        print(f"[step2] throughput: calls={save_state['done']} seconds={_dt:.1f} "
              f"rate={save_state['done'] / max(_dt, 1e-9):.2f}/s | keys: {lanes.status()}", flush=True)
        with open(os.path.join(paths["workdir"], "step2_throughput.json"), "w", encoding="utf-8") as f:
            json.dump({"calls": save_state["done"], "seconds": round(_dt, 2), "workers": MAX_WORKERS,
                       "paced_interval_s": lanes.min_interval_seconds,
                       "key_mode": lanes.mode, "per_key": lanes.per_key()}, f)

    save_cache()

    # Fast pandas mapping to populate the dataframe
    print("Mapping related intents back to main dataframe...")
    df["user_intents_related"] = df["user_profile"].map(lambda x: str(cache["user"].get(str(x).strip(), [])))
    df["item_intents_related"] = df["item_profile"].map(lambda x: str(cache["item"].get(str(x).strip(), [])))
    
    write_csv(df, paths["step2_output"])
    print(f"[step2] saved: {paths['step2_output']}")
    if keys_exhausted.is_set():
        print(f"[step2] STOPPED: all LLM API keys exhausted; {failed['user']} user / {failed['item']} item "
              f"profiles not done. Add/recharge keys in .env and re-run the same command.", flush=True)
        raise SystemExit(4)
    if deferred["user"] or deferred["item"]:
        print(f"[step2] PAUSED: time budget reached or STOP file found; {deferred['item']} item / {deferred['user']} user profiles "
              f"left. Re-run the same command to continue (finished calls are cached).", flush=True)
        raise SystemExit(EXIT_PAUSED)
    if failed["user"] or failed["item"]:
        # Do NOT let the pipeline mark this stage complete: failed profiles are not
        # cached, so re-running the same command calls only those.
        print(f"[step2] INCOMPLETE: {failed['user']} user / {failed['item']} item profiles failed "
              f"(API errors). Re-run the same command to retry only these.", flush=True)
        raise SystemExit(3)

if __name__ == "__main__":
    main()
