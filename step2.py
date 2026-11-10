'''
RAG expand related intents (from fixed intent vocab)
'''

import os
import yaml, numpy as np
import pandas as pd
from tqdm import tqdm
from ikgr_core.utils import load_json, save_json, read_csv, write_csv, ensure_dir
from ikgr_core.rag import IntentEncoderIndex, knn_strings
from ikgr_core.llm_client import AllKeysExhausted, LocalLLM
from ikgr_core.rate_limit import SharedRateLimiter, backoff_with_jitter, retry_after_seconds
from ikgr_core.llm_run import EXIT_PAUSED, Progress, deadline, keep_awake, time_up
import re, ast, json, time
import requests

# Number of concurrent LLM requests for RAG intent expansion.
# Override via env var IKGR_STEP2_WORKERS if the provider allows more throughput.
MAX_WORKERS = int(os.environ.get("IKGR_STEP2_WORKERS", "8"))
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
    cfg = yaml.safe_load(open(os.environ.get("IKGR_CONFIG", "config.yaml"), encoding="utf-8"))
    paths = cfg["paths"]
    rag_cfg = cfg["rag"]
    llm_cfg = cfg["llm"]

    ensure_dir(paths["workdir"])
    p_rel = load_prompt("prompts/step2_related.txt")
    df = read_csv(paths["step1_output"]).fillna("")

    # 1) Freeze vocabulary from step1 exact intents
    def _parse_exact_list(s):
        val = _safe_eval_list(s)
        return [str(x) for x in val if isinstance(x, str)]

    vocab = set()
    for col in ["user_intents_exact", "item_intents_exact"]:
        for v in df[col].dropna():
            vocab.update(_parse_exact_list(v))
    vocab = sorted(list(vocab))
    save_json(vocab, rag_cfg["vocab_json"])

    # 2) Encode vocab + build Index
    enc = IntentEncoderIndex(rag_cfg["encoder"])
    emb = enc.encode(vocab)
    np.save(rag_cfg["encoding_npy"], emb)
    enc.build_ann(emb, emb.shape[1], rag_cfg["annoy_trees"], rag_cfg["annoy_index"])

    ann = enc.load_ann(emb.shape[1], rag_cfg["annoy_index"])
    llm = LocalLLM(**llm_cfg)
    print(f"[step2] LLM keys: {llm.keys.status()}", flush=True)
    if keep_awake():
        print("[step2] Windows sleep blocked while this step runs", flush=True)
    sys_prompt = "You are a helpful assistant that returns ONLY JSON lists of integers."

    # 3) For each row, expand related intents for user & item via RAG + LLM selection
    cache_path = os.path.join(paths["workdir"], "step2_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Loaded cache from {cache_path} (users={len(cache.get('user', {}))}, items={len(cache.get('item', {}))})")
    else:
        cache = {"user": {}, "item": {}}

    def save_cache():
        # Atomic write to avoid corrupting the resumable cache on interrupt.
        tmp = cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        os.replace(tmp, cache_path)

    rate_limiter = SharedRateLimiter()
    progress = Progress()

    def chat_with_retry(sys_prompt, prompt, max_retries=LLM_MAX_RETRIES, initial_delay=LLM_RETRY_DELAY):
        # Shared pacing across workers (ikgr_core/rate_limit.py): honours Retry-After, adds jitter,
        # and switches every worker to the stable interval after the first 429.
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                rate_limiter.acquire()
                out = llm.chat(sys_prompt, prompt).strip()
                rate_limiter.note_success()        # lets the pacing relax again after a 429-free streak
                return out
            except AllKeysExhausted:
                raise                      # no point retrying: every API key is out of credit
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else "unknown"
                progress.error("429" if status == 429 else "http")
                if status == 429:
                    wait = max(delay, retry_after_seconds(e.response) or 0.0)
                    switched = rate_limiter.defer_all(wait)
                    sleep_for = backoff_with_jitter(wait)
                    print(f"\n[Rate Limit 429] shared cooldown {wait:.1f}s; retrying in {sleep_for:.1f}s "
                          f"(attempt {attempt+1}/{max_retries})...")
                    if switched:
                        print(f"[Rate Limit 429] all workers now paced at one request start per "
                              f"{rate_limiter.min_interval_seconds:.2f}s")
                    time.sleep(sleep_for)
                else:
                    sleep_for = backoff_with_jitter(delay)
                    print(f"\n[HTTP Error {status}] Retrying in {sleep_for:.1f} seconds...")
                    time.sleep(sleep_for)
                delay = min(delay * 2, 60)
            except Exception as e:
                progress.error("other")
                sleep_for = backoff_with_jitter(delay)
                print(f"\n[Error] {e}. Retrying in {sleep_for:.1f} seconds...")
                time.sleep(sleep_for)
                delay = min(delay * 2, 60)
        rate_limiter.acquire()
        return llm.chat(sys_prompt, prompt).strip()

    unique_users = df[df["user_profile"] != ""]["user_profile"].unique()
    unique_items = df[df["item_profile"] != ""]["item_profile"].unique()

    print(f"Unique Users for RAG: {len(unique_users)}, Unique Items for RAG: {len(unique_items)}")

    # Create mapping table for exact intents by profile to construct options_filtered
    user_exact_map = {}
    item_exact_map = {}
    for _, r in df.iterrows():
        u_prof, i_prof = str(r.get("user_profile", "")).strip(), str(r.get("item_profile", "")).strip()
        if u_prof and u_prof not in user_exact_map:
            user_exact_map[u_prof] = _parse_exact_list(r.get("user_intents_exact", "[]"))
        if i_prof and i_prof not in item_exact_map:
            item_exact_map[i_prof] = _parse_exact_list(r.get("item_intents_exact", "[]"))

    # ---- Build pending job list (skip cached + dedup within a kind) ----
    exact_maps = {"user": user_exact_map, "item": item_exact_map}
    jobs = []  # (kind, profile_key)
    seen = {"user": set(), "item": set()}
    for kind, profiles in (("item", unique_items), ("user", unique_users)):        # items first
        for prof in profiles:
            key = str(prof).strip()
            if key and key not in cache[kind] and key not in seen[kind]:
                seen[kind].add(key)
                jobs.append((kind, key))

    print(f"Pending RAG+LLM calls: {len(jobs)} | workers={MAX_WORKERS} "
          f"(cached users={len(cache['user'])}, items={len(cache['item'])})")

    # ---- Pre-encode all pending profiles in ONE batch ----
    # SentenceTransformer is NOT called concurrently (PyTorch forward is not
    # guaranteed thread-safe); we batch-encode here, then only do ANN reads +
    # LLM calls inside threads.
    emb_map = {}
    if jobs:
        all_profs = [key for _, key in jobs]
        prof_embs = enc.encode(all_profs)
        for (kind, key), e in zip(jobs, prof_embs):
            emb_map[(kind, key)] = e

    # ---- Concurrent expansion (thread-safe cache + periodic atomic save) ----
    from concurrent.futures import ThreadPoolExecutor, as_completed
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
        exact = exact_maps[kind].get(key, [])
        q_emb = emb_map[(kind, key)]
        options = knn_strings(ann, q_emb, vocab, cfg["rag"]["knn_k"])  # ANN read is thread-safe
        options_filtered = [o for o in options if o not in exact]
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
        progress.start(os.path.join(paths["workdir"], "llm_progress_step2.csv"), len(jobs), rate_limiter, llm.keys)
        try:
            run_interruptible(process, jobs, MAX_WORKERS, "Expanding Intents", save_cache)
        finally:
            progress.stop()
        _dt = time.time() - _t0
        print(f"[step2] throughput: calls={save_state['done']} seconds={_dt:.1f} "
              f"rate={save_state['done'] / max(_dt, 1e-9):.2f}/s | keys: {llm.keys.status()}", flush=True)
        with open(os.path.join(paths["workdir"], "step2_throughput.json"), "w", encoding="utf-8") as f:
            json.dump({"calls": save_state["done"], "seconds": round(_dt, 2), "workers": MAX_WORKERS,
                       "paced_interval_s": rate_limiter.min_interval_seconds}, f)

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
