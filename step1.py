'''
Extract exact intents
'''

import yaml, os
import re, ast, json, time
import pandas as pd
from tqdm import tqdm
from ikgr_core.llm_client import AllKeysExhausted, LocalLLM
from ikgr_core.key_lanes import KeyLanes, chat_with_retry as lanes_chat
from ikgr_core.llm_run import EXIT_PAUSED, Progress, deadline, keep_awake, time_up
from ikgr_core.utils import read_csv, write_csv, ensure_dir

# Worker threads: by default 8 per API key, all keys in parallel (ikgr_core/key_lanes.py);
# IKGR_STEP1_WORKERS fixes the total instead.
WORKERS_ENV = os.environ.get("IKGR_STEP1_WORKERS")
# Retry policy for transient API errors (429 rate limits, 5xx). Overridable for tests.
LLM_MAX_RETRIES = int(os.environ.get("IKGR_LLM_MAX_RETRIES", "10"))
LLM_RETRY_DELAY = float(os.environ.get("IKGR_LLM_RETRY_DELAY", "5"))
# Full-cache rewrite interval (the cache reaches 100+ MB at ~150k profiles; every rewrite holds
# the worker lock). Ctrl+C always saves immediately; a hard kill loses at most this many calls.
CACHE_SAVE_EVERY = int(os.environ.get("IKGR_CACHE_SAVE_EVERY", "500"))

def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def parse_intent_list(s):
    """
    Robustly parse an LLM response into a list of intent strings.
    Handles markdown code fences, surrounding prose, and JSON/Python list syntax.
    Returns a list[str] (possibly empty).
    """
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not s:
        return []
    # Strip markdown code fences (```python / ```json / ```)
    s = re.sub(r"^```(?:python|json)?", "", s, flags=re.MULTILINE)
    s = re.sub(r"```$", "", s, flags=re.MULTILINE)
    s = s.strip()
    # Extract the first [...] block if surrounded by prose
    m = re.search(r"\[.*\]", s, re.DOTALL)
    content = m.group(0) if m else s
    for parser in (ast.literal_eval, json.loads):
        try:
            val = parser(content)
            if isinstance(val, list):
                return [str(x).strip() for x in val if str(x).strip()]
        except Exception:
            pass
    return []

def main():
    cfg = yaml.safe_load(open(os.environ.get("IKGR_CONFIG", "config.yaml"), encoding="utf-8"))
    in_csv = cfg["paths"]["input_csv"]
    out_csv = cfg["paths"]["step1_output"]
    work = cfg["paths"]["workdir"]
    ensure_dir(work)

    llm_cfg = cfg["llm"]
    llm = LocalLLM(**llm_cfg)
    print(f"[step1] LLM keys: {len(llm.keys)} configured, {len(llm.keys.usable())} usable", flush=True)
    if keep_awake():
        print("[step1] Windows sleep blocked while this step runs", flush=True)
    if deadline():
        print(f"[step1] time budget: stops starting new calls at {time.strftime('%Y-%m-%d %H:%M', time.localtime(deadline()))}",
              flush=True)

    df = read_csv(in_csv).fillna("")
    # expected columns: user_id,user_profile,item_id,item_profile
    sys_prompt = "You are a helpful assistant that returns ONLY Python lists."

    p_user = load_prompt("prompts/step1_intents.txt")
    p_item = p_user  # same template

    cache_path = os.path.join(work, "step1_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Loaded cache from {cache_path} (users={len(cache.get('user', {}))}, items={len(cache.get('item', {}))})")
    else:
        cache = {"user": {}, "item": {}}

    def save_cache():
        # Atomic write: serialize to a temp file then replace, so an interrupt
        # mid-write cannot corrupt the resumable cache.
        tmp = cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        os.replace(tmp, cache_path)

    lanes = KeyLanes(llm.keys, total_workers=int(WORKERS_ENV) if WORKERS_ENV else None)
    MAX_WORKERS = lanes.workers
    print(f"[step1] API keys: {lanes.describe()}", flush=True)
    progress = Progress()

    def chat_with_retry(sys_prompt, prompt, max_retries=LLM_MAX_RETRIES, initial_delay=LLM_RETRY_DELAY):
        # Per-key pacing (each key its own 429 cooldown / interval) and retries: ikgr_core/key_lanes.py
        return lanes_chat(llm, lanes, progress, sys_prompt, prompt, max_retries, initial_delay)

    # Extract unique profiles
    unique_users = df[df["user_profile"] != ""]["user_profile"].unique()
    unique_items = df[df["item_profile"] != ""]["item_profile"].unique()

    print(f"Unique Users: {len(unique_users)}, Unique Items: {len(unique_items)}")

    # ---- Concurrent intent extraction (thread-safe cache + periodic atomic save) ----
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    cache_lock = threading.Lock()
    failed = {"user": 0, "item": 0}
    save_state = {"since_save": 0, "done": 0}
    keys_exhausted = threading.Event()
    deferred = {"user": 0, "item": 0}           # not started because the time budget ran out

    def process(kind, prof, template):
        key = str(prof).strip()
        if not key:
            return
        # Skip if another thread (or a previous run) already cached it.
        with cache_lock:
            if key in cache[kind]:
                return
        if keys_exhausted.is_set():            # all API keys gone: skip, re-run resumes these
            with cache_lock:
                failed[kind] += 1
            return
        if time_up():                          # time budget over: leave it for the next run
            with cache_lock:
                deferred[kind] += 1
            return
        try:
            ans = chat_with_retry(sys_prompt, template.replace("{PROFILE}", key))
            parsed = str(parse_intent_list(ans))
        except AllKeysExhausted as e:
            if not keys_exhausted.is_set():
                keys_exhausted.set()
                print(f"\n[step1] {e}", flush=True)
            with cache_lock:
                failed[kind] += 1
            return
        except Exception as e:
            # Don't cache on permanent failure -> it will be retried next run.
            print(f"\n[skip {kind}] permanent failure, will retry next run: {e}")
            with cache_lock:
                failed[kind] += 1
            return
        with cache_lock:
            cache[kind][key] = parsed
            save_state["since_save"] += 1
            save_state["done"] += 1
            progress.success(kind)
            if save_state["since_save"] >= CACHE_SAVE_EVERY:
                save_cache()
                save_state["since_save"] = 0

    # Build the pending job list (dedup within a kind + skip already-cached). Items first: their
    # intents depend only on the static catalog text, so they stay valid across splits and
    # user-profile formats even if a run is paused part-way.
    jobs, seen = [], {"user": set(), "item": set()}
    for kind, profiles, template in (("item", unique_items, p_item), ("user", unique_users, p_user)):
        for prof in profiles:
            key = str(prof).strip()
            if key and key not in cache[kind] and key not in seen[kind]:
                seen[kind].add(key)
                jobs.append((kind, prof, template))

    print(f"Pending LLM calls: {len(jobs)} | workers={MAX_WORKERS} "
          f"(cached users={len(cache['user'])}, items={len(cache['item'])})")

    if jobs:
        from ikgr_core.jobs import run_interruptible
        _t0 = time.time()
        progress.start(os.path.join(work, "llm_progress_step1.csv"), len(jobs), lanes, lanes)
        try:
            run_interruptible(process, jobs, MAX_WORKERS, "Extracting Intents", save_cache)
        finally:
            progress.stop()
        _dt = time.time() - _t0
        print(f"[step1] throughput: calls={save_state['done']} seconds={_dt:.1f} "
              f"rate={save_state['done'] / max(_dt, 1e-9):.2f}/s | keys: {lanes.status()}", flush=True)
        with open(os.path.join(work, "step1_throughput.json"), "w", encoding="utf-8") as f:
            json.dump({"calls": save_state["done"], "seconds": round(_dt, 2), "workers": MAX_WORKERS,
                       "paced_interval_s": lanes.min_interval_seconds,
                       "key_mode": lanes.mode, "per_key": lanes.per_key()}, f)

    save_cache()

    # Fast pandas mapping to populate the dataframe
    print("Mapping intents back to main dataframe...")
    df["user_intents_exact"] = df["user_profile"].map(lambda x: cache["user"].get(str(x).strip(), "[]"))
    df["item_intents_exact"] = df["item_profile"].map(lambda x: cache["item"].get(str(x).strip(), "[]"))
    
    write_csv(df, out_csv)
    print(f"[step1] saved: {out_csv}")
    if keys_exhausted.is_set():
        print(f"[step1] STOPPED: all LLM API keys exhausted; {failed['user']} user / {failed['item']} item "
              f"profiles not done. Add/recharge keys in .env and re-run the same command.", flush=True)
        raise SystemExit(4)
    if deferred["user"] or deferred["item"]:
        print(f"[step1] PAUSED: time budget reached or STOP file found; {deferred['item']} item / {deferred['user']} user profiles "
              f"left. Re-run the same command to continue (finished calls are cached).", flush=True)
        raise SystemExit(EXIT_PAUSED)
    if failed["user"] or failed["item"]:
        # Do NOT let the pipeline mark this stage complete: failed profiles are not
        # cached, so re-running the same command calls only those.
        print(f"[step1] INCOMPLETE: {failed['user']} user / {failed['item']} item profiles failed "
              f"(API errors). Re-run the same command to retry only these.", flush=True)
        raise SystemExit(3)

if __name__ == "__main__":
    main()
