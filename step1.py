'''
Extract exact intents
'''

import yaml, os
import re, ast, json
import pandas as pd
from tqdm import tqdm
from ikgr_core.llm_client import LocalLLM
from ikgr_core.utils import read_csv, write_csv, ensure_dir

# Number of concurrent LLM requests for intent extraction.
# Conservative default to respect API rate limits; override via env var
# (e.g. set IKGR_STEP1_WORKERS=12 if the provider allows higher throughput).
MAX_WORKERS = int(os.environ.get("IKGR_STEP1_WORKERS", "8"))
# Retry policy for transient API errors (429 rate limits, 5xx). Overridable for tests.
LLM_MAX_RETRIES = int(os.environ.get("IKGR_LLM_MAX_RETRIES", "10"))
LLM_RETRY_DELAY = float(os.environ.get("IKGR_LLM_RETRY_DELAY", "5"))

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

    df = read_csv(in_csv).fillna("")
    # expected columns: user_id,user_profile,item_id,item_profile
    sys_prompt = "You are a helpful assistant that returns ONLY Python lists."

    p_user = load_prompt("prompts/step1_intents.txt")
    p_item = p_user  # same template

    import json, time, requests
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
            json.dump(cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp, cache_path)

    def chat_with_retry(sys_prompt, prompt, max_retries=LLM_MAX_RETRIES, initial_delay=LLM_RETRY_DELAY):
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                return llm.chat(sys_prompt, prompt).strip()
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else "unknown"
                if status == 429:
                    print(f"\n[Rate Limit 429] Sleeping for {delay} seconds before retry (attempt {attempt+1}/{max_retries})...")
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                else:
                    print(f"\n[HTTP Error {status}] Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
            except Exception as e:
                print(f"\n[Error] {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay = min(delay * 2, 60)
        return llm.chat(sys_prompt, prompt).strip()

    # Extract unique profiles
    unique_users = df[df["user_profile"] != ""]["user_profile"].unique()
    unique_items = df[df["item_profile"] != ""]["item_profile"].unique()

    print(f"Unique Users: {len(unique_users)}, Unique Items: {len(unique_items)}")

    # ---- Concurrent intent extraction (thread-safe cache + periodic atomic save) ----
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    cache_lock = threading.Lock()
    failed = {"user": 0, "item": 0}
    save_state = {"since_save": 0}

    def process(kind, prof, template):
        key = str(prof).strip()
        if not key:
            return
        # Skip if another thread (or a previous run) already cached it.
        with cache_lock:
            if key in cache[kind]:
                return
        try:
            ans = chat_with_retry(sys_prompt, template.replace("{PROFILE}", key))
            parsed = str(parse_intent_list(ans))
        except Exception as e:
            # Don't cache on permanent failure -> it will be retried next run.
            print(f"\n[skip {kind}] permanent failure, will retry next run: {e}")
            with cache_lock:
                failed[kind] += 1
            return
        with cache_lock:
            cache[kind][key] = parsed
            save_state["since_save"] += 1
            if save_state["since_save"] >= 50:
                save_cache()
                save_state["since_save"] = 0

    # Build the pending job list (dedup within a kind + skip already-cached).
    jobs, seen = [], {"user": set(), "item": set()}
    for kind, profiles, template in (("user", unique_users, p_user), ("item", unique_items, p_item)):
        for prof in profiles:
            key = str(prof).strip()
            if key and key not in cache[kind] and key not in seen[kind]:
                seen[kind].add(key)
                jobs.append((kind, prof, template))

    print(f"Pending LLM calls: {len(jobs)} | workers={MAX_WORKERS} "
          f"(cached users={len(cache['user'])}, items={len(cache['item'])})")

    if jobs:
        from ikgr_core.jobs import run_interruptible
        run_interruptible(process, jobs, MAX_WORKERS, "Extracting Intents", save_cache)

    save_cache()

    # Fast pandas mapping to populate the dataframe
    print("Mapping intents back to main dataframe...")
    df["user_intents_exact"] = df["user_profile"].map(lambda x: cache["user"].get(str(x).strip(), "[]"))
    df["item_intents_exact"] = df["item_profile"].map(lambda x: cache["item"].get(str(x).strip(), "[]"))
    
    write_csv(df, out_csv)
    print(f"[step1] saved: {out_csv}")
    if failed["user"] or failed["item"]:
        # Do NOT let the pipeline mark this stage complete: failed profiles are not
        # cached, so re-running the same command calls only those.
        print(f"[step1] INCOMPLETE: {failed['user']} user / {failed['item']} item profiles failed "
              f"(API errors). Re-run the same command to retry only these.", flush=True)
        raise SystemExit(3)

if __name__ == "__main__":
    main()
