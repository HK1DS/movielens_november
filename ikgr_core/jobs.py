"""Interruptible thread-pool runner for the paid LLM stages (step1 / step2)."""
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


def run_interruptible(fn, jobs, workers, desc, save):
    """Run fn(*job) for every job on a thread pool.

    On Ctrl+C: cancel the queued jobs, wait only for the few calls already in
    flight, persist the cache with save(), and exit with code 130. Re-running
    the same command resumes from the cache, so at most the in-flight calls are
    paid twice.
    """
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [ex.submit(fn, *job) for job in jobs]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=desc):
            pass
    except KeyboardInterrupt:
        print("\n[interrupt] cancelling queued calls, waiting for in-flight calls ...", flush=True)
        ex.shutdown(wait=True, cancel_futures=True)
        save()
        print("[interrupt] cache saved. Re-run the same command to resume.", flush=True)
        raise SystemExit(130)
    ex.shutdown(wait=True)
