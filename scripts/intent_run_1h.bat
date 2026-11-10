@echo off
rem Paid LLM intent augmentation (B/C) for at most 1 hour, NO confirmation prompt (e.g. a speed test
rem after changing the key mode). Pauses with the cache saved; run any intent_*.bat again to continue.
rem Per-key counters (parallel keys): "keys" column of run_k30_TO\llm_progress_step1.csv
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_intent_augmentation.ps1 -Full -Hours 1 -Yes %*
pause
