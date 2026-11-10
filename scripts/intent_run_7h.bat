@echo off
rem Paid LLM intent augmentation (B/C) for at most 7 hours, NO confirmation prompt.
rem Stops starting new calls after 7 h, saves the cache and pauses; run again to continue.
rem Throughput trend per minute: run_k30_TO\llm_progress_step1.csv
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_intent_augmentation.ps1 -Full -Hours 7 -Yes %*
pause
