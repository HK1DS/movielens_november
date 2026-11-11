@echo off
rem Paid LLM intent augmentation (B/C) for at most 6 hours, NO confirmation prompt.
rem Pauses with the cache saved; run any intent_*.bat again to continue. Graceful pause earlier: intent_stop.bat
rem Trend per minute (per-key counters in the "keys" column): run_k30_TO\llm_progress_step1.csv
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_intent_augmentation.ps1 -Full -Hours 6 -Yes %*
pause
