@echo off
rem Paid LLM intent augmentation (B/C) for at most 10 hours, NO confirmation prompt. Mostly API calls; the
rem GPU is used only once per split, when step2 builds its RAG inputs (sentence encoder, a few minutes).
rem Pauses with the cache saved; run any intent_*.bat again to continue. Graceful pause earlier: intent_stop.bat
rem Trend per minute (per-key counters in the "keys" column): run_k30_TO\llm_progress_step1.csv
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_intent_augmentation.ps1 -Full -Hours 10 -Yes %*
pause
