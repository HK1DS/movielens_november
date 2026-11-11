@echo off
rem Paid LLM intent augmentation (B/C) for at most 9 hours, NO confirmation prompt. API calls only:
rem the GPU is hidden from this run (CUDA_VISIBLE_DEVICES=-1), so step2's sentence encoder would run on CPU.
rem Pauses with the cache saved; run any intent_*.bat again to continue. Graceful pause earlier: intent_stop.bat
rem Trend per minute (per-key counters in the "keys" column): run_k30_TO\llm_progress_step1.csv
cd /d "%~dp0.."
set "CUDA_VISIBLE_DEVICES=-1"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_intent_augmentation.ps1 -Full -Hours 9 -Yes %*
pause
