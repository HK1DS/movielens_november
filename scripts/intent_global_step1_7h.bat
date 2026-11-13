@echo off
rem TO_GLOBAL step1 (B) only, for at most 7 hours, NO confirmation prompt; step2 does not start in this run
rem (its one-time GPU encoding should run in the daytime: start TO_GLOBAL step2 with intent_global_run_7h.bat).
rem Item intents are reused from run_k30_TO automatically. Re-run this file to continue; pause: intent_stop.bat
rem Trend per minute: run_k30_TO_GLOBAL\llm_progress_step1.csv
cd /d "%~dp0.."
set "CUDA_VISIBLE_DEVICES=-1"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_intent_augmentation.ps1 -Split TO_GLOBAL -Full -PaidSteps B -Hours 7 -Yes %*
pause
