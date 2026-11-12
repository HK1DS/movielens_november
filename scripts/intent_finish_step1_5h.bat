@echo off
rem Finish step1 (B) only, for at most 5 hours, NO confirmation prompt; step2 (C) does not start in this run.
rem API calls only (GPU hidden). If step1 is not done after 5 hours, run this file again to continue.
rem Trend per minute: run_k30_TO\llm_progress_step1.csv   Graceful pause earlier: intent_stop.bat
cd /d "%~dp0.."
set "CUDA_VISIBLE_DEVICES=-1"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_intent_augmentation.ps1 -Full -PaidSteps B -Hours 5 -Yes %*
pause
