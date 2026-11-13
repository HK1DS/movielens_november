@echo off
rem TO_GLOBAL paid stages (step1 -> step2 -> KG packs) for at most 7 hours, NO confirmation prompt.
rem The GPU is used once, for a few to ~15 minutes, when TO_GLOBAL step2 first builds its RAG inputs: start
rem the run that reaches step2 in the daytime. Re-run this file to continue; pause: intent_stop.bat
rem Trend per minute: run_k30_TO_GLOBAL\llm_progress_step{1,2}.csv
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_intent_augmentation.ps1 -Split TO_GLOBAL -Full -Hours 7 -Yes %*
pause
