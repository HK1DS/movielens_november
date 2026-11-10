@echo off
rem User-profile format changed (2026-11-10: top-rated titles from the whole train history).
rem 1) rebuilds the train-only profiles of TO and TO_GLOBAL (free)
rem 2) resumes the paid LLM run of TO without a prompt until 19:32 today (end of today's 7-hour trend run);
rem    item intents already cached are reused, so only the remaining calls are paid.
rem Trend per minute: run_k30_TO\llm_progress_step1.csv   Graceful pause: scripts\intent_stop.bat
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_intent_augmentation.ps1 -RebuildProfiles -Full -Until 19:32 -Yes %*
pause
