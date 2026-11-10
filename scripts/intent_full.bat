@echo off
rem Full LLM intent augmentation (stages B/C) + KG packs (D). Asks y/N before paying.
rem Ctrl+C to stop; run again to resume from the caches.
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_intent_augmentation.ps1 -Full %*
pause
