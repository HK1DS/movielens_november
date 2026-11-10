@echo off
rem Free data prep only (no LLM calls): ml-25m download, k=30 core, train-only profiles for TO and TO_GLOBAL,
rem size check against the earlier MovieLens repo, LLM cost estimates.
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_intent_augmentation.ps1 -PrepOnly %*
pause
