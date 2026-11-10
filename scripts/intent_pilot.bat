@echo off
rem Data prep + small LLM pilot (projects the full run time/cost). Safe to re-run.
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_intent_augmentation.ps1 %*
pause
