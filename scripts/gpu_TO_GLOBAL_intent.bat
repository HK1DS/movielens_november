@echo off
rem GPU: TO_GLOBAL intent models IKGR_dyn + IKGR_lgcn_dyn x seeds 2020,2021,2022 (~22 h). Needs TO_GLOBAL step2 + D
rem (run_k30_TO_GLOBAL\kg_pack.pt). Stop/continue as in gpu_TO_intent.bat.
cd /d "%~dp0.."
set "PY=%USERPROFILE%\.venvs\goodreads-ikgr\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
rem the metadata KG must have the team-wide content (deterministic build): fail fast otherwise
"%PY%" scripts\check_meta_kg.py || (echo [stop] metadata KG differs: run  "%PY%" scripts\check_meta_kg.py --rebuild & pause & exit /b 1)
if not exist run_k30_TO_GLOBAL\kg_pack.pt (echo [stop] run_k30_TO_GLOBAL\kg_pack.pt missing: finish TO_GLOBAL step2 + D first & pause & exit /b 1)
"%PY%" -u run_pipeline.py --k 30 --split TO_GLOBAL --steps E --seeds 2020,2021,2022 --specs IKGR_dyn,IKGR_lgcn_dyn %*
pause
