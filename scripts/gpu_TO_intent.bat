@echo off
rem GPU: TO intent models IKGR_dyn + IKGR_lgcn_dyn x seeds 2020,2021,2022 (~27 h in total). No API call, so the
rem TO_GLOBAL API run can go on at the same time. Stop with Ctrl+C right after a new
rem '===== <spec> seed=<n> (TO) =====' header (nothing lost); double-click again to continue.
cd /d "%~dp0.."
set "PY=%USERPROFILE%\.venvs\goodreads-ikgr\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
rem the metadata KG must have the team-wide content (deterministic build): fail fast otherwise
"%PY%" scripts\check_meta_kg.py || (echo [stop] metadata KG differs: run  "%PY%" scripts\check_meta_kg.py --rebuild & pause & exit /b 1)
"%PY%" -u run_pipeline.py --k 30 --split TO --steps E --seeds 2020,2021,2022 --specs IKGR_dyn,IKGR_lgcn_dyn %*
pause
