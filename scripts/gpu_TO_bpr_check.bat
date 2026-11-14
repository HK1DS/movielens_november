@echo off
rem GPU: calibration run BPR seed 2020 on TO (~15 min). Compare its test NDCG@10 with the teammate's BPR seed 2020
rem TO (run_k30_TO\slice_eval_TO_result.json). No API call.
cd /d "%~dp0.."
set "PY=%USERPROFILE%\.venvs\goodreads-ikgr\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
rem the metadata KG must have the team-wide content (deterministic build): fail fast otherwise
"%PY%" scripts\check_meta_kg.py || (echo [stop] metadata KG differs: run  "%PY%" scripts\check_meta_kg.py --rebuild & pause & exit /b 1)
"%PY%" -u run_pipeline.py --k 30 --split TO --steps E --seeds 2020 --specs BPR %*
pause
