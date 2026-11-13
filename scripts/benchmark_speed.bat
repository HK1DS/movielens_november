@echo off
rem Training-speed benchmark on the real TO data: times a few optimizer steps per spec x batch size x sparse
rem format on the GPU and projects minutes per epoch / hours per 40-epoch run (no model is trained, no API
rem call; ~15-30 min, mostly loading 24M interactions). Builds the metadata KG first if it is missing.
rem Result: run_k30_TO\speed_benchmark.json and speed_benchmark.log
cd /d "%~dp0.."
set "PY=%USERPROFILE%\.venvs\goodreads-ikgr\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
if not exist run_k30_TO\meta_kg_pack.pt "%PY%" build_meta_kg.py --inter data\kc_k30\ikgr-custom\ikgr-custom.inter --out run_k30_TO\meta_kg_pack.pt
"%PY%" -u scripts\benchmark_train_speed.py --split TO %*
pause
