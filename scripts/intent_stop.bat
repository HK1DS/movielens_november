@echo off
rem Graceful pause of a running paid LLM stage (step1/step2): no new call starts, in-flight calls finish,
rem the cache is saved and the run exits as "paused". Run the same intent_*.bat again to continue.
cd /d "%~dp0.."
for /d %%D in (run_k*_TO run_k*_TO_GLOBAL) do echo stop> "%%D\STOP"
echo [stop] requested: the running LLM step pauses within about a minute (cache saved).
pause
