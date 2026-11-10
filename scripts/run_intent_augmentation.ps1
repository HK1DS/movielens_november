# MovieLens 25M -> LLM intent augmentation (stages B/C) on Windows.
#
#   scripts\intent_prep.bat    (= this script -PrepOnly)  free data prep only, both splits (TO + TO_GLOBAL),
#                                                       same scale as HK1DS/hk1ds_jwkim_movielens (k=30)
#   scripts\intent_pilot.bat   (= this script)            data prep + small LLM pilot (time/cost projection)
#   scripts\intent_full.bat    (= this script -Full)      data prep (skipped if done) + full B/C + KG packs (D)
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_intent_augmentation.ps1 [-Full] [-K 30] [-Split TO]
#
# Steps: Python env check -> .env check -> download/extract ml-25m (if missing)
#        -> run_pipeline.py --steps 0ATPS (free: interactions, k-core, RecBole .inter, train-only
#           profiles, LLM cost estimate) -> pilot_llm.py  OR  run_pipeline.py --steps BCD (asks y/N first).
# Stop any time with Ctrl+C and run the same .bat again: finished work and LLM caches are reused.
# Exit code 4 from the LLM stage = every API key in .env is out of credit: add/recharge keys, re-run.
param(
    [int]$K = 30,
    [string]$Split = "TO",
    [switch]$Full,
    [switch]$PrepOnly,
    [string]$Python = "",
    [int]$PilotUsers = 300,
    [int]$PilotItems = 100
)

$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function Fail([string]$msg) { Write-Host "[error] $msg" -ForegroundColor Red; exit 1 }

# 1) Python environment (same packages as the Goodreads repo -> reuse its venv)
if (-not $Python) {
    $Python = Join-Path $env:USERPROFILE ".venvs\goodreads-ikgr\Scripts\python.exe"
    if (-not (Test-Path $Python)) {
        Write-Host "[env] $Python not found -> creating it (scripts\setup_venv.ps1, ~10-20 min)"
        & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_venv.ps1
        if ($LASTEXITCODE -ne 0) { Fail "venv setup failed" }
    }
}
& $Python -c "import pandas, yaml, requests, sentence_transformers, sklearn; print('[env] python ok')"
if ($LASTEXITCODE -ne 0) { Fail "the Python environment is missing packages: run scripts\setup_venv.ps1 (or pass -Python <python.exe>)" }

# 2) API keys
if (-not (Test-Path ".env")) { Fail ".env not found: copy .env.example to .env and fill LUXIA_API_KEY (and _2, _3, ...)" }

# 3) MovieLens 25M raw files
$raw = "raw\ml-25m"
if (-not ((Test-Path "$raw\ratings.csv") -and (Test-Path "$raw\movies.csv"))) {
    New-Item -ItemType Directory -Force -Path "raw" | Out-Null
    $zip = "raw\ml-25m.zip"
    if (-not (Test-Path $zip)) {
        Write-Host "[data] downloading ml-25m.zip (~250 MB) from files.grouplens.org ..."
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $ProgressPreference = "SilentlyContinue"
        try {
            Invoke-WebRequest -Uri "https://files.grouplens.org/datasets/movielens/ml-25m.zip" -OutFile "$zip.part" -UseBasicParsing
            Move-Item "$zip.part" $zip -Force
        } catch {
            Fail "download failed ($($_.Exception.Message)). Download https://files.grouplens.org/datasets/movielens/ml-25m.zip manually into the raw folder and re-run."
        }
    }
    Write-Host "[data] extracting $zip ..."
    Expand-Archive -Path $zip -DestinationPath "raw" -Force
    if (-not (Test-Path "$raw\ratings.csv")) { Fail "$raw\ratings.csv still missing after extraction" }
}
Write-Host "[data] $raw ready"

# 4) free stages: interactions -> k-core -> RecBole .inter -> train-only profiles -> cost estimate
& $Python -u run_pipeline.py --k $K --split $Split --steps 0ATPS
if ($LASTEXITCODE -ne 0) { Fail "data preparation failed (see above)" }

if ($PrepOnly) {
    # the other temporal protocol of the earlier repo: only its train-only profiles + estimate (free)
    $other = if ($Split -eq "TO") { "TO_GLOBAL" } else { "TO" }
    & $Python -u run_pipeline.py --k $K --split $other --steps PS
    if ($LASTEXITCODE -ne 0) { Fail "data preparation for $other failed (see above)" }
    Write-Host ""
    Write-Host "[done] data prepared (no LLM call made):"
    Write-Host "  size check : run_k$($K)_$($Split)\scale_check.json"
    Write-Host "  estimates  : run_k$($K)_$($Split)\llm_cost_estimate.json, run_k$($K)_$($other)\llm_cost_estimate.json"
    Write-Host "  next       : scripts\intent_pilot.bat (pilot) or scripts\intent_full.bat (full run)"
    exit 0
}

if (-not $Full) {
    # 5a) pilot: small paid sample -> real throughput, projected time, sample intents
    & $Python -u pilot_llm.py --k $K --split $Split --n_users $PilotUsers --n_items $PilotItems
    if ($LASTEXITCODE -ne 0) { Fail "pilot failed (see above)" }
    Write-Host ""
    Write-Host "Pilot report: run_k$($K)_$($Split)_pilot\pilot_report.md"
    Write-Host "Full run:     scripts\intent_full.bat"
    exit 0
}

# 5b) full LLM stages (run_pipeline asks y/N with the cost estimate before the first paid call)
& $Python -u run_pipeline.py --k $K --split $Split --steps BCD
$rc = $LASTEXITCODE
if ($rc -ne 0) { Fail "LLM stage stopped (exit $rc). Fix the cause above and run the same .bat again to resume." }
Write-Host "[done] intents + KG packs in run_k$($K)_$($Split)\  -> next: training/evaluation (--steps ER)"
