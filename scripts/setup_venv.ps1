# Create the Python environment for the IKGR pipeline on Windows (NVIDIA GPU).
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1
#
# - Location: %USERPROFILE%\.venvs\goodreads-ikgr (outside OneDrive; a GPU torch venv is ~5 GB).
#   Same dependency set as the Goodreads repo, so the existing venv is reused as-is.
#   Pass -VenvDir to keep a separate environment for this project.
# - Base interpreter: the first Python 3.10-3.12 found among
#     -BasePython, "py -3.11", "py -3.12", "py -3.10", %USERPROFILE%\anaconda3\python.exe, python
# - Installs torch 2.3.0 (CUDA 12.1 build) + requirements-pipeline.txt, then checks imports and CUDA.
# - Idempotent: an existing, working venv is reused (use -Force to rebuild).
param(
    [string]$VenvDir = "$env:USERPROFILE\.venvs\goodreads-ikgr",
    [string]$BasePython = "",
    [string]$TorchSpec = "torch==2.3.0",
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu121",
    [switch]$Force
)

$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
if ($IsLinux -or $IsMacOS) { $venvPy = Join-Path $VenvDir "bin/python" } else { $venvPy = Join-Path $VenvDir "Scripts\python.exe" }
$check = "import ssl, torch, recbole, sentence_transformers, sklearn, scipy, yaml, requests; " +
         "print('torch', torch.__version__, '| recbole', recbole.__version__, '| cuda', torch.cuda.is_available())"

function Invoke-Cmd([string[]]$Cmd, [string[]]$Rest) {
    $exe = $Cmd[0]
    $pre = @()
    if ($Cmd.Count -gt 1) { $pre = $Cmd[1..($Cmd.Count - 1)] }
    & $exe @pre @Rest
}

function Get-PyVersion([string[]]$Cmd) {
    try {
        $v = Invoke-Cmd $Cmd @("-c", "import sys; print('%d.%d' % sys.version_info[:2])") 2>$null
        if ($LASTEXITCODE -eq 0 -and $v) { return ([string]$v).Trim() }
    } catch { }
    return $null
}

if ((Test-Path $venvPy) -and -not $Force) {
    & $venvPy -c $check
    if ($LASTEXITCODE -eq 0) { Write-Host "[venv] ready: $venvPy"; exit 0 }
    Write-Host "[venv] $VenvDir exists but imports fail -> installing packages into it"
} else {
    if ((Test-Path $VenvDir) -and $Force) { Remove-Item -Recurse -Force $VenvDir }
    $candidates = @()
    if ($BasePython) { $candidates += , @($BasePython) }
    $candidates += , @("py", "-3.11")
    $candidates += , @("py", "-3.12")
    $candidates += , @("py", "-3.10")
    $candidates += , @("$env:USERPROFILE\anaconda3\python.exe")
    $candidates += , @("python")
    $base = $null
    foreach ($c in $candidates) {
        if (-not (Get-Command $c[0] -ErrorAction SilentlyContinue) -and -not (Test-Path $c[0])) { continue }
        $ver = Get-PyVersion $c
        Write-Host "[venv] candidate $($c -join ' ') -> $ver"
        if ($ver -in @("3.10", "3.11", "3.12")) { $base = $c; break }
    }
    if (-not $base) {
        Write-Host "[venv] No Python 3.10-3.12 found. Install Python 3.11 from https://www.python.org/downloads/ (check 'py launcher'),"
        Write-Host "       or pass -BasePython <path to python.exe>, or run the pipeline with -Python <existing env python>."
        exit 1
    }
    Write-Host "[venv] creating $VenvDir with $($base -join ' ')"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $VenvDir) | Out-Null
    Invoke-Cmd $base @("-m", "venv", $VenvDir)
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPy)) { Write-Host "[venv] venv creation failed"; exit 1 }
}

# A venv made from an Anaconda/conda base python cannot see <base>\Library\bin, where the
# OpenSSL (and other) DLLs live -> "SSL module is not available" in pip / HTTPS calls.
# Register that directory at interpreter start-up via a .pth file (no-op elsewhere).
if (-not ($IsLinux -or $IsMacOS)) {
    $cfgFile = Join-Path $VenvDir "pyvenv.cfg"
    if (Test-Path $cfgFile) {
        $homeLine = Get-Content $cfgFile | Where-Object { $_ -match "^\s*home\s*=" } | Select-Object -First 1
        if ($homeLine) {
            $baseHome = ($homeLine -split "=", 2)[1].Trim()
            $libBin = Join-Path $baseHome "Library\bin"
            $site = Join-Path $VenvDir "Lib\site-packages"
            if ((Test-Path $libBin) -and (Test-Path $site)) {
                $line = "import os; os.add_dll_directory(r'$libBin') if hasattr(os, 'add_dll_directory') and os.path.isdir(r'$libBin') else None"
                Set-Content -Path (Join-Path $site "conda_base_dlls.pth") -Value $line -Encoding ASCII
                Write-Host "[venv] registered conda DLL dir for SSL: $libBin"
            }
        }
    }
}

& $venvPy -m pip install --upgrade pip
if ($TorchIndex) { & $venvPy -m pip install $TorchSpec --index-url $TorchIndex } else { & $venvPy -m pip install $TorchSpec }
if ($LASTEXITCODE -ne 0) { Write-Host "[venv] torch install failed"; exit 1 }
& $venvPy -m pip install -r requirements-pipeline.txt
if ($LASTEXITCODE -ne 0) { Write-Host "[venv] requirements install failed"; exit 1 }
& $venvPy -c $check
if ($LASTEXITCODE -ne 0) { Write-Host "[venv] import check failed"; exit 1 }
Write-Host "[venv] ready: $venvPy"
exit 0
