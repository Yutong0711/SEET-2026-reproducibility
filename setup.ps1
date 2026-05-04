# Bootstrap the seet2026 project on Windows (PowerShell).
# Idempotent: safe to re-run. Will not clobber existing commits.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# 1. Pick a Python 3.11+ interpreter
$py = "python"
$pyv = & $py -c "import sys;print('%d.%d'%sys.version_info[:2])"
if ($pyv -notmatch '^3\.(1[1-9]|[2-9]\d)') {
  Write-Error "Need Python 3.11+, found $pyv"
  exit 1
}
Write-Host "Using $py ($pyv)"

# 2. venv
if (-not (Test-Path .venv)) {
  & $py -m venv .venv
}
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel

# 3. Install pinned minimums
pip install `
  "numpy>=1.26" `
  "pandas>=2.1" `
  "scikit-learn>=1.4" `
  "lightgbm>=4.1" `
  "matplotlib>=3.8" `
  "scipy>=1.11" `
  "pytest>=7.4" `
  "pyyaml>=6.0" `
  "pandas-datareader>=0.10" `
  "yfinance>=0.2.40" `
  "requests>=2.31" `
  "tqdm>=4.66" `
  "joblib>=1.3"

# 4. Lockfile
pip freeze | Out-File -Encoding ascii requirements.lock

# 5. Ensure runtime dirs exist
New-Item -Force -ItemType Directory data\raw, data\processed, data\manifests, outputs | Out-Null

# 6. Git init + initial commit (only if no commits yet)
if (-not (Test-Path .git)) {
  git init -q
  git symbolic-ref HEAD refs/heads/main 2>$null
}
git rev-parse --verify HEAD 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
  git add -A
  $env:GIT_AUTHOR_NAME = if ($env:GIT_AUTHOR_NAME) { $env:GIT_AUTHOR_NAME } else { "seet" }
  $env:GIT_AUTHOR_EMAIL = if ($env:GIT_AUTHOR_EMAIL) { $env:GIT_AUTHOR_EMAIL } else { "seet@local" }
  $env:GIT_COMMITTER_NAME = $env:GIT_AUTHOR_NAME
  $env:GIT_COMMITTER_EMAIL = $env:GIT_AUTHOR_EMAIL
  git commit -q -m "Initial scaffold: dirs, deps lock, fetch_raw"
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Next:  .\.venv\Scripts\Activate.ps1; python scripts\fetch_raw.py"
