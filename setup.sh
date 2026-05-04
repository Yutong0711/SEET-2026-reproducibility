#!/usr/bin/env bash
# Bootstrap the seet2026 project: venv, pinned-min deps, lockfile, git init.
# Idempotent: safe to re-run. Will not clobber existing commits.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# 1. Pick a Python 3.11+ interpreter
PY=python3
if command -v python3.11 >/dev/null 2>&1; then PY=python3.11; fi
PYV=$("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])')
case "$PYV" in
  3.11|3.12|3.13|3.14|3.15) ;;
  *) echo "Need Python 3.11+, found $PYV" >&2; exit 1 ;;
esac
echo "Using $PY ($PYV)"

# 2. venv
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel

# 3. Install pinned minimums
pip install \
  "numpy>=1.26" \
  "pandas>=2.1" \
  "scikit-learn>=1.4" \
  "lightgbm>=4.1" \
  "matplotlib>=3.8" \
  "scipy>=1.11" \
  "pytest>=7.4" \
  "pyyaml>=6.0" \
  "pandas-datareader>=0.10" \
  "yfinance>=0.2.40" \
  "requests>=2.31" \
  "tqdm>=4.66" \
  "joblib>=1.3"

# 4. Lockfile
pip freeze > requirements.lock

# 5. Ensure runtime dirs exist (some are gitignored)
mkdir -p data/raw data/processed data/manifests outputs

# 6. Git init + initial commit (only if no commits yet)
if [ ! -d .git ]; then
  git init -q
  git symbolic-ref HEAD refs/heads/main 2>/dev/null || true
fi
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  git add -A
  GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-seet}" \
  GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-seet@local}" \
  GIT_COMMITTER_NAME="${GIT_COMMITTER_NAME:-seet}" \
  GIT_COMMITTER_EMAIL="${GIT_COMMITTER_EMAIL:-seet@local}" \
  git commit -q -m "Initial scaffold: dirs, deps lock, fetch_raw"
fi

echo
echo "Setup complete."
echo "Next:  source .venv/bin/activate && python scripts/fetch_raw.py"
