# seet2026

Experimental infrastructure for the SEET 2026 paper revision.

## Layout

```
seet2026/
├── data/
│   ├── raw/         # downloaded CSVs (gitignored)
│   ├── processed/   # derived datasets (gitignored)
│   └── manifests/   # raw_manifest.json, etc. (tracked)
├── src/seet/        # package code (added per-turn)
├── experiments/     # experiment entry points
├── outputs/         # figures, tables, run logs (gitignored)
├── tests/           # pytest suites
├── scripts/
│   └── fetch_raw.py # one-shot raw-data downloader
├── setup.sh         # bash bootstrap (Linux/macOS/WSL)
└── setup.ps1        # PowerShell bootstrap (Windows)
```

## First-time setup

Linux / macOS / WSL:

```bash
cd seet2026
bash setup.sh
source .venv/bin/activate
python scripts/fetch_raw.py
```

Windows (PowerShell):

```powershell
cd C:\SEET-2026
.\setup.ps1
.\.venv\Scripts\Activate.ps1
python scripts\fetch_raw.py
```

`setup.{sh,ps1}` will:
1. verify Python 3.11+,
2. create `.venv`,
3. install pinned-minimum versions of the dependencies,
4. write `requirements.lock` (output of `pip freeze`),
5. create runtime directories (`data/raw`, `data/processed`, `data/manifests`, `outputs`),
6. `git init` and create the initial commit if no commits exist.

## Pinned-minimum dependencies

`numpy>=1.26`, `pandas>=2.1`, `scikit-learn>=1.4`, `lightgbm>=4.1`,
`matplotlib>=3.8`, `scipy>=1.11`, `pytest>=7.4`, `pyyaml>=6.0`,
`pandas-datareader>=0.10`, `yfinance>=0.2.40`, `requests>=2.31`,
`tqdm>=4.66`, `joblib>=1.3`. Isotonic regression is provided by
`sklearn.isotonic`.

## Raw data

`scripts/fetch_raw.py` downloads daily close prices from Yahoo Finance
(`yfinance`) for:

| Label | Yahoo symbol | Notes |
|-------|--------------|-------|
| SPX   | `^GSPC`      | S&P 500 index |
| NDX   | `^NDX`       | Nasdaq-100 index |
| RUT   | `^RUT`       | Russell 2000 index |
| VIX   | `^VIX`       | SPX 30-day implied vol |
| VIX9D | `^VIX9D`     | series begins later than 2007 |
| VIX3M | `^VIX3M`     | series begins later than 2007 |
| VIX6M | `^VIX6M`     | series begins later than 2007 |
| VVIX  | `^VVIX`      | series begins later than 2007 |
| VXN   | `^VXN`       | NDX implied vol |
| RVX   | `^RVX`       | RUT implied vol |

Window: `2007-01-01` to today. Each ticker is saved as
`data/raw/<LABEL>.csv` with columns `Date,Close`. After the run, the
script writes `data/manifests/raw_manifest.json` containing, for each
ticker, the source symbol, file path, SHA-256, row count, first date,
last date, and download timestamp; and prints a STATUS block. Failed
tickers are reported, never interpolated.
