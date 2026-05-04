# CLAUDE.md — SEET 2026 Submission

This file orients Claude (and other AI assistants) working on this
repository. It is internal scaffolding, not paper text. The paper itself
is written by humans, per SEET 2026 policy (see Conventions below).

## Paper

Revision for SEET 2026. The paper proposes a reference architecture and a
four-layer evaluation methodology for rare-event ML monitoring,
demonstrated on an equity-stress case study. The methodological
contribution is the four-layer evaluation framework; the equity-stress
analysis is the empirical anchor.

## Repository layout

```
seet2026/
├── data/
│   ├── raw/         downloaded CSVs (gitignored; provenance in manifest)
│   ├── processed/   built panels (gitignored; provenance in manifest)
│   └── manifests/   raw_manifest.json, processed_manifest.json (tracked)
├── src/seet/        package code (added per-turn)
├── experiments/     experiment entry points (added per-track)
├── outputs/         figures, tables, run logs (gitignored)
├── tests/           pytest suites
├── scripts/         one-shot runners
│   ├── fetch_raw.py             yfinance + CBOE-RVX-fallback raw fetch
│   ├── import_rvx_from_cboe.py  standalone CBOE RVX normalizer
│   ├── build_processed.py       build the four processed panels
│   └── audit_panels.py          NaN-region and structural diagnostics
├── setup.sh / setup.ps1         venv, deps, lockfile, git init
├── requirements.lock            full pin from pip freeze
└── CLAUDE.md                    this file
```

## Data layer — current state

### Raw

10 daily close series in `data/raw/`, all from 2007-01-01 onward (latest
trading day at fetch time is the upper bound). Sourced from yfinance
except `RVX`, which yfinance does not serve and is fetched manually from
the CBOE export at
`https://www.cboe.com/tradable_products/vix/rvx_historical_data/`. Every
file is registered in `data/manifests/raw_manifest.json` with source
ticker, file path, sha256, row count, first/last date, and download
timestamp. The CBOE source CSV's own sha256 is also recorded.

| Series | Source       | First date | Notes |
|--------|--------------|------------|-------|
| SPX    | yfinance ^GSPC  | 2007-01-03 | |
| NDX    | yfinance ^NDX   | 2007-01-03 | |
| RUT    | yfinance ^RUT   | 2007-01-03 | |
| VIX    | yfinance ^VIX   | 2007-01-03 | |
| VIX9D  | yfinance ^VIX9D | 2011-01-03 | yfinance backfill before real launch (2011-02-23) |
| VIX3M  | yfinance ^VIX3M | 2007-01-03 | yfinance backfill before real launch (2007-12-04) |
| VIX6M  | yfinance ^VIX6M | 2008-01-02 | matches real launch (~Jan 2008) |
| VVIX   | yfinance ^VVIX  | 2007-01-03 | yfinance backfill before real launch (Mar 2012) |
| VXN    | yfinance ^VXN   | 2007-01-03 | |
| RVX    | CBOE export     | 2009-09-16 | yfinance returns empty; CBOE history begins here |

`fetch_raw.py` is idempotent and auto-falls-back to the CBOE RVX file
when yfinance returns empty for `^RVX`. If neither is available the
script prints a clear pointer and records `RVX` as FAIL — no
interpolation, no synthetic fill.

### Processed

Four panels in `data/processed/`. Truncations are applied **here**, not
in the raw layer; the raw layer is preserved as-downloaded so the
synthetic-backfill regions remain auditable. In each panel below, an
asterisk marks a column whose pre-launch values have been replaced with
NaN; the leading dates of the panel still exist (anchored to the
panel's anchor series), but the truncated column is null until its real
launch.

| Panel | Anchor | First date | Last date | Rows | Columns |
|-------|--------|------------|-----------|------|---------|
| `spx_core_2007.csv`     | SPX   | 2007-01-03 | 2026-05-01 | 4863 | SPX, VIX, VIX3M*, VVIX* |
| `spx_extended_2011.csv` | VIX9D | 2011-02-23 | 2026-05-01 | 3820 | SPX, VIX, VIX9D, VIX3M, VIX6M, VVIX* |
| `ndx_2007.csv`          | NDX   | 2007-01-03 | 2026-05-01 | 4863 | NDX, VXN |
| `rut_2009.csv`          | RUT   | 2009-09-16 | 2026-05-01 | 4182 | RUT, RVX |

Truncation thresholds (Date < threshold → NaN in the processed layer):

| Series | Drop before | Reason |
|--------|-------------|--------|
| VIX9D  | 2011-02-23  | CBOE launch date |
| VIX3M  | 2007-12-04  | CBOE launch date (originally VXV) |
| VVIX   | 2012-04-01  | CBOE launched March 2012; +1 month buffer for clean data |

Per-column metadata (raw_source, non_null_count, first_valid_date,
last_valid_date, truncated_before, truncation_reason) is in
`data/manifests/processed_manifest.json`.

### Known data quality issues

`rut_2009.csv` has 5 RVX-NaN rows where RUT has a price but the CBOE
RVX file does not: **2018-05-01, 2018-12-03, 2019-02-20, 2019-07-05,
2020-10-16**. Pattern is scattered (5 isolated single-day gaps over
~2.4 years; no clustering). These are recorded under
`rut_2009.data_quality_notes` in `processed_manifest.json`. Downstream
code may drop those rows or carry the NaN — **do not interpolate**.

`spx_core_2007.csv` and `spx_extended_2011.csv` have 8 post-truncation
VVIX-NaN rows on identical dates: **2019-07-05, 2020-06-11, 2021-01-22,
2021-02-17, 2021-05-05, 2021-05-11, 2021-05-20, 2021-05-25**. 4 of the
8 cluster in May 2021 (within a 3-week window — possibly a CBOE feed
incident that month); the other 4 are scattered. All 8 are verified
absent from `data/raw/VVIX.csv` but present in `data/raw/SPX.csv` on
the same dates, confirming the gaps originate in the CBOE VVIX feed,
not in our build pipeline. Note that 2019-07-05 (Friday after July 4)
is also a missing date in the `rut_2009` RVX gap list, suggesting a
cross-product CBOE feed issue that day. Recorded under
`data_quality_notes` for both SPX panels. Same policy: **do not
interpolate**.

`scripts/audit_panels.py` re-checks all panels for unexpected NaN
regions on demand; it treats both the documented truncations and any
`data_quality_notes` entries as expected, so a clean run (exit 0, all
panels `[OK]`) means every on-disk NaN is either a known truncation
block or a documented data-quality gap. The smoke test in
`tests/test_processed_panels.py` further asserts that the manifested
NaN regions match the on-disk panels exactly.

## Experimental plan — Tracks A through G

To be filled in as the plan finalizes. Each track produces a STATUS
block as its primary turn-end artifact (see Conventions). Placeholder
labels:

- Track A: TBD
- Track B: TBD
- Track C: TBD
- Track D: TBD
- Track E: TBD
- Track F: TBD
- Track G: TBD

When a track is defined, this section should be replaced with the
specific feature set, model class, evaluation protocol, and STATUS
schema for that track.

## Conventions

- **Random seed**: 42 is the primary seed everywhere a single seed is
  needed.
- **Stochastic models**: report results across **five seeds** — 42, 43,
  44, 45, 46 — unless a track explicitly varies this. Aggregate with
  median + 95% CI; never report a single-seed number as the headline.
- **Confidence intervals**: **95% bootstrap CIs everywhere** a point
  estimate is reported. Default bootstrap config: 10,000 resamples,
  paired across models for difference statistics.
- **Model comparisons**: **paired Wilcoxon signed-rank** (two-sided) on
  per-fold or per-window metric vectors. Report the test statistic, the
  p-value, and the median paired difference with its 95% CI. Do not
  rely on the p-value alone.
- **STATUS blocks**: every track turn ends with a STATUS block. The
  STATUS block is plain ASCII, fixed-width, includes one row per
  artifact produced (file path, sha256[:12], shape/rows, headline
  metric where applicable), and is the artifact the user inspects
  before approving the next turn.
- **Reproducibility**: manifest sha256s are the contract. Tests assert
  that on-disk files match the manifest. If a sha changes, do not
  silently re-commit — re-derive and re-test.
- **No LLM-generated paper prose**, per SEET 2026 policy. AI assistants
  may produce: code, code comments, commit messages, README/CLAUDE
  prose, plots, tables, captions for those plots/tables in code, and
  other dev artifacts. The paper body text itself (LaTeX or
  equivalent) is **written by humans**.

## Reproducibility checklist for new turns

1. Activate the venv at `.venv` (Python 3.13.5).
2. `requirements.lock` is the contract — do not upgrade without a
   deliberate reason.
3. To refresh data end-to-end:
   `python scripts/fetch_raw.py` →
   `python scripts/build_processed.py` →
   `python scripts/audit_panels.py` →
   `pytest tests/ -v`.
4. Commit per logical step, never bundle a raw refresh with a feature
   change.
5. Confirm git working tree is clean before claiming a turn is done.

## Provenance milestones

- `60355437` — initial scaffold
- `e46fd3c1` — raw data manifest + CBOE RVX manual import
- `7713290`  — fetch_raw.py auto-fallback to CBOE RVX
- `ee950dc`  — processed panels + manifest
- (subsequent commits this turn add audit_panels.py, the smoke test,
  and this CLAUDE.md)
