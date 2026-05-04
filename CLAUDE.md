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
│   ├── raw/                     downloaded CSVs (gitignored; provenance in manifest)
│   ├── processed/               built panels (gitignored; provenance in manifest)
│   │   └── features/            per-panel feature CSVs (gitignored; provenance in features_manifest)
│   └── manifests/               raw_manifest.json, processed_manifest.json,
│                                features_manifest.json (all tracked)
├── src/seet/
│   ├── __init__.py
│   └── features.py              feature builder — build_features(panel_df, feature_set_id)
├── experiments/                 experiment entry points (added per-track)
├── outputs/                     figures, tables, run logs (gitignored)
├── tests/
│   ├── test_processed_panels.py smoke test for processed layer
│   └── test_features.py         structural + warmup + no-look-ahead + NaN propagation
├── scripts/                     one-shot runners
│   ├── fetch_raw.py             yfinance + CBOE-RVX-fallback raw fetch
│   ├── import_rvx_from_cboe.py  standalone CBOE RVX normalizer
│   ├── build_processed.py       build the four processed panels
│   ├── build_features.py        build the per-panel feature CSVs
│   └── audit_panels.py          panel + feature NaN-region diagnostics
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

### Features

Per-panel feature CSVs in `data/processed/features/`, one per processed
panel. Built by `python scripts/build_features.py`, which dispatches to
the public API in `src/seet/features.py`:

```python
from seet.features import build_features
features_df, specs = build_features(panel_df, feature_set_id)
```

Provenance recorded in `data/manifests/features_manifest.json`
(tracked) with per-panel sha256, row count, first/last date,
feature_count, max_warmup_rows, source-panel sha256 (so feature/panel
drift is detectable), schema_version, build timestamp, and the full
per-feature schema (`name`, `feature_group`, `formula`, `depends_on`,
`lookback_window`, `schema_version=1`).

| Panel | feature_set_id | Features | Notes |
|-------|----------------|---------:|-------|
| `spx_extended_2011` | `spx_full`     | 35 | Full SPX vol family — Groups 1, 2, 3, 4 |
| `spx_core_2007`     | `spx_core`     | 23 | Reduced — no VIX9D/VIX6M; subset of Group 3 |
| `ndx_2007`          | `ndx_minimal`  | 12 | NDX/VXN only — Groups 1, 2 |
| `rut_2009`          | `rut_minimal`  | 12 | RUT/RVX only — Groups 1, 2 |

Feature groups:

1. **Group 1 — index returns and realized vol** on the panel's primary
   price column. `ret_1d`, `ret_5d`, `ret_10d`, `ret_20d`, `logret_1d`,
   `rv_10d`, `rv_20d`, `drawdown_20d`. Realized vol is rolling std of
   `logret_1d` annualized by `sqrt(252)`.
2. **Group 2 — vol level transformations** on each vol column.
   `<vol>_chg_1d`, `<vol>_chg_5d`, `<vol>_pctchg_5d`,
   `<vol>_pctile_252d` (rolling 252-day percentile rank).
3. **Group 3 — term structure** (SPX panels only). Differences and
   ratios across VIX, VIX3M, VIX6M, VIX9D plus binary indicators
   (`curve_inversion`, `short_end_spike`). Indicators are
   float-with-NaN, never coerced to 0.
4. **Group 4 — vol-of-vol** (SPX panels only). Group 2 mechanics
   applied to VVIX. Tagged `feature_group=4` in the manifest to mark
   the second-order interpretation, even though the formulas are
   identical to Group 2.

#### Contracts (project invariants — anything breaking these is a bug)

- **No look-ahead.** Every feature at date *t* uses only data with
  `Date <= t`. All rolling and shift operations are trailing-only.
  Verified by `tests/test_features.py::test_no_lookahead`, which
  samples 50 random dates per panel, slices the panel to rows
  `[0..t]`, rebuilds features on the slice, and asserts bit-for-bit
  equality with the materialized `features[t]`. A failure here
  invalidates every downstream model — do not paper over it.
- **Warmup is NaN, never zero.** A feature with `lookback_window = N`
  has its first `N − 1` rows NaN. No `fillna(0)`, no forward-fill, no
  interpolation. Max warmup across all four panels is **251 rows**
  (driven by `*_pctile_252d`).
- **Data-quality NaNs propagate.** The 5 documented RVX gap dates and
  the 8 documented VVIX gap dates produce NaN in every feature whose
  `depends_on` includes the gap-bearing column, at the exact gap dates.
  Rolling-window features additionally have NaN for up to
  `lookback_window` trailing rows after each gap (pandas defaults with
  `min_periods=window`). Same policy as the raw layer: **do not
  interpolate**.
- **Group 3 indicators are NaN-aware.** `curve_inversion` and
  `short_end_spike` are float64 with explicit NaN propagation; either
  operand NaN → result NaN.

#### Smoke-test guarantees (`tests/test_features.py`)

40 parametrized test cases (10 functions × 4 panels). Expected:
**39 passed, 1 skipped** (the skip is
`test_data_quality_nan_propagates_to_features[ndx_2007]` because
`ndx_2007` has no `data_quality_notes`). Coverage:

- **Structural.** Row count matches source panel; Date column matches
  source panel; columns match manifest order; all feature dtypes are
  `float64`; feature CSV sha256 matches manifest;
  `features_manifest.source_panel_sha256` matches
  `processed_manifest.sha256` (catches drift if a panel rebuilds
  without features re-building); manifest specs are well-formed
  (schema_version, group ∈ {1,2,3,4}, lookback_window ≥ 1).
- **Warmup.** First `lookback_window − 1` rows of each feature are
  NaN.
- **Data-quality propagation.** At every documented missing date in
  `processed_manifest.data_quality_notes`, every feature whose
  `depends_on` includes the missing series is NaN at that exact row.
- **No look-ahead.** The integrity test described above.

#### Auditor coverage (`scripts/audit_panels.py`)

Two-section report:

1. **PANEL AUDIT** — raw NaN regions classified against truncations
   and `data_quality_notes` (truncation blocks at index 0 are
   expected; documented gap dates are expected; everything else is
   flagged).
2. **FEATURE AUDIT** — each feature CSV's NaN cells classified
   against `expected_NaN = warmup_mask ∪ rolling-OR(input_nan_mask,
   lookback_window)` over each input column. The expected mask is
   conservative (a strict superset of pandas' actual NaN); any actual
   NaN beyond the expected mask is a genuine anomaly. Skipped if
   `features_manifest.json` is absent.

Exit 0 with all panels `[OK]` in both sections means every on-disk
NaN — panel or feature — is warmup, a known truncation, a documented
data-quality gap, or the rolling-window propagation of one of those.

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
3. To refresh the data + features layer end-to-end:
   `python scripts/fetch_raw.py` →
   `python scripts/build_processed.py` →
   `python scripts/build_features.py` →
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
- `209a3df`  — audit_panels.py + RVX/VVIX gap notes in processed_manifest.json
- `7b71b1f`  — smoke test for processed panels
- `0ac6aab`  — CLAUDE.md (project orientation)
- `c186ae1`  — track scripts/import_rvx_from_cboe.py (predecessor of
  fetch_raw.py auto-fallback; was on disk but never `git add`-ed
  earlier)
- `491dcaa`  — feature engineering: src/seet/features.py + four feature panels
- `f7722b9`  — smoke test for features (structural + warmup + no-look-ahead + NaN propagation)
- `0c84df3`  — audit_panels.py: feature-aware FEATURE AUDIT section
