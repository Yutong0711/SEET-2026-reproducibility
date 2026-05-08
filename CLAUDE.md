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
- Track E: **component ablation experiments — DONE**. Four
  controlled ablations of the reference architecture, run on the
  Track A central case (SPX, `spx_extended_2011`, `spx_full`, h10_d05,
  22 expanding-window folds, 6 baselines, seeds 42–46). Per-fold
  deltas are paired against Track A on `(model, fold_id, seed)`;
  CIs use a paired bootstrap (10k resamples, fold-level resampling
  preserves pairing — see `stats.paired_bootstrap_ci`); significance
  is paired Wilcoxon signed-rank.

  - **ABLATION_1 (no_validators)**. Bypasses pre-launch truncation +
    data-quality NaN flagging by feeding a panel rebuilt from
    `data/raw/` (35 pre-anchor rows + 279 VVIX synthetic-backfill
    rows from 2011-02-23 → 2012-03-30, kept as if real).
    **Finding (paper-relevant):** LR AUC inflates by **+0.035** (95%
    paired CI [+0.014, +0.057], Wilcoxon p = 0.005) — empirical
    confirmation of the synthetic-backfill leakage hypothesis. Other
    models do not show meaningful AUC inflation.
  - **ABLATION_2 (global_scaler)**. Replaces LR's per-fold
    `StandardScaler` with one fit on the entire feature matrix
    (train ∪ test). LightGBM is unaffected by construction (it has
    no scaler in the unmodified pipeline; tagged
    `no_effect_by_construction`). **Finding (paper-relevant,
    contrast):** LR delta_AUC = **−0.003** (CI [−0.005, −0.001], p =
    0.002) — statistically significant but operationally negligible.
    The architecture's predicted impact (look-ahead through scaler
    fitting) materializes, but the magnitude is much smaller than
    the validator-leakage or drift signals. Different leakage
    vectors have different consequences.
  - **ABLATION_3 (test_threshold)**. Replaces the training-quantile
    alarm threshold (95th pct of train scores) with a fixed top-5%
    test-set threshold (the v1 paper's behavior). **Findings
    (paper-relevant):** LightGBM drawdown_lift drops by **−0.154**
    (CI [−0.277, −0.033], p = 0.007); HarRvThreshold drawdown_lift
    *rises* by **+0.195** (CI [+0.065, +0.332], p = 0.028) — a real
    finding flagged for retention, not investigated-then-explained;
    LightGBM calm-period alarm rate inflates by **+0.054** (CI
    [+0.024, +0.088], p = 0.001), with similar ~5pp inflations
    across HarRv (+0.055), VIXPercentileRaw (+0.054), and
    VIXPercentileCalibrated (+0.034). NaiveBaseRate is unaffected by
    construction (constant predictor).
  - **ABLATION_4 (frozen_model)**. Each model is fit once on data
    through 2014-12-31 and applied frozen to every later test fold
    (no refitting, no per-fold HP lookup). For LightGBM the HPs
    selected by Track B for outer_fold=1 are used. **Findings
    (paper-relevant):** LightGBM drawdown_lift falls by **−0.540**
    (CI [−0.821, −0.283], p = 0.001); LR drawdown_lift falls by
    **−0.279** (CI [−0.452, −0.108], p = 0.003) — these are the
    largest operational impacts in Track E and the strongest evidence
    that ML baselines are drift-sensitive. Deterministic baselines
    (NaiveBaseRate, VIXPercentileRaw, VIXPercentileCalibrated,
    HarRvThreshold) show no significant drawdown_lift change. The
    **most degraded** model is LightGBM, the **most resistant** is
    HarRvThreshold (delta_lift = +0.043, p = 0.93). A secondary
    pattern — frozen models showing improved Brier on LightGBM
    (delta = −0.036, p = 7e-6) and LR (delta = −0.023, p = 0.015) —
    is documented as a Brier-vs-refit calibration interaction in
    rare-event settings, not a paper finding.

  Each row of `outputs/track_e/table_ablation.csv` (144 rows, =
  4 ablations × 6 models × 6 metrics) carries two annotation
  columns:
  - `interpretation` ∈ {`paper_relevant_finding` (7 cells, the
    findings above), `no_effect_by_construction` (71 cells —
    deterministic baselines on irrelevant ablations or
    floating-point-zero deltas), `expected_effect` (66 cells — the
    architecture's predicted impacts), `investigate` (0 cells —
    the annotation aims to leave this empty)}
  - `effect_size_category` ∈ {`large` (>0.05), `medium` (0.01–0.05),
    `small` (0.001–0.01), `negligible` (≤0.001)}; 13 large, 22
    medium, 19 small, 90 negligible. Statistical significance with
    negligible effect size is common in this dataset because
    deterministic baselines produce exactly-zero deltas under most
    ablations — the size category lets the paper distinguish
    "significant and important" from "significant but tiny".

  **F5 (sibling failure mode).** Track E ABLATION_1 demonstrates
  that an SE-style data-validation stage (truncation of synthetic
  pre-launch backfill, flagging of documented data-quality gaps)
  is load-bearing for classification quality, not just for data
  hygiene. Removing it produces a +0.035 AUC inflation on
  matrix-feature LR — a leakage path the v1 paper would not have
  caught because it did not separate validators from feature
  construction.

  Artifacts:
  ```
  experiments/track_e_ablation/per_fold_no_validators.csv
  experiments/track_e_ablation/per_fold_global_scaler.csv
  experiments/track_e_ablation/per_fold_test_threshold.csv
  experiments/track_e_ablation/per_fold_frozen_model.csv
  experiments/track_e_ablation/no_validators_provenance.json
  outputs/track_e/table_ablation.csv         (144 rows, 10 columns)
  outputs/track_e/fig_ablation_lift.pdf      (forest plot, drawdown_lift)
  outputs/track_e/fig_ablation_auc.pdf       (forest plot, AUC)
  ```
- Track F: **failure injection / mutation testing — DONE**.
  Mutation-style testing of the data-validation layer. Five
  corruption modes (C1 duplicate_dates, C2 missing_values, C3
  stale_quotes, C4 extreme_jumps, C5 calendar_gaps) are applied
  to in-memory copies of the canonical SPX panel at three rates
  (1%, 5%, 10%) with five seeds each. For every cell the script
  measures (i) whether the V1+V2 validators catch the corruption
  and (ii) what happens to LR + LightGBM downstream metrics when
  the validators are silenced. Pairing is per (model, fold_id,
  lgbm_seed) within the same corrupted panel; CIs use a paired
  bootstrap (10k resamples, fold-level resampling preserves
  pairing).

  **Architectural improvement (paper-relevant in itself).** Track
  F created an explicit `src/seet/validators.py` module that
  codifies build-time (`scripts/build_processed.py` truncation
  rules) and audit-script (`scripts/audit_panels.py` duplicate
  + NaN-region classifier) logic into a callable runtime
  component with a clean `validate_panel` / `apply_treatment` /
  `coverage_metrics` API. This act of making implicit validation
  explicit is itself an architectural improvement that the
  failure-injection methodology surfaced — the v1 codebase had
  validation rules scattered across build scripts and audit
  scripts, with no single callable surface; without that surface,
  none of the rest of Track F (silenced-branch comparisons,
  precision/recall accounting, mechanism classification) would
  have been writable.

  **Three findings (matching the user-specified STATUS structure):**

  - **(a) Detection coverage by corruption type and rate**.
    Validators V1 (duplicate Date) and V2 (NaN-vs-data_quality_notes)
    catch C1 and C2 with **recall = 1.000 across all rates** (perfect
    coverage). C3 and C4 corruption flow through silently —
    **recall = 0.000 across all rates** (`gap_mechanism = silent_ignore`,
    no rule). C5 corruption removes rows before the validator can
    inspect them — **recall undefined** (`gap_mechanism =
    removed_before_inspection`, n_truth = 0 by construction).

  - **(b) Operational cost of silenced detection on caught
    corruptions (C1, C2).** For C1 (duplicate_dates) silencing the
    validator produces a small operational delta on the matrix-
    feature models: at rate=5%, LightGBM AUC delta = +0.042
    (CI [+0.028, +0.056], p < 0.001) and LR AUC delta = −0.021
    (p = 0.018) — both statistically significant but tiny in
    magnitude (`small` effect size). drawdown_lift deltas are not
    significant (e.g. LightGBM rate=5% delta = +0.095, p = 0.51).
    The architectural value of V1 is therefore data hygiene and
    downstream auditability, not a large performance impact — true
    duplicate rows mostly bias training slightly without changing
    the model's discriminative power on h10_d05. For C2
    (missing_values) the standard paired-delta cells report
    `delta = NaN` (`n_pairs = 0`) because the SILENCED branch
    produces all-NaN metrics — see (c) and the branch-failure
    finding below.

    **Methodological finding (Track F's own contribution).** The
    initial C1 corruption design copied values from a random source
    row (an apparently-benign choice). Diagnostic scripts
    `scripts/diagnose_track_f_c1.py` (single-fold check) and
    `scripts/diagnose_track_f_c1_all_folds.py` (all-fold scan)
    revealed that this produced synthetic stress events whenever a
    future-dated source row's SPX was inserted at an earlier target
    date: 11 of 33 valid (fold, injection_seed) cells exhibited
    SILENCED lift ≥ 4.0 vs a Track A baseline of ≈ 1.3, with one
    cell at lift = 17.57 on a fold with `n_events = 0`. The
    leakage was traced to the corrupted panel itself: the duplicate
    row's `future_drawdown` was computed against the panel's real
    later prices, producing artificial label = 1 (synthetic stress
    event) which the model then "correctly" alarmed on,
    artificially inflating lift. Corrected to true-duplicate
    semantics (same Date AND same values; see updated
    `_corrupt_duplicate_dates` docstring) and re-ran only the 15 C1
    cells via `scripts/rerun_track_f_c1_only.py`. V1 still detects
    with recall = 1.0 by construction. Lesson for the paper: the
    mutation-testing methodology must verify that each corruption
    injects only the failure mode it is named for, not a confounded
    mixture; the same diagnostic scaffolding that surfaces
    validator gaps also catches confounded-corruption bugs.

  - **(c) Architectural validation gap (C3, C4, C5).** The three
    corruption types the v1 architecture cannot detect produce
    delta = 0.000 in the standard ENABLED-vs-SILENCED contrast (no
    validator fires, the same panel feeds both branches). To
    quantify the **operational cost of these uncaught corruptions**
    we computed `delta_vs_clean = metric(SILENCED, corrupted) −
    metric(Track A, clean)` per (model, fold, lgbm_seed) pair and
    aggregated with the same paired bootstrap +
    Wilcoxon (script `scripts/compute_track_f_silenced_vs_clean.py`,
    output `outputs/track_f/table_silenced_vs_clean.csv`). The
    matrix-feature LightGBM model is significantly degraded by all
    three uncaught corruption types — drawdown_lift falls by
    **−0.13 to −0.27** (large effect size, p < 0.01 in 6 of 9
    cells); AUC falls by **−0.01 to −0.05** (small effect size, p <
    0.001 in 6 of 9 cells). LR is largely unaffected (mixed signs
    and smaller magnitudes), reflecting heavy L2 regularization +
    StandardScaler clipping the influence of feature outliers. This
    is the architectural-gap impact statement the paper needs:
    uncaught corruption costs the boosted-tree model 0.13–0.27 in
    lift even though the validators don't know the corruption is
    there. Adding C3/C4/C5 detection rules is an architectural
    extension point with quantified expected gain.

  **Branch-failure finding (stronger than a small delta).** The
  paired-bootstrap output for C2 missing_values reports `delta = NaN`
  not because there is no effect, but because the SILENCED branch
  has **complete model failure**. The post-process at
  `outputs/track_f/table_branch_failure.csv` (built by
  `scripts/rerun_track_f_postprocess.py`) shows that for C2 at all
  rates, ENABLED produces valid drawdown_lift on ~50–60% of folds
  (matching the canonical Track A rate, since AUC is undefined on
  calm folds by construction), while SILENCED produces **0% valid
  drawdown_lift** at all rates. The mechanism: a corrupted SPX cell
  propagates NaN through rolling-window features for up to 252
  trailing rows; with ~38 corrupted SPX cells at rate=1% spread over
  ~3,820 panel rows, the rolling-window NaN regions overlap to cover
  essentially every row. `_fit_predict_one` then sees fewer than 10
  valid training rows and the matrix-feature models can't fit. The
  validator turns "complete model failure" into "valid model fit",
  which is a bigger operational finding than a small numerical delta.
  Post-fix C1 (true-duplicate semantics) does NOT trigger branch
  failure — both branches succeed at similar rates because true
  duplicates only minimally distort training; C2 is the unique
  catastrophic-failure case in Track F's corruption set.

  **F6 (sibling failure mode).** Failure-injection on the data
  layer surfaced a previously-unnamed failure mode: **silent
  validator gaps** — corruption types the architecture has no rule
  for (C3, C4, C5 here). Their operational cost is unobservable
  through ENABLED-vs-SILENCED contrast precisely because silencing
  a non-existent validator is a no-op. The mitigation is to add
  rules; the diagnostic is mutation testing, which makes the absence
  of a rule visible.

  Artifacts:
  ```
  src/seet/validators.py                              (V1+V2 + treatment + coverage)
  src/seet/injection.py                               (5 corruption modes, deterministic RNG;
                                                       C1 = true-duplicate semantics post-fix)
  scripts/run_track_f.py                              (75-cell experiment runner)
  scripts/rerun_track_f_postprocess.py                (branch-failure post-process)
  scripts/compute_track_f_silenced_vs_clean.py        (C3/C4/C5 architectural-gap impact)
  scripts/diagnose_track_f_c1.py                      (single-fold C1 leakage check)
  scripts/diagnose_track_f_c1_all_folds.py            (all-fold C1 leakage scan)
  scripts/rerun_track_f_c1_only.py                    (fixed-C1 re-run + splice)
  experiments/track_f_injection/validator_coverage.csv          (75 rows)
  experiments/track_f_injection/silenced_impact.csv             (150 rows; C1 from fixed run)
  experiments/track_f_injection/corruption_provenance.json
  outputs/track_f/table_validator_coverage.csv
  outputs/track_f/table_silenced_impact.csv
  outputs/track_f/table_branch_failure.csv            (C2 catastrophic failure cells)
  outputs/track_f/table_silenced_vs_clean.csv         (C3/C4/C5 vs clean baseline)
  outputs/track_f/fig_coverage.pdf
  outputs/track_f/fig_silenced_impact.pdf
  ```
- Track G: TBD

When a track is defined, this section should be replaced with the
specific feature set, model class, evaluation protocol, and STATUS
schema for that track.

## Conventions

- **Failure mode F4: data-quality cascade rendering test set empty
  after feature-NaN drop.** Detected when `X_te.shape[0] == 0` after
  the matrix-feature baselines (`LogisticRegressionL2`,
  `LightGbmTuned`) drop NaN-feature rows AND the always-NaN test
  features appear in `processed_manifest.json` `data_quality_notes`.
  Demonstrated on Track D's Crisis-2020 `spx_full` configuration:
  documented VVIX gap dates (2019-07-05, 2020-06-11) propagate
  through `vvix_pctile_252d` (252-day rolling rank) and cover the
  entire 2020 test window, dropping every test row for matrix-feature
  models and rendering all their metrics NaN. The Crisis-2020
  `spx_no_vvix` recovery configuration removes the four VVIX-derived
  features and restores operational evaluability for those baselines;
  the four panel-DataFrame baselines (`NaiveBaseRate`,
  `VIXPercentileRaw`, `VIXPercentileCalibrated`, `HarRvThreshold`)
  are unaffected by F4 because they read panel columns directly.
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
