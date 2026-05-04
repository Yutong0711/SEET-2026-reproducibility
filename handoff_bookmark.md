# SEET 2026 — Session Handoff Bookmark

**Date:** 2026-05-04
**Branch:** `main`
**HEAD:** `72f935e`

## Status

**Track A and Track B are 100% complete. Artifacts are frozen.**

The repository is internally consistent: working tree clean, 12 commits
on `main`, all tests pass. Every module imported by the runner and the
tuner is in version control. The full pipeline — data → features →
statistics → baselines → Track A four-layer evaluation → Track B nested
HP tuning → Track A re-application — reproduces from a fresh clone.

## Immediate next step on resumption

**PROMPT 3 — Track C: multi-asset replication.**

Do not start new experiments before PROMPT 3 is written. Whatever Track C
needs (asset selection, fold structure, evaluation scheme, model
inheritance from Track A/B, etc.) will be specified there.

## Sanity check after resuming

```powershell
cd C:\SEET-2026
.\.venv\Scripts\Activate.ps1
git status                  # expect: nothing to commit, working tree clean
git log --oneline -1        # expect: 72f935e on top
pytest -q tests/            # expect: all tests pass
```

## Frozen artifacts (where to look)

- Code: `src/seet/{features,stats,baselines,tuning,run_track_a}.py`
- Tuning: `experiments/track_b_tuning/{grid.yaml, selected_hp.csv, inner_scores.csv}`
- Track A run: `experiments/track_a_headline/{fold_definitions.csv, per_fold_metrics.csv}`
- Aggregated tables + figures: `outputs/track_a/`, `outputs/track_b/`
- Project orientation for future sessions: `CLAUDE.md`

Do not edit any of the above outside of an explicit Track C prompt.
