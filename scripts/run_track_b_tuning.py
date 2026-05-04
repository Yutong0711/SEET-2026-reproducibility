"""Track B orchestrator.

Phase B steps 1-3 + 6: probe, gate, full nested tuning, persist
selected_hp.csv and inner_scores.csv. STATUS pause comes after this
script completes — review STATUS before running the apply step.

Run from repo root:

    python scripts/run_track_b_tuning.py
    python scripts/run_track_b_tuning.py --probe-only

Wall-time gate (per the Track B spec):
  - Probe times one combination on the largest outer fold.
  - Project per-cell time for full grid (864 combos) and random search
    (80 samples), assuming n_jobs=4 parallelism.
  - If full-grid per-cell <= 15 min AND total <= 8 hr: run full grid.
  - Else if random total <= 8 hr: run random search.
  - Else: abort and print the projection.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from seet.run_track_a import (  # noqa: E402
    INITIAL_TRAIN_END,
    PANEL_NAME,
    PRICE_COL,
    STRESS_DEFS,
    build_folds_with_init,
    compute_stress_labels,
)
from seet.tuning import nested_expanding_cv_tune  # noqa: E402


# Wall-time gate parameters (per user spec, this turn)
GATE_PER_CELL_MIN = 15.0
TOTAL_BUDGET_HOURS = 8.0
RANDOM_N_SAMPLES = 80
N_JOBS = 4

PROBE_COMBO = {
    "max_depth": 3,
    "num_leaves": 15,
    "n_estimators": 200,
    "learning_rate": 0.03,
    "min_child_samples": 80,
    "reg_alpha": 2.0,
    "reg_lambda": 10.0,
}


def load_grid() -> dict:
    grid_path = REPO_ROOT / "experiments" / "track_b_tuning" / "grid.yaml"
    with grid_path.open("r", encoding="utf-8") as f:
        grid = yaml.safe_load(f)
    if not isinstance(grid, dict):
        raise ValueError(f"grid.yaml is not a mapping: {grid!r}")
    return grid


def grid_size(grid: dict) -> int:
    n = 1
    for v in grid.values():
        n *= len(list(v))
    return n


def load_panel_and_features() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    panel_path = REPO_ROOT / "data" / "processed" / f"{PANEL_NAME}.csv"
    feat_path = REPO_ROOT / "data" / "processed" / "features" / f"{PANEL_NAME}_features.csv"
    panel_df = pd.read_csv(panel_path, parse_dates=["Date"])
    features_df = pd.read_csv(feat_path, parse_dates=["Date"])
    if not panel_df["Date"].equals(features_df["Date"]):
        raise ValueError("panel and features Date columns are not aligned")
    feat_cols = [c for c in features_df.columns if c != "Date"]
    return panel_df, features_df, feat_cols


def project_decision(
    probe_seconds: float,
    n_full_grid: int,
    n_random: int,
    n_cells: int,
) -> dict:
    """Decide grid vs random based on per-cell and total projections."""
    full_grid_per_cell_sec = (n_full_grid * probe_seconds) / N_JOBS
    random_per_cell_sec = (n_random * probe_seconds) / N_JOBS
    full_grid_total_sec = full_grid_per_cell_sec * n_cells
    random_total_sec = random_per_cell_sec * n_cells

    full_grid_per_cell_min = full_grid_per_cell_sec / 60.0
    random_per_cell_min = random_per_cell_sec / 60.0
    full_grid_total_hr = full_grid_total_sec / 3600.0
    random_total_hr = random_total_sec / 3600.0

    grid_acceptable = (
        full_grid_per_cell_min <= GATE_PER_CELL_MIN
        and full_grid_total_hr <= TOTAL_BUDGET_HOURS
    )
    random_acceptable = random_total_hr <= TOTAL_BUDGET_HOURS

    if grid_acceptable:
        method = "grid"
        reason = (
            f"grid per-cell {full_grid_per_cell_min:.1f} min <= "
            f"{GATE_PER_CELL_MIN} min AND grid total "
            f"{full_grid_total_hr:.2f} hr <= {TOTAL_BUDGET_HOURS} hr"
        )
    elif random_acceptable:
        method = "random"
        reason = (
            f"grid per-cell {full_grid_per_cell_min:.1f} min > "
            f"{GATE_PER_CELL_MIN} min OR grid total "
            f"{full_grid_total_hr:.2f} hr > {TOTAL_BUDGET_HOURS} hr; "
            f"random total {random_total_hr:.2f} hr <= "
            f"{TOTAL_BUDGET_HOURS} hr"
        )
    else:
        method = "abort"
        reason = (
            f"both grid ({full_grid_total_hr:.2f} hr) and random "
            f"({random_total_hr:.2f} hr) exceed the {TOTAL_BUDGET_HOURS}"
            "-hour total budget"
        )

    return {
        "method": method,
        "reason": reason,
        "full_grid_per_cell_min": full_grid_per_cell_min,
        "random_per_cell_min": random_per_cell_min,
        "full_grid_total_hr": full_grid_total_hr,
        "random_total_hr": random_total_hr,
    }


def run_probe(
    features_df: pd.DataFrame,
    feat_cols: list[str],
    panel_df: pd.DataFrame,
    largest_train_end,
) -> float:
    """Time one combo on the largest outer fold for h10_d05. Returns
    seconds (wall) for one combination across 3 inner folds, n_jobs=1
    (clean serial timing)."""
    sd = next(s for s in STRESS_DEFS if s["name"] == "h10_d05")
    prices = panel_df[PRICE_COL].to_numpy(dtype=float)
    _, labels = compute_stress_labels(prices, sd["h"], sd["d"])
    df = features_df.copy()
    df["label"] = labels

    probe_grid = {k: [v] for k, v in PROBE_COMBO.items()}
    print(
        f"[Probe] timing one combination on h10_d05 outer-train-end "
        f"{pd.Timestamp(largest_train_end).strftime('%Y-%m-%d')} "
        f"(n_jobs=1, 3 inner folds)..."
    )
    result = nested_expanding_cv_tune(
        df, feat_cols, "label",
        outer_train_end=largest_train_end,
        param_grid=probe_grid,
        scoring="pr_auc",
        n_inner_folds=3,
        seed=42,
        n_jobs=1,
        search_method="grid",
    )
    print(
        f"[Probe] one combination wall time = "
        f"{result['wall_time_sec']:.2f} s; "
        f"in-window rows = {result['n_in_window_rows']}; "
        f"inner fold PR-AUCs = {result['inner_fold_scores']}"
    )
    return float(result["wall_time_sec"])


def main(probe_only: bool = False) -> int:
    grid = load_grid()
    n_full_grid = grid_size(grid)
    print(
        f"[Setup] grid loaded from grid.yaml: "
        f"{n_full_grid} combinations across {len(grid)} parameters"
    )

    panel_df, features_df, feat_cols = load_panel_and_features()

    folds = build_folds_with_init(panel_df["Date"], INITIAL_TRAIN_END)
    n_cells = len(folds) * len(STRESS_DEFS)
    print(
        f"[Setup] {len(folds)} outer folds x {len(STRESS_DEFS)} stress_defs "
        f"= {n_cells} (stress_def, outer_fold) cells"
    )

    largest_fold = folds[-1]
    probe_seconds = run_probe(
        features_df, feat_cols, panel_df, largest_fold["train_end"]
    )

    decision = project_decision(probe_seconds, n_full_grid, RANDOM_N_SAMPLES, n_cells)
    print()
    print("[Wall-time projections, n_jobs=4]")
    print(
        f"  full grid: {decision['full_grid_per_cell_min']:.1f} min/cell"
        f"  -> {decision['full_grid_total_hr']:.2f} hr total over {n_cells} cells"
    )
    print(
        f"  random  : {decision['random_per_cell_min']:.1f} min/cell"
        f"  -> {decision['random_total_hr']:.2f} hr total over {n_cells} cells"
    )
    print(f"[Decision] {decision['method']} -- {decision['reason']}")

    if decision["method"] == "abort":
        print(
            "\n[ABORT] both methods exceed the 8-hour budget. "
            "Reduce grid size or increase budget. Stopping per spec."
        )
        return 2

    if probe_only:
        print("\n[Probe-only mode] decision printed; no full tuning. Exiting.")
        return 0

    method = decision["method"]
    out_dir = REPO_ROOT / "experiments" / "track_b_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_rows: list[dict] = []
    inner_score_rows: list[dict] = []

    prices = panel_df[PRICE_COL].to_numpy(dtype=float)

    print()
    print(
        f"[Tuning] starting {n_cells} cells x "
        f"({n_full_grid if method == 'grid' else RANDOM_N_SAMPLES}) combos x 3 inner folds..."
    )
    t_phase = time.time()

    for fold in folds:
        for sd in STRESS_DEFS:
            sd_name = sd["name"]
            t_cell = time.time()
            _, labels = compute_stress_labels(prices, sd["h"], sd["d"])
            df = features_df.copy()
            df["label"] = labels
            try:
                result = nested_expanding_cv_tune(
                    df, feat_cols, "label",
                    outer_train_end=fold["train_end"],
                    param_grid=grid,
                    scoring="pr_auc",
                    n_inner_folds=3,
                    seed=42,
                    n_jobs=N_JOBS,
                    search_method=method,
                    n_random_samples=RANDOM_N_SAMPLES,
                )
            except Exception as e:
                print(
                    f"  [{sd_name} fold {fold['fold_id']}] FAIL: {e}"
                )
                continue
            cell_secs = time.time() - t_cell
            inner_mean = (
                float(np.nanmean(result["inner_fold_scores"]))
                if any(not np.isnan(s) for s in result["inner_fold_scores"])
                else float("nan")
            )
            print(
                f"  [{sd_name} fold {fold['fold_id']:2d}] "
                f"{cell_secs:6.1f}s  inner_pr_auc={inner_mean:.4f}  "
                f"best={result['best_params']}"
            )

            selected_rows.append(
                {
                    "stress_def": sd_name,
                    "outer_fold": int(fold["fold_id"]),
                    "train_end": fold["train_end"].strftime("%Y-%m-%d"),
                    "best_params": json.dumps(
                        result["best_params"], sort_keys=True
                    ),
                    "inner_pr_auc": inner_mean,
                    "search_method": result["search_method"],
                    "n_combinations_tried": int(result["n_combinations"]),
                    "wall_time_sec": float(result["wall_time_sec"]),
                }
            )
            for combo_score in result["all_scores"]:
                fs = combo_score["fold_scores"]
                inner_score_rows.append(
                    {
                        "stress_def": sd_name,
                        "outer_fold": int(fold["fold_id"]),
                        "params": json.dumps(
                            combo_score["params"], sort_keys=True
                        ),
                        "fold0_pr_auc": fs[0] if len(fs) > 0 else float("nan"),
                        "fold1_pr_auc": fs[1] if len(fs) > 1 else float("nan"),
                        "fold2_pr_auc": fs[2] if len(fs) > 2 else float("nan"),
                        "mean_score": combo_score["mean_score"],
                    }
                )

    phase_total_min = (time.time() - t_phase) / 60.0

    sel_df = pd.DataFrame(selected_rows)
    inner_df = pd.DataFrame(inner_score_rows)
    sel_df.to_csv(out_dir / "selected_hp.csv", index=False)
    inner_df.to_csv(out_dir / "inner_scores.csv", index=False)

    print()
    print("===== TRACK B STATUS (Phase B step 7) =====")
    print(f"search method:                 {method}")
    if not sel_df.empty:
        print(f"combinations evaluated/cell:   {sel_df['n_combinations_tried'].iloc[0]}")
    print(f"cells tuned:                   {len(sel_df)} / {n_cells}")
    print(f"total tuning wall time:        {phase_total_min:.1f} min")
    if not sel_df.empty:
        print()
        print("mean inner PR-AUC by stress_def:")
        for sd_name, grp in sel_df.groupby("stress_def"):
            print(
                f"  {sd_name:<10}  mean={grp['inner_pr_auc'].mean():.4f}  "
                f"median={grp['inner_pr_auc'].median():.4f}  "
                f"folds={len(grp)}"
            )
        print()
        print("modal hyperparameters by stress_def "
              "(across folds, ties broken by first occurrence):")
        for sd_name, grp in sel_df.groupby("stress_def"):
            param_dicts = [json.loads(p) for p in grp["best_params"]]
            modal: dict = {}
            for k in sorted(param_dicts[0].keys()):
                series = pd.Series([d[k] for d in param_dicts])
                modal[k] = series.mode().iloc[0]
            print(f"  {sd_name:<10}  {modal}")
    print("===========================================")
    print()
    print(f"selected_hp.csv saved to:  {out_dir / 'selected_hp.csv'}")
    print(f"inner_scores.csv saved to: {out_dir / 'inner_scores.csv'}")
    print()
    print(
        "NEXT: review STATUS, then ask Claude to write apply_track_b.py "
        "(STATUS pause #1 per the Phase B plan)."
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track B nested HP tuning")
    parser.add_argument(
        "--probe-only", action="store_true",
        help="Run only the wall-time probe and decision; do not start tuning.",
    )
    args = parser.parse_args()
    sys.exit(main(probe_only=args.probe_only))
