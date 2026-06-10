#!/usr/bin/env python3
"""
Batch Evaluation Script for the 1st DAFx Challenge
===================================================

Iterates over all participant folders inside a given submissions directory and
runs the appropriate evaluation script for each task found.

Expected folder layout
----------------------
<submissions_dir>/
    <team_name>/
        EVALUATE/
            TaskA/               # normal: CSV files directly here
                random_IR_params_0001.csv
                ...
            TaskB/               # exceptional: sub-folders instead of CSVs
                run1/
                    random_IR_modes_0001.csv
                    ...
                run2/
                    ...

For the normal case the relevant eval script is called once with the TaskA or
TaskB directory as --experiment_folder.

For the exceptional case (no CSV files directly in the task folder, only
sub-folders) the eval script is called once per sub-folder.

Usage
-----
    python batchEval.py <submissions_dir> [--dataset_folder DATASET_FOLDER]
                        [--output_summary OUTPUT_SUMMARY]
                        [--onlytask {TaskA,TaskB}]

Arguments
---------
    submissions_dir         Path to the folder containing one sub-folder per team.
    --dataset_folder        Path to the dataset folder holding ground-truth CSV
                            files (default: 2026-DATASET next to this script).
    --output_summary        Path to write a CSV summary of all evaluation runs
                            (default: <submissions_dir>/batch_eval_summary.csv).
    --onlytask              Restrict evaluation to a single task (TaskA or TaskB).
                            When set, the other task is silently skipped.
"""

import sys
import os
import argparse
import subprocess
import csv
from pathlib import Path

try:
    import pandas as pd
    _HAVE_PANDAS = True
except ImportError:
    _HAVE_PANDAS = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
TASK_A_EVAL = REPO_ROOT / "TaskA" / "eval.py"
TASK_B_EVAL = REPO_ROOT / "TaskB" / "eval.py"


def _contains_csv(folder: Path) -> bool:
    """Return True if *folder* has at least one CSV file directly inside it."""
    return any(folder.glob("*.csv"))


def _get_subfolders(folder: Path) -> list[Path]:
    """Return immediate sub-directories of *folder* (sorted)."""
    return sorted(p for p in folder.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# Metric columns written to the batch summary CSV
# ---------------------------------------------------------------------------

# TaskA: evaluation_summary.csv has columns (metric, value) with these metric names.
_TASK_A_SUMMARY_FILE = "evaluation_summary.csv"
_TASK_A_METRICS = [
    "parameter_nmse_mean",
    "parameter_nmse_std",
    "parameter_nmse_min",
    "parameter_nmse_max",
    "spectral_mse_mean",
    "spectral_mse_std",
    "spectral_mse_min",
    "spectral_mse_max",
]

# TaskB: evaluation_summary_TaskB.csv has columns (metric, mean, std, min, max, median).
_TASK_B_SUMMARY_FILE = "evaluation_summary_TaskB.csv"
_TASK_B_METRICS = ["RE_omega", "RE_sigma", "RE_gain", "RE0", "delta_M", "RE"]
_TASK_B_STATS   = ["mean", "std", "min", "max", "median"]

# All possible metric columns that may appear in the summary CSV.
_ALL_METRIC_COLS = (
    [f"A_{m}" for m in _TASK_A_METRICS]
    + [f"B_{m}_{s}" for m in _TASK_B_METRICS for s in _TASK_B_STATS]
)


def _read_taskA_metrics(experiment_folder: Path) -> dict:
    """Read TaskA evaluation_summary.csv and return a flat dict of metric columns."""
    summary = experiment_folder / _TASK_A_SUMMARY_FILE
    cols = {f"A_{m}": "" for m in _TASK_A_METRICS}
    if not summary.is_file():
        return cols
    try:
        with open(summary, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"A_{row['metric']}"
                if key in cols:
                    cols[key] = row.get("value", "")
    except Exception as exc:
        print(f"      [warn] Could not read TaskA summary: {exc}")
    return cols


def _read_taskB_metrics(experiment_folder: Path) -> dict:
    """Read TaskB evaluation_summary_TaskB.csv and return a flat dict of metric columns."""
    summary = experiment_folder / _TASK_B_SUMMARY_FILE
    cols = {f"B_{m}_{s}": "" for m in _TASK_B_METRICS for s in _TASK_B_STATS}
    if not summary.is_file():
        return cols
    try:
        with open(summary, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric = row.get("metric", "")
                for stat in _TASK_B_STATS:
                    key = f"B_{metric}_{stat}"
                    if key in cols:
                        cols[key] = row.get(stat, "")
    except Exception as exc:
        print(f"      [warn] Could not read TaskB summary: {exc}")
    return cols


def run_eval(eval_script: Path, experiment_folder: Path, dataset_folder: Path) -> dict:
    """
    Run an eval.py script and return a result dict with keys:
        returncode, stdout, stderr, plus metric columns for the task.
    """
    cmd = [
        sys.executable,
        str(eval_script),
        "--experiment_folder", str(experiment_folder),
        "--target_folder", str(dataset_folder),
    ]
    print(f"    Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"    OK (exit 0)")
    else:
        print(f"    FAILED (exit {result.returncode})")
    if result.stdout:
        for line in result.stdout.splitlines():
            print(f"      [stdout] {line}")
    if result.stderr:
        for line in result.stderr.splitlines():
            print(f"      [stderr] {line}")
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def evaluate_task_folder(task_name: str, task_folder: Path, dataset_folder: Path) -> list[dict]:
    """
    Evaluate a single TaskA or TaskB folder for one team.

    Detects whether task_folder contains CSV files directly (normal case) or
    only sub-folders (exceptional case) and dispatches accordingly.

    Returns a list of result dicts (one per eval call).
    """
    eval_script = TASK_A_EVAL if task_name == "TaskA" else TASK_B_EVAL
    results = []

    def _enrich_with_metrics(res: dict, exp_folder: Path) -> dict:
        """Append task-specific metric columns to *res* after a successful run."""
        if res["returncode"] == 0:
            if task_name == "TaskA":
                res.update(_read_taskA_metrics(exp_folder))
            else:
                res.update(_read_taskB_metrics(exp_folder))
        return res

    if _contains_csv(task_folder):
        # Normal case: CSVs live directly inside the task folder.
        print(f"  [{task_name}] Normal case – evaluating {task_folder}")
        res = run_eval(eval_script, task_folder, dataset_folder)
        res["task"] = task_name
        res["experiment_folder"] = str(task_folder)
        res["sub_run"] = ""
        _enrich_with_metrics(res, task_folder)
        results.append(res)
    else:
        subfolders = _get_subfolders(task_folder)
        if not subfolders:
            print(f"  [{task_name}] WARNING: no CSV files and no sub-folders in {task_folder} – skipping")
            return results

        # Exceptional case: sub-folders instead of CSVs.
        print(f"  [{task_name}] Exceptional case – found {len(subfolders)} sub-folder(s) in {task_folder}")
        for sub in subfolders:
            print(f"  [{task_name}] Sub-run: {sub.name}")
            res = run_eval(eval_script, sub, dataset_folder)
            res["task"] = task_name
            res["experiment_folder"] = str(sub)
            res["sub_run"] = sub.name
            _enrich_with_metrics(res, sub)
            results.append(res)

    return results


def evaluate_team(team_folder: Path, dataset_folder: Path,
                  only_task: str | None = None) -> list[dict]:
    """
    Evaluate all tasks for a single team.

    Looks for an EVALUATE sub-folder containing TaskA and/or TaskB.
    If *only_task* is set to "TaskA" or "TaskB", the other task is skipped.
    Returns a list of result dicts.
    """
    evaluate_dir = team_folder / "EVALUATE"
    if not evaluate_dir.is_dir():
        print(f"  WARNING: no EVALUATE folder found in {team_folder} – skipping")
        return []

    tasks_to_run = ["TaskA", "TaskB"] if only_task is None else [only_task]

    results = []
    for task_name in tasks_to_run:
        task_folder = evaluate_dir / task_name
        if not task_folder.is_dir():
            print(f"  [{task_name}] not found – skipping")
            continue
        task_results = evaluate_task_folder(task_name, task_folder, dataset_folder)
        for r in task_results:
            r["team"] = team_folder.name
        results.extend(task_results)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation of all participant submissions for the 1st DAFx Challenge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python batchEval.py SUBMISSIONS --dataset_folder 2026-DATASET
    python batchEval.py SUBMISSIONS --onlytask TaskA
    python batchEval.py SUBMISSIONS --onlytask TaskB --dataset_folder 2026-DATASET
        """
    )
    parser.add_argument(
        "submissions_dir",
        help="Path to the folder containing one sub-folder per participating team."
    )
    parser.add_argument(
        "--dataset_folder",
        default=str(REPO_ROOT / "2026-DATASET"),
        help="Path to the ground-truth dataset folder (default: 2026-DATASET next to this script)."
    )
    parser.add_argument(
        "--output_summary",
        default=None,
        help="Path for the CSV summary file (default: <submissions_dir>/batch_eval_summary.csv)."
    )
    parser.add_argument(
        "--onlytask",
        choices=["TaskA", "TaskB"],
        default=None,
        help="Restrict evaluation to a single task. The other task is skipped entirely."
    )
    args = parser.parse_args()

    submissions_dir = Path(args.submissions_dir).resolve()
    dataset_folder = Path(args.dataset_folder).resolve()

    if not submissions_dir.is_dir():
        print(f"ERROR: submissions directory not found: {submissions_dir}")
        sys.exit(1)

    if not dataset_folder.is_dir():
        print(f"ERROR: dataset folder not found: {dataset_folder}")
        sys.exit(1)

    if not TASK_A_EVAL.is_file():
        print(f"ERROR: TaskA eval script not found: {TASK_A_EVAL}")
        sys.exit(1)

    if not TASK_B_EVAL.is_file():
        print(f"ERROR: TaskB eval script not found: {TASK_B_EVAL}")
        sys.exit(1)

    output_summary = Path(args.output_summary) if args.output_summary else submissions_dir / "batch_eval_summary.csv"

    # Collect team folders (skip hidden entries and files).
    team_folders = sorted(
        p for p in submissions_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )

    if not team_folders:
        print(f"No team folders found in {submissions_dir}")
        sys.exit(0)

    print(f"Found {len(team_folders)} team folder(s) in {submissions_dir}")
    print(f"Dataset folder : {dataset_folder}")
    print(f"Summary output : {output_summary}")
    if args.onlytask:
        print(f"Only task      : {args.onlytask}")
    print("=" * 70)

    all_results = []

    for team_folder in team_folders:
        print(f"\nTeam: {team_folder.name}")
        print("-" * 60)
        team_results = evaluate_team(team_folder, dataset_folder, only_task=args.onlytask)
        all_results.extend(team_results)

    # Write CSV summary.
    print("\n" + "=" * 70)
    print(f"Writing summary to {output_summary} ...")
    # Build fieldnames: fixed columns first, then all possible metric columns.
    fixed_fields = ["team", "task", "sub_run", "experiment_folder", "returncode"]
    metric_fields = _ALL_METRIC_COLS
    trailing_fields = ["stdout", "stderr"]
    fieldnames = fixed_fields + metric_fields + trailing_fields
    with open(output_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # Print final summary table.
    passed = sum(1 for r in all_results if r["returncode"] == 0)
    failed = len(all_results) - passed
    print(f"\nBatch evaluation complete: {passed} passed, {failed} failed out of {len(all_results)} run(s).")

    if failed:
        print("\nFailed runs:")
        for r in all_results:
            if r["returncode"] != 0:
                sub = f" / {r['sub_run']}" if r["sub_run"] else ""
                print(f"  {r['team']} – {r['task']}{sub}  (exit {r['returncode']})")
        sys.exit(1)


if __name__ == "__main__":
    main()
