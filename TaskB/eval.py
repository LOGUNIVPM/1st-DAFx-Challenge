#!/usr/bin/env python3
"""
Evaluation Script for TaskB Experiment Results
===============================================

This script evaluates modal parameter estimation results for Task B of the
1st DAFx Parameter Estimation Challenge, following the metrics defined in
DAFxChallengeDetails.pdf (Section IV-C, Equations 19–24).

Metrics
-------
For each plate, the evaluation computes:

  RE_Ω  – Mean relative error on modal frequencies   (Eq. 19)
  RE_σ  – Mean relative error on decay constants      (Eq. 20)
  RE_b  – Mean relative error on modal gains          (Eq. 21)
  RE0   – Combined relative error  = (RE_Ω + RE_σ + RE_b) / 3   (Eq. 22)
  ΔM    – Mode-count mismatch     = |M - M̃|                     (Eq. 23)
  RE    – Final metric             = RE0 + ΔM                    (Eq. 24)

Unidentified actual modes are assigned an estimated value of zero, yielding
a per-mode relative error of 1.0 for each parameter component.

Identified and actual modes are matched by frequency (closest match),
with identified parameters sorted according to their closest frequency
match in the actual set (as specified in the challenge description).

Usage
-----
    python TaskB/eval.py [options]

Arguments
---------
    --experiment_folder   Path to folder with identified-mode CSVs
                          (default: experiment_results_TaskB)
    --target_folder       Path to folder with ground-truth mode CSVs
                          (default: random-IR-10-1.0s)
    --fmin                Lower frequency bound in Hz for evaluation
                          (default: 0, i.e. no lower filter)
    --fmax                Upper frequency bound in Hz for evaluation
                          (default: inf, i.e. no upper filter)
    --output              Path to save evaluation results CSV
                          (default: <experiment_folder>/evaluation_results_TaskB.csv)

Examples
--------
    python TaskB/eval.py
    python TaskB/eval.py --experiment_folder experiment_results_TaskB --target_folder random-IR-10-1.0s
    python TaskB/eval.py --target_folder random-IR-10-1.0s --fmin 50 --fmax 10000
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Column name mappings (case-insensitive lookup will be used)
# Ground-truth CSV columns (from DatasetGen: random_IR_modes_XXXX.csv)
GT_FREQ_COLS = ["f0", "freq", "frequency", "f0_actual"]
GT_SIGMA_COLS = ["sigma", "sigma_actual", "decay", "damping"]
GT_GAIN_COLS = ["gain", "gain_actual", "amplitude", "b", "modal_gain"]

# Estimated CSV columns (from baseline: random_IR_identifiedModes_XXXX.csv)
EST_FREQ_COLS = ["f0_ident", "f0_identified", "f0_est", "f0_estimated", "freq_ident"]
EST_SIGMA_COLS = ["sigma_ident", "sigma_identified", "sigma_est", "sigma_estimated"]
EST_GAIN_COLS = ["gain_ident", "gain_identified", "gain_est", "gain_estimated"]


# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------------

def _get_col(df, candidates):
    """Return the first column from *candidates* found in *df* (case-insensitive)."""
    df_cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in df_cols_lower:
            return df_cols_lower[cand.lower()]
    return None


def load_ground_truth(csv_path, fmin=0.0, fmax=np.inf):
    """
    Load ground-truth modal parameters from a CSV produced by DatasetGen.

    Returns
    -------
    f0 : np.ndarray   – modal frequencies (Hz)
    sigma : np.ndarray – decay constants (s⁻¹)
    gain : np.ndarray  – modal gains
    """
    df = pd.read_csv(csv_path)

    freq_col = _get_col(df, GT_FREQ_COLS)
    sigma_col = _get_col(df, GT_SIGMA_COLS)
    gain_col = _get_col(df, GT_GAIN_COLS)

    if freq_col is None:
        raise ValueError(f"No frequency column found in {csv_path}. "
                         f"Expected one of {GT_FREQ_COLS}. Got columns: {list(df.columns)}")
    if sigma_col is None:
        raise ValueError(f"No sigma column found in {csv_path}. "
                         f"Expected one of {GT_SIGMA_COLS}. Got columns: {list(df.columns)}")
    if gain_col is None:
        raise ValueError(f"No gain column found in {csv_path}. "
                         f"Expected one of {GT_GAIN_COLS}. Got columns: {list(df.columns)}")

    f0 = df[freq_col].values.astype(float)
    sigma = df[sigma_col].values.astype(float)
    gain = df[gain_col].values.astype(float)

    # Filter to [fmin, fmax]
    mask = (f0 >= fmin) & (f0 <= fmax)
    return f0[mask], sigma[mask], gain[mask]


def load_estimated(csv_path):
    """
    Load estimated (identified) modal parameters from a CSV produced by
    the baseline or a participant's method.

    Returns
    -------
    f0 : np.ndarray   – identified modal frequencies (Hz)
    sigma : np.ndarray – identified decay constants (s⁻¹)
    gain : np.ndarray  – identified modal gains
    """
    df = pd.read_csv(csv_path)

    freq_col = _get_col(df, EST_FREQ_COLS)
    sigma_col = _get_col(df, EST_SIGMA_COLS)
    gain_col = _get_col(df, EST_GAIN_COLS)

    if freq_col is None:
        raise ValueError(f"No frequency column found in {csv_path}. "
                         f"Expected one of {EST_FREQ_COLS}. Got columns: {list(df.columns)}")
    if sigma_col is None:
        raise ValueError(f"No sigma column found in {csv_path}. "
                         f"Expected one of {EST_SIGMA_COLS}. Got columns: {list(df.columns)}")
    if gain_col is None:
        raise ValueError(f"No gain column found in {csv_path}. "
                         f"Expected one of {EST_GAIN_COLS}. Got columns: {list(df.columns)}")

    f0 = df[freq_col].values.astype(float)
    sigma = df[sigma_col].values.astype(float)
    gain = df[gain_col].values.astype(float)

    # Drop rows where all three are NaN (padding from baseline)
    valid = np.isfinite(f0) & np.isfinite(sigma) & np.isfinite(gain)
    return f0[valid], sigma[valid], gain[valid]


# ---------------------------------------------------------------------------
# MODE MATCHING
# ---------------------------------------------------------------------------

def match_modes_by_frequency(f0_actual, f0_ident):
    """
    Match identified modes to actual modes by closest frequency.

    Each identified mode is assigned to its nearest actual mode.  If multiple
    identified modes map to the same actual mode, only the closest one is
    kept; the others are treated as extra (unmatched) identified modes.

    Parameters
    ----------
    f0_actual : np.ndarray, shape (M,)
    f0_ident  : np.ndarray, shape (M_tilde,)

    Returns
    -------
    matches : list of (actual_idx, ident_idx) tuples
        One-to-one mapping between actual and identified modes.
    unmatched_actual : list of int
        Indices into f0_actual that have no corresponding identified mode.
    unmatched_ident : list of int
        Indices into f0_ident that have no corresponding actual mode.
    """
    M = len(f0_actual)
    M_tilde = len(f0_ident)

    if M == 0:
        return [], [], list(range(M_tilde))
    if M_tilde == 0:
        return [], list(range(M)), []

    # For each identified mode, find closest actual mode
    # Use the Hungarian-style greedy: sort identified by frequency,
    # then assign greedily by smallest distance, preferring closer pairs.

    # Build cost matrix (absolute frequency difference)
    cost = np.abs(f0_actual[:, None] - f0_ident[None, :])  # (M, M_tilde)

    matches = []
    used_actual = set()
    used_ident = set()

    # Greedy matching: repeatedly pick the globally smallest cost pair
    flat_order = np.argsort(cost, axis=None)
    for flat_idx in flat_order:
        a_idx = flat_idx // M_tilde
        i_idx = flat_idx % M_tilde
        if a_idx in used_actual or i_idx in used_ident:
            continue
        matches.append((int(a_idx), int(i_idx)))
        used_actual.add(a_idx)
        used_ident.add(i_idx)
        if len(matches) == min(M, M_tilde):
            break

    unmatched_actual = sorted(set(range(M)) - used_actual)
    unmatched_ident = sorted(set(range(M_tilde)) - used_ident)

    return matches, unmatched_actual, unmatched_ident


# ---------------------------------------------------------------------------
# CORE METRICS  (Equations 19–24 from DAFxChallengeDetails.pdf)
# ---------------------------------------------------------------------------

def compute_taskB_metrics(f0_actual, sigma_actual, gain_actual,
                          f0_ident, sigma_ident, gain_ident):
    """
    Compute the Task B evaluation metrics.

    Parameters
    ----------
    f0_actual, sigma_actual, gain_actual : np.ndarray
        Ground-truth modal parameters (M modes).
    f0_ident, sigma_ident, gain_ident : np.ndarray
        Identified modal parameters (M̃ modes).

    Returns
    -------
    metrics : dict
        RE_omega, RE_sigma, RE_gain, RE0, delta_M, RE, plus per-mode details.
    """
    M = len(f0_actual)
    M_tilde = len(f0_ident)
    delta_M = abs(M - M_tilde)

    # Edge case: no actual modes
    if M == 0:
        return dict(
            RE_omega=0.0, RE_sigma=0.0, RE_gain=0.0,
            RE0=0.0, delta_M=delta_M, RE=float(delta_M),
            M_actual=M, M_identified=M_tilde,
            n_matched=0, n_unmatched_actual=0, n_unmatched_ident=M_tilde,
            per_mode=[]
        )

    # Match modes by frequency
    matches, unmatched_actual, unmatched_ident = match_modes_by_frequency(
        f0_actual, f0_ident
    )

    # ----- Compute relative errors (Eqs. 19–21) -----
    # For matched modes: normal relative error
    # For unmatched actual modes: estimated = 0  →  |0 - ref| / |ref| = 1.0
    #   (as stated in the paper: "assigning a value of zero to each
    #    unidentified modal parameter")

    re_omega_terms = np.zeros(M)
    re_sigma_terms = np.zeros(M)
    re_gain_terms = np.zeros(M)
    per_mode_details = []

    # Process matched modes
    matched_actual_set = set()
    for a_idx, i_idx in matches:
        matched_actual_set.add(a_idx)

        f0_ref = f0_actual[a_idx]
        sigma_ref = sigma_actual[a_idx]
        gain_ref = gain_actual[a_idx]

        f0_est = f0_ident[i_idx]
        sigma_est = sigma_ident[i_idx]
        gain_est = gain_ident[i_idx]

        # Relative errors (guard against zero reference)
        re_f = abs(f0_est - f0_ref) / abs(f0_ref) if abs(f0_ref) > 1e-30 else 0.0
        re_s = abs(sigma_est - sigma_ref) / abs(sigma_ref) if abs(sigma_ref) > 1e-30 else 0.0
        re_g = abs(gain_est - gain_ref) / abs(gain_ref) if abs(gain_ref) > 1e-30 else 0.0

        re_omega_terms[a_idx] = re_f
        re_sigma_terms[a_idx] = re_s
        re_gain_terms[a_idx] = re_g

        per_mode_details.append(dict(
            actual_idx=a_idx, ident_idx=i_idx, matched=True,
            f0_actual=f0_ref, f0_ident=f0_est,
            sigma_actual=sigma_ref, sigma_ident=sigma_est,
            gain_actual=gain_ref, gain_ident=gain_est,
            re_omega=re_f, re_sigma=re_s, re_gain=re_g
        ))

    # Process unmatched actual modes (estimated = 0 → relative error = 1.0)
    for a_idx in unmatched_actual:
        re_omega_terms[a_idx] = 1.0
        re_sigma_terms[a_idx] = 1.0
        re_gain_terms[a_idx] = 1.0

        per_mode_details.append(dict(
            actual_idx=a_idx, ident_idx=None, matched=False,
            f0_actual=f0_actual[a_idx], f0_ident=np.nan,
            sigma_actual=sigma_actual[a_idx], sigma_ident=np.nan,
            gain_actual=gain_actual[a_idx], gain_ident=np.nan,
            re_omega=1.0, re_sigma=1.0, re_gain=1.0
        ))

    # Mean relative errors (Eqs. 19–21)
    RE_omega = np.mean(re_omega_terms)
    RE_sigma = np.mean(re_sigma_terms)
    RE_gain = np.mean(re_gain_terms)

    # Combined relative error (Eq. 22)
    RE0 = (RE_omega + RE_sigma + RE_gain) / 3.0

    # Final metric (Eq. 24)
    RE = RE0 + delta_M

    return dict(
        RE_omega=RE_omega,
        RE_sigma=RE_sigma,
        RE_gain=RE_gain,
        RE0=RE0,
        delta_M=delta_M,
        RE=RE,
        M_actual=M,
        M_identified=M_tilde,
        n_matched=len(matches),
        n_unmatched_actual=len(unmatched_actual),
        n_unmatched_ident=len(unmatched_ident),
        per_mode=per_mode_details
    )


# ---------------------------------------------------------------------------
# FILE DISCOVERY
# ---------------------------------------------------------------------------

def find_matching_files(experiment_folder, target_folder):
    """
    Find pairs of (estimated CSV, ground-truth CSV) that share the same
    file ID.

    The baseline produces files named ``random_IR_identifiedModes_XXXX.csv``
    in *experiment_folder*, while DatasetGen produces ground-truth files
    named ``random_IR_modes_XXXX.csv`` in *target_folder*.

    Returns
    -------
    list of (estimated_csv_path, ground_truth_csv_path, file_id)
    """
    exp_path = Path(experiment_folder)
    tgt_path = Path(target_folder)

    if not exp_path.exists():
        raise ValueError(f"Experiment results folder does not exist: {experiment_folder}")
    if not tgt_path.exists():
        raise ValueError(f"Target (ground-truth) folder does not exist: {target_folder}")

    # Discover estimated files
    est_files = sorted(exp_path.glob("random_IR_identifiedModes_*.csv"))
    if not est_files:
        # Also try a more generic pattern
        est_files = sorted(exp_path.glob("*identifiedModes*.csv"))
    if not est_files:
        # Fallback: any CSV that is not an index
        est_files = sorted(
            f for f in exp_path.glob("*.csv")
            if not f.name.startswith("_") and "evaluation" not in f.name.lower()
        )

    # Build a map from file_id → estimated CSV
    est_map = {}
    for f in est_files:
        # Try to extract numeric ID from filename
        parts = f.stem.split("_")
        # Take the last purely-numeric segment
        for part in reversed(parts):
            if part.isdigit():
                est_map[part] = f
                break

    # Find matching ground-truth files
    matches = []
    for file_id, est_csv in sorted(est_map.items()):
        # Ground-truth modes file
        gt_csv = tgt_path / f"random_IR_modes_{file_id}.csv"
        if not gt_csv.exists():
            # Try alternative naming
            gt_csv = tgt_path / f"random_IR_modes_{file_id}.csv"
        if gt_csv.exists():
            matches.append((est_csv, gt_csv, file_id))
        else:
            print(f"  Warning: No ground-truth file found for ID {file_id} "
                  f"(expected {gt_csv}), skipping.")

    return matches


# ---------------------------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------------------------

def run_evaluation(experiment_folder, target_folder, fmin=0.0, fmax=np.inf,
                   output_path=None):
    """
    Run Task B evaluation over all matching files.

    Parameters
    ----------
    experiment_folder : str
        Path to the folder containing identified-mode CSVs.
    target_folder : str
        Path to the folder containing ground-truth mode CSVs.
    fmin, fmax : float
        Frequency band for evaluation (Hz).
    output_path : str or None
        Where to save the detailed results CSV.
    """
    print("=" * 60)
    print("TASK B — MODAL PARAMETER ESTIMATION EVALUATION")
    print("=" * 60)
    print(f"  Experiment results : {experiment_folder}")
    print(f"  Ground truth       : {target_folder}")
    if fmin > 0 or np.isfinite(fmax):
        print(f"  Frequency band     : [{fmin:.1f}, {fmax:.1f}] Hz")
    print()

    # Discover file pairs
    matches = find_matching_files(experiment_folder, target_folder)
    if not matches:
        print("ERROR: No matching file pairs found.")
        print("  Estimated files should be named: random_IR_identifiedModes_XXXX.csv")
        print("  Ground-truth files should be named: random_IR_modes_XXXX.csv")
        return None

    print(f"Found {len(matches)} matching file pair(s).\n")

    # ---- Per-file evaluation ----
    all_results = []

    for est_csv, gt_csv, file_id in matches:
        print(f"--- File ID {file_id} ---")
        try:
            f0_act, sigma_act, gain_act = load_ground_truth(gt_csv, fmin, fmax)
            f0_est, sigma_est, gain_est = load_estimated(est_csv)

            # Also filter estimated modes to [fmin, fmax] for fair comparison
            if fmin > 0 or np.isfinite(fmax):
                mask_est = (f0_est >= fmin) & (f0_est <= fmax)
                f0_est = f0_est[mask_est]
                sigma_est = sigma_est[mask_est]
                gain_est = gain_est[mask_est]

            metrics = compute_taskB_metrics(
                f0_act, sigma_act, gain_act,
                f0_est, sigma_est, gain_est
            )

            print(f"  M_actual={metrics['M_actual']:4d}  "
                  f"M_identified={metrics['M_identified']:4d}  "
                  f"ΔM={metrics['delta_M']:4d}  "
                  f"matched={metrics['n_matched']:4d}")
            print(f"  RE_Ω={metrics['RE_omega']:.6f}  "
                  f"RE_σ={metrics['RE_sigma']:.6f}  "
                  f"RE_b={metrics['RE_gain']:.6f}")
            print(f"  RE0={metrics['RE0']:.6f}  "
                  f"ΔM={metrics['delta_M']}  "
                  f"RE={metrics['RE']:.6f}")

            all_results.append(dict(
                file_id=file_id,
                M_actual=metrics["M_actual"],
                M_identified=metrics["M_identified"],
                n_matched=metrics["n_matched"],
                delta_M=metrics["delta_M"],
                RE_omega=metrics["RE_omega"],
                RE_sigma=metrics["RE_sigma"],
                RE_gain=metrics["RE_gain"],
                RE0=metrics["RE0"],
                RE=metrics["RE"],
                est_csv=str(est_csv),
                gt_csv=str(gt_csv),
            ))

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("\nNo files were successfully evaluated.")
        return None

    # ---- Summary statistics ----
    df = pd.DataFrame(all_results)

    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Files evaluated : {len(df)}")
    print()

    for col, label in [("RE_omega", "RE_Ω (frequency)"),
                        ("RE_sigma", "RE_σ (decay)"),
                        ("RE_gain",  "RE_b (gain)"),
                        ("RE0",      "RE0  (combined)"),
                        ("delta_M",  "ΔM   (mode count)"),
                        ("RE",       "RE   (final)")]:
        vals = df[col].values
        print(f"  {label:25s}  mean={np.mean(vals):.6f}  "
              f"std={np.std(vals):.6f}  "
              f"min={np.min(vals):.6f}  "
              f"max={np.max(vals):.6f}")

    print()
    print(f"  Total modes (actual)     : {int(df['M_actual'].sum())}")
    print(f"  Total modes (identified) : {int(df['M_identified'].sum())}")
    print(f"  Total matched            : {int(df['n_matched'].sum())}")

    # ---- Save results ----
    exp_path = Path(experiment_folder)
    if output_path is None:
        output_path = exp_path / "evaluation_results_TaskB.csv"
    else:
        output_path = Path(output_path)

    # Detailed per-file results
    df.to_csv(output_path, index=False)
    print(f"\nSaved per-file results  → {output_path}")

    # Summary CSV
    summary_rows = []
    for col in ["RE_omega", "RE_sigma", "RE_gain", "RE0", "delta_M", "RE"]:
        vals = df[col].values
        summary_rows.append(dict(
            metric=col,
            mean=np.mean(vals), std=np.std(vals),
            min=np.min(vals), max=np.max(vals),
            median=np.median(vals)
        ))
    summary_df = pd.DataFrame(summary_rows)
    summary_path = exp_path / "evaluation_summary_TaskB.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary statistics → {summary_path}")

    # ---- Plots ----
    _make_plots(df, exp_path)

    print(f"\nEvaluation completed successfully!")
    return df


def _make_plots(df, output_dir):
    """Generate evaluation plots and save to *output_dir*."""

    # 1) Histogram of RE (final metric)
    fig, ax = plt.subplots(figsize=(10, 5))
    re_vals = df["RE"].values
    ax.hist(re_vals, bins=min(20, len(re_vals)), alpha=0.7,
            color="steelblue", edgecolor="black")
    ax.axvline(np.mean(re_vals), color="red", ls="--", lw=2,
               label=f"Mean: {np.mean(re_vals):.4f}")
    ax.set_xlabel("RE (final metric)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of RE — Task B  ({len(df)} files)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "taskB_RE_histogram.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved histogram         → {path}")

    # 2) Grouped bar chart of RE_Ω, RE_σ, RE_b per file
    if len(df) <= 50:
        fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.6), 5))
        x = np.arange(len(df))
        w = 0.25
        ax.bar(x - w, df["RE_omega"], w, label="RE_Ω (freq)", color="cornflowerblue")
        ax.bar(x,     df["RE_sigma"], w, label="RE_σ (decay)", color="sandybrown")
        ax.bar(x + w, df["RE_gain"],  w, label="RE_b (gain)",  color="mediumseagreen")
        ax.set_xticks(x)
        ax.set_xticklabels(df["file_id"], rotation=45, ha="right")
        ax.set_ylabel("Relative Error")
        ax.set_title("Per-Component Relative Errors — Task B")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        path = output_dir / "taskB_per_component_errors.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Saved component chart   → {path}")

    # 3) Mode count comparison
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.5), 5))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["M_actual"],     0.4, label="Actual modes (M)",  color="steelblue")
    ax.bar(x + 0.2, df["M_identified"], 0.4, label="Identified modes (M̃)", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(df["file_id"], rotation=45, ha="right")
    ax.set_ylabel("Number of modes")
    ax.set_title("Mode Count: Actual vs Identified — Task B")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = output_dir / "taskB_mode_counts.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved mode-count chart  → {path}")


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Task B (modal parameter estimation) results "
                    "against ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python TaskB/eval.py
  python TaskB/eval.py --experiment_folder experiment_results_TaskB \\
                       --target_folder random-IR-10-1.0s
  python TaskB/eval.py --fmin 50 --fmax 10000
        """
    )

    parser.add_argument(
        "--experiment_folder",
        default="experiment_results_TaskB",
        help="Path to folder containing identified-mode CSVs "
             "(default: experiment_results_TaskB)"
    )
    parser.add_argument(
        "--target_folder",
        default="random-IR-10-1.0s",
        help="Path to folder containing ground-truth mode CSVs "
             "(default: random-IR-10-1.0s)"
    )
    parser.add_argument(
        "--fmin", type=float, default=0.0,
        help="Lower frequency bound for evaluation in Hz (default: 0)"
    )
    parser.add_argument(
        "--fmax", type=float, default=np.inf,
        help="Upper frequency bound for evaluation in Hz (default: inf)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save the detailed evaluation CSV "
             "(default: <experiment_folder>/evaluation_results_TaskB.csv)"
    )

    args = parser.parse_args()

    try:
        run_evaluation(
            experiment_folder=args.experiment_folder,
            target_folder=args.target_folder,
            fmin=args.fmin,
            fmax=args.fmax,
            output_path=args.output,
        )
    except ValueError as e:
        print(f"\nError: {e}")
        parser.print_help()
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
