#!/usr/bin/env python3
"""
Evaluation Script for TaskA experiment Results
============================================

This script evaluates the results from any experiment (e.g. baseline.py) by comparing:
1. Estimated parameters vs target (ground truth) parameters using normalized MSE
2. Generated audio vs target audio using MSE of log magnitude FFT

Usage:
    python eval.py --experiment_folder [experiment_results_folder] --target_folder [target_folder]
    
Arguments:
    --experiment_folder: Path to folder containing experiment results (mandatory)
    --target_folder: Path to folder containing target (ground truth) parameter CSV files (mandatory)
    
Output:
    - Evaluation metrics printed to console
    - evaluation_results.csv with detailed metrics per file
    - parameter_nmse_histogram.png with NMSE distribution
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
import librosa

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ModalPlate.ParamRange import (params as plate_params,
                                   ParamRange,
                                   get_variable_params,
                                   get_fixed_params)

# ===========================
# CONFIGURATION
# ===========================

SAMPLE_RATE = 44100

# Six identifiable parameters S(P) = {mu, D/mu, T0/mu, Ly, op_x, op_y} (Eq. 15).
# Their induced ranges from the raw-parameter box in ParamRange.py follow from
# the monotonicity of each map; we compute them once here so the NMSE
# denominator is meaningful for the three derived quantities.
DERIVED_KEYS = ["mu", "D_mu", "T0_mu", "Ly", "op_x", "op_y"]

# Aliases tolerated in submission CSVs (case-insensitive).
DERIVED_ALIASES = {
    "mu":    ["mu", "rho_h", "rhoh"],
    "D_mu":  ["D_mu", "d_mu", "D/mu", "d/mu", "Dmu"],
    "T0_mu": ["T0_mu", "t0_mu", "T0/mu", "t0/mu", "T0mu"],
    "Ly":    ["Ly", "ly", "L_y"],
    "op_x":  ["op_x", "opx", "xo", "x_o"],
    "op_y":  ["op_y", "opy", "yo", "y_o"],
}


def _derived_bounds():
    rho_lo, rho_hi = plate_params["rho"].low, plate_params["rho"].high
    h_lo, h_hi = plate_params["h"].low, plate_params["h"].high
    E_lo, E_hi = plate_params["E"].low, plate_params["E"].high
    T0_lo, T0_hi = plate_params["T0"].low, plate_params["T0"].high
    nu = plate_params["nu"].low
    return {
        "mu":    (rho_lo * h_lo, rho_hi * h_hi),
        "D_mu":  (E_lo * h_lo * h_lo / (12.0 * (1 - nu ** 2) * rho_hi),
                  E_hi * h_hi * h_hi / (12.0 * (1 - nu ** 2) * rho_lo)),
        "T0_mu": (T0_lo / (rho_hi * h_hi), T0_hi / (rho_lo * h_lo)),
        "Ly":    (plate_params["Ly"].low, plate_params["Ly"].high),
        "op_x":  (plate_params["op_x"].low, plate_params["op_x"].high),
        "op_y":  (plate_params["op_y"].low, plate_params["op_y"].high),
    }


DERIVED_BOUNDS = _derived_bounds()


def _col_lookup(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def raw_to_derived(raw):
    """Map a raw-parameter dict to S(P) = {mu, D/mu, T0/mu, Ly, op_x, op_y}."""
    rho = float(raw["rho"]); h = float(raw["h"])
    E = float(raw["E"]); T0 = float(raw["T0"])
    nu = float(raw.get("nu", plate_params["nu"].low))
    mu = rho * h
    return {
        "mu": mu,
        "D_mu": E * h * h / (12.0 * (1 - nu ** 2) * rho),
        "T0_mu": T0 / mu,
        "Ly": float(raw["Ly"]),
        "op_x": float(raw["op_x"]),
        "op_y": float(raw["op_y"]),
    }


def load_derived_csv(csv_file):
    """Load S(P) from a CSV; accept either 6-col S(P) or raw 7-col format."""
    df = pd.read_csv(csv_file)
    if df.empty:
        raise ValueError(f"{csv_file} is empty")
    row = df.iloc[0].to_dict()
    # Prefer 6-col S(P) if all columns resolve
    cols = {k: _col_lookup(df, DERIVED_ALIASES[k]) for k in DERIVED_KEYS}
    if all(c is not None for c in cols.values()):
        return {k: float(row[cols[k]]) for k in DERIVED_KEYS}
    # Else treat as raw seven-column submission (backwards-compatible with
    # DatasetGen's ground-truth CSVs and legacy submissions).
    for req in ["rho", "h", "E", "T0", "Ly", "op_x", "op_y"]:
        c = _col_lookup(df, [req])
        if c is None:
            raise ValueError(
                f"{csv_file}: missing both S(P) columns and raw '{req}'. "
                f"Found columns: {list(df.columns)}")
    nu_col = _col_lookup(df, ["nu"])
    raw = {k: float(row[_col_lookup(df, [k])])
           for k in ["rho", "h", "E", "T0", "Ly", "op_x", "op_y"]}
    if nu_col is not None:
        raw["nu"] = float(row[nu_col])
    return raw_to_derived(raw)

# ===========================
# UTILITY FUNCTIONS
# ===========================

def load_parameter_csv(csv_file):
    """
    Load parameters from CSV file.
    
    Args:
        csv_file: Path to CSV file
        
    Returns:
        dict: Parameter dictionary
    """
    df = pd.read_csv(csv_file)
    return df.iloc[0].to_dict()


def compute_normalized_parameter_mse(target_params, estimated_params):
    """
    Compute NMSE on the six identifiable parameters S(P)
    = {mu, D/mu, T0/mu, Ly, op_x, op_y} (paper Eq. 17).

    Both inputs must be dicts keyed by the six S(P) names.
    Denominators are the S(P) bounding box induced by ParamRange.py.
    """
    normalized_errors = []
    individual_errors = {}
    for key in DERIVED_KEYS:
        lo, hi = DERIVED_BOUNDS[key]
        span = hi - lo
        if span > 0:
            err = ((target_params[key] - estimated_params[key]) / span) ** 2
        else:
            err = 0.0 if target_params[key] == estimated_params[key] else 1.0
        normalized_errors.append(err)
        individual_errors[key] = err
    return float(np.mean(normalized_errors)), individual_errors


def compute_spectral_mse(target_audio, estimated_audio, sample_rate=SAMPLE_RATE):
    """
    Compute MSE between log magnitude FFT spectra of two audio signals.
    
    Args:
        target_audio: Target audio signal
        estimated_audio: Estimated audio signal
        sample_rate: Sample rate
        
    Returns:
        float: MSE between log magnitude spectra
    """
    # Ensure same length
    min_len = min(len(target_audio), len(estimated_audio))
    target_trimmed = target_audio[:min_len]
    estimated_trimmed = estimated_audio[:min_len]
    
    # Compute FFT
    target_fft = np.fft.fft(target_trimmed)
    estimated_fft = np.fft.fft(estimated_trimmed)
    
    # Compute magnitude spectra
    target_mag = np.abs(target_fft)
    estimated_mag = np.abs(estimated_fft)
    
    # Convert to log magnitude (with small epsilon to avoid log(0))
    epsilon = 1e-10
    target_log_mag = np.log(target_mag + epsilon)
    estimated_log_mag = np.log(estimated_mag + epsilon)
    
    # Compute MSE
    spectral_mse = np.mean((target_log_mag - estimated_log_mag) ** 2)
    
    return spectral_mse


def find_matching_files(experiment_folder, target_folder):
    """
    Find matching files between experiment results and target (ground truth) data.

    Args:
        experiment_folder: Path to experiment results folder
        target_folder: Path to target folder containing the ground truth data

    Returns:
        list: List of tuples (experiment_params_file, experiment_audio_file, target_audio_file, target_params_file, file_id)
    """
    experiment_path = Path(experiment_folder)
    target_path = Path(target_folder)
    
    if not experiment_path.exists():
        raise ValueError(f"Experiment results folder {experiment_folder} does not exist")
    
    if not target_path.exists():
        raise ValueError(f"Target folder {target_folder} does not exist")

    # Find all experiment parameter files
    experiment_param_files = list(experiment_path.glob("best_params_*.csv"))
    
    matches = []
    
    for experiment_param_file in sorted(experiment_param_files):
        # Extract file ID from experiment filename (e.g., best_params_0001.csv -> 0001)
        file_id = experiment_param_file.stem.split('_')[-1]
        
        # Find corresponding files (npz is the official, unnormalized input)
        experiment_audio_file = experiment_path / f"best_audio_{file_id}.npz"
        target_audio_file = target_path / f"random_IR_{file_id}.npz"
        target_params_file = target_path / f"random_IR_params_{file_id}.csv"
        
        # Check if all required files exist
        if (experiment_audio_file.exists() and
            target_audio_file.exists() and
            target_params_file.exists()):
            
            matches.append((
                experiment_param_file,
                experiment_audio_file,
                target_audio_file,
                target_params_file,
                file_id
            ))
            
        else:
            print(f"Warning: Missing files for ID {file_id}, skipping")
            if not experiment_audio_file.exists():
                print(f"  Missing: {experiment_audio_file}")
            if not target_audio_file.exists():
                print(f"  Missing: {target_audio_file}")
            if not target_params_file.exists():
                print(f"  Missing: {target_params_file}")
    
    return matches


# ===========================
# MAIN EVALUATION FUNCTION
# ===========================

def run_evaluation(experiment_folder="experiment_results_taskA", target_folder="random-IR-10-1.0s"):
    """
    Run evaluation of experiment results against ground truth.
    
    Args:
        experiment_folder: Path to experiment results folder
        target_folder: Path to ground truth folder
    """
    print("=" * 60)
    print("TASKΑ experiment EVALUATION")
    print("=" * 60)
    
    # Find matching files
    print(f"Finding matching files...")
    print(f"experiment results: {experiment_folder}")
    print(f"Ground truth: {target_folder}")
    
    matches = find_matching_files(experiment_folder, target_folder)
    
    if len(matches) == 0:
        print("Error: No matching files found between experiment results and ground truth")
        return
    
    print(f"Found {len(matches)} matching files")
    
    # Storage for results
    results = []
    parameter_nmse_values = []
    spectral_mse_values = []
    
    # Process each matching file
    for i, (experiment_params_file, experiment_audio_file, target_audio_file, target_params_file, file_id) in enumerate(matches):
        
        print(f"\n" + "=" * 40)
        print(f"Processing file {i+1}/{len(matches)}: ID {file_id}")
        print("=" * 40)
        
        try:
            # Load parameters in S(P) form (conversion from raw happens inside
            # load_derived_csv for GT CSVs or legacy raw submissions).
            print(f"Loading parameters...")
            estimated_params = load_derived_csv(experiment_params_file)
            target_params = load_derived_csv(target_params_file)

            # Compute parameter NMSE (Eq. 17) on S(P).
            param_nmse, individual_errors = compute_normalized_parameter_mse(
                target_params, estimated_params
            )

            print(f"Parameter NMSE: {param_nmse:.6f}")

            # Load audio files (official input is the unnormalized .npz).
            print(f"Loading audio files...")

            def _load_ir(npz_path):
                with np.load(str(npz_path)) as d:
                    return np.asarray(d["ir"], dtype=np.float64)

            estimated_audio = _load_ir(experiment_audio_file)
            target_audio = _load_ir(target_audio_file)
            
            # Compute spectral MSE
            spectral_mse = compute_spectral_mse(target_audio, estimated_audio)
            
            print(f"Spectral MSE (log magnitude): {spectral_mse:.6f}")
            
            # Store results
            result = {
                'file_id': file_id,
                'parameter_nmse': param_nmse,
                'spectral_mse': spectral_mse,
                'individual_param_errors': individual_errors,
                'experiment_params_file': str(experiment_params_file),
                'target_params_file': str(target_params_file),
                'experiment_audio_file': str(experiment_audio_file),
                'target_audio_file': str(target_audio_file)
            }
            
            results.append(result)
            parameter_nmse_values.append(param_nmse)
            spectral_mse_values.append(spectral_mse)
            
            print(f"✓ Successfully processed file {file_id}")
            
        except Exception as e:
            print(f"✗ Error processing file {file_id}: {e}")
            continue
    
    if len(results) == 0:
        print("Error: No files were successfully processed")
        return
    
    # ===========================
    # COMPUTE SUMMARY STATISTICS
    # ===========================
    
    print(f"\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    # Parameter NMSE statistics
    param_nmse_mean = np.mean(parameter_nmse_values)
    param_nmse_std = np.std(parameter_nmse_values)
    param_nmse_min = np.min(parameter_nmse_values)
    param_nmse_max = np.max(parameter_nmse_values)
    
    # Spectral MSE statistics
    spectral_mse_mean = np.mean(spectral_mse_values)
    spectral_mse_std = np.std(spectral_mse_values)
    spectral_mse_min = np.min(spectral_mse_values)
    spectral_mse_max = np.max(spectral_mse_values)
    
    print(f"Successfully evaluated {len(results)} files")
    print()
    print(f"PARAMETER NORMALIZED MSE (NMSE):")
    print(f"  Mean:  {param_nmse_mean:.6f}")
    print(f"  Std:   {param_nmse_std:.6f}")
    print(f"  Min:   {param_nmse_min:.6f}")
    print(f"  Max:   {param_nmse_max:.6f}")
    print()
    print(f"SPECTRAL MSE (Log Magnitude):")
    print(f"  Mean:  {spectral_mse_mean:.6f}")
    print(f"  Std:   {spectral_mse_std:.6f}")
    print(f"  Min:   {spectral_mse_min:.6f}")
    print(f"  Max:   {spectral_mse_max:.6f}")
    
    # ===========================
    # SAVE DETAILED RESULTS
    # ===========================
    
    print(f"\n" + "=" * 40)
    print("SAVING RESULTS")
    print("=" * 40)
    
    # Create detailed results DataFrame
    detailed_results = []
    for result in results:
        detailed_results.append({
            'file_id': result['file_id'],
            'parameter_nmse': result['parameter_nmse'],
            'spectral_mse': result['spectral_mse']
        })
    
    detailed_df = pd.DataFrame(detailed_results)
    
    # Save detailed results
    experiment_path = Path(experiment_folder)
    detailed_results_file = experiment_path / "evaluation_results.csv"
    detailed_df.to_csv(detailed_results_file, index=False)
    print(f"Saved detailed results to: {detailed_results_file}")
    
    # Save summary statistics
    summary_stats = {
        'metric': ['parameter_nmse_mean', 'parameter_nmse_std', 'parameter_nmse_min', 'parameter_nmse_max',
                  'spectral_mse_mean', 'spectral_mse_std', 'spectral_mse_min', 'spectral_mse_max'],
        'value': [param_nmse_mean, param_nmse_std, param_nmse_min, param_nmse_max,
                 spectral_mse_mean, spectral_mse_std, spectral_mse_min, spectral_mse_max]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_file = experiment_path / "evaluation_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary statistics to: {summary_file}")
    
    # ===========================
    # CREATE HISTOGRAM
    # ===========================
    
    print(f"Creating histogram...")
    
    # Create histogram of parameter NMSE values
    plt.figure(figsize=(10, 6))
    
    # Main histogram
    plt.hist(parameter_nmse_values, bins=min(20, len(parameter_nmse_values)), 
             alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add vertical lines for statistics
    plt.axvline(param_nmse_mean, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {param_nmse_mean:.4f}')
    plt.axvline(param_nmse_mean + param_nmse_std, color='orange', linestyle='--', 
               label=f'Mean + Std: {param_nmse_mean + param_nmse_std:.4f}')
    plt.axvline(param_nmse_mean - param_nmse_std, color='orange', linestyle='--', 
               label=f'Mean - Std: {param_nmse_mean - param_nmse_std:.4f}')
    
    plt.xlabel('Parameter Normalized MSE (NMSE)')
    plt.ylabel('Count')
    plt.title(f'Distribution of Parameter NMSE\n({len(results)} files)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text box with statistics
    textstr = f'Files: {len(results)}\nMean: {param_nmse_mean:.4f}\nStd: {param_nmse_std:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save histogram
    histogram_file = experiment_path / "parameter_nmse_histogram.png"
    plt.savefig(histogram_file, dpi=300, bbox_inches='tight')
    print(f"Saved histogram to: {histogram_file}")
    plt.close()

    # ===========================
    # CREATE BOXPLOT FOR INDIVIDUAL PARAMETER ERRORS
    # ===========================
    
    print(f"Creating boxplot for individual parameter errors...")

    # Plot the per-component errors on the six S(P) parameters.
    variable_param_names = list(DERIVED_KEYS)

    # Collect individual errors for each parameter across all files
    param_errors_dict = {name: [] for name in variable_param_names}
    
    for result in results:
        individual_errors = result['individual_param_errors']
        for param_name in variable_param_names:
            if param_name in individual_errors:
                param_errors_dict[param_name].append(individual_errors[param_name])
    
    # Prepare data for boxplot
    errors_data = [param_errors_dict[name] for name in variable_param_names]
    
    # Create boxplot (handle Matplotlib API differences: tick_labels vs labels)
    plt.figure(figsize=(12, 8))
    try:
        # Newer Matplotlib versions (e.g., 3.12) support tick_labels
        box_plot = plt.boxplot(errors_data, tick_labels=variable_param_names, patch_artist=True)
    except TypeError:
        # Older versions expect 'labels'
        box_plot = plt.boxplot(errors_data, labels=variable_param_names, patch_artist=True)
    
    # Customize boxplot colors
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 
              'lightpink', 'lightgray', 'lightcyan']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.xlabel('Parameters')
    plt.ylabel('Normalized Squared Error')
    plt.title(f'Individual Parameter Errors Distribution\n({len(results)} files)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add mean values as points
    means = [np.mean(param_errors_dict[name]) for name in variable_param_names]
    x_positions = range(1, len(variable_param_names) + 1)
    plt.scatter(x_positions, means, color='red', marker='o', s=50, 
               label='Mean', zorder=10)
    
    plt.legend()
    plt.tight_layout()
    
    # Save boxplot
    boxplot_file = experiment_path / "individual_parameter_errors_boxplot.png"
    plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
    print(f"Saved boxplot to: {boxplot_file}")
    plt.close()

    print(f"\nEvaluation completed successfully!")
    print(f"All results saved to: {experiment_path.absolute()}")
    
    return results


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Evaluate TaskA experiment results against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python eval.py --experiment_folder experiment_results_taskA --target_folder random-IR-10-1.0s
        """
    )
    
    parser.add_argument(
        '--experiment_folder',
        required=True,
        help='Path to experiment results folder (mandatory)'
    )
    
    parser.add_argument(
        '--target_folder',
        required=True,
        help='Path to ground truth folder (mandatory)'
    )
    
    args = parser.parse_args()
    
    try:
        results = run_evaluation(args.experiment_folder, args.target_folder)
        print("\nEvaluation completed successfully!")

    except ValueError as e:
        print(f"\nError: {e}")
        print("\n" + "="*60)
        parser.print_help()
        sys.exit(1)
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
