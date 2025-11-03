#!/usr/bin/env python3
"""
Evaluation Script for TaskA experiment Results
============================================

This script evaluates the results from any experiment (e.g. baseline.py) by comparing:
1. Estimated parameters vs target (ground truth) parameters using normalized MSE
2. Generated audio vs target audio using MSE of log magnitude FFT

Usage:
    python eval.py [experiment_results_folder] [target_folder]
    
Arguments:
    experiment_results_folder: Path to folder containing experiment results (default: experiment_results)
    target_folder: Path to folder containing target (ground truth) parameter CSV files (default: random-IR-10-1.0s)
    
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
    Compute normalized MSE between parameter sets using parameter ranges.
    
    Args:
        target_params: Dictionary of target parameters
        estimated_params: Dictionary of estimated parameters
        
    Returns:
        tuple: (normalized_mse, individual_errors)
    """
    # Get common parameter keys
    common_keys = set(target_params.keys()) & set(estimated_params.keys())
    
    if len(common_keys) == 0:
        raise ValueError("No common parameters found between target and estimated")
    
    normalized_errors = []
    individual_errors = {}
    
    for key in common_keys:
        target_val = target_params[key]
        estimated_val = estimated_params[key]
        
        if key in plate_params:
            param_range = plate_params[key]
            range_span = param_range.high - param_range.low
            
            if range_span > 0:
                # Normalize error by parameter range
                normalized_error = ((target_val - estimated_val) / range_span) ** 2
            else:
                # Fixed parameter - perfect match expected
                normalized_error = 0.0 if target_val == estimated_val else 1.0
        else:
            # Fallback: use relative error if possible
            if abs(target_val) > 1e-10:
                normalized_error = ((target_val - estimated_val) / target_val) ** 2
            else:
                normalized_error = (target_val - estimated_val) ** 2
        
        normalized_errors.append(normalized_error)
        individual_errors[key] = normalized_error
    
    # Compute mean normalized MSE
    normalized_mse = np.mean(normalized_errors)
    
    return normalized_mse, individual_errors


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
        
        # Find corresponding files
        experiment_audio_file = experiment_path / f"best_audio_{file_id}.wav"
        target_audio_file = target_path / f"random_IR_{file_id}.wav"  # Target audio from ground truth folder
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

def run_evaluation(experiment_folder="experiment_results", target_folder="random-IR-10-1.0s"):
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
            # Load parameters
            print(f"Loading parameters...")
            estimated_params = load_parameter_csv(experiment_params_file)
            target_params = load_parameter_csv(target_params_file)
            
            # Compute parameter NMSE
            param_nmse, individual_errors = compute_normalized_parameter_mse(
                target_params, estimated_params
            )
            
            print(f"Parameter NMSE: {param_nmse:.6f}")
            
            # Load audio files
            print(f"Loading audio files...")
            estimated_audio, _ = librosa.load(experiment_audio_file, sr=SAMPLE_RATE)
            target_audio, _ = librosa.load(target_audio_file, sr=SAMPLE_RATE)
            
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
    
    plt.show()
    
    # ===========================
    # CREATE BOXPLOT FOR INDIVIDUAL PARAMETER ERRORS
    # ===========================
    
    print(f"Creating boxplot for individual parameter errors...")
    
    # Get variable parameter names
    variable_params = get_variable_params()
    variable_param_names = list(variable_params.keys())
    
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
    
    plt.show()
    
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
    python eval.py                                          # Use default folders
    python eval.py experiment_results random-IR-10-1.0s  # Specify folders
        """
    )
    
    parser.add_argument(
        'experiment_folder',
        nargs='?',
        default='experiment_results',
        help='Path to experiment results folder (default: experiment_results)'
    )
    
    parser.add_argument(
        'target_folder', 
        nargs='?',
        default='random-IR-10-1.0s',
        help='Path to ground truth folder (default: random-IR-10-1.0s)'
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