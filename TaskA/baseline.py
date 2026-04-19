#!/usr/bin/env python3
"""
Baseline PSO Script for Task A
===============================

This script uses PSO to find plate parameters for each impulse response in the target folder.

Usage:
    python baseline.py [target_folder]
    
Arguments:
    target_folder: Path to folder containing target IR files (generate it using ModalPlate/DatasetGen.py)
    
Output:
    Creates experiment_results_taskA/ folder containing:
    - best parameters CSV for each target
    - synthesized audio using best parameters
    - experiment logs
    
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import librosa
import soundfile as sf
import platform

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from pso import PSO
from ModalPlate.ModalPlate import ModalPlate
from ModalPlate.ParamRange import (params as plate_params, 
                                   get_variable_params, 
                                   get_fixed_params,
                                   variable_params_to_full_params,
                                   full_params_to_variable_params)
from mss_loss import multi_scale_spectral_loss
import logger
import multiprocessing

# ===========================
# CONFIGURATION
# ===========================

# STFT configurations for multi-scale spectral loss
STFT_CONFIGS = [(512, 128), (2048, 512), (8192, 2048)]

# PSO parameters
PARTICLES = 10  # Number of particles
ITERATIONS = 5  # Number of iterations  
PSO_W = 0.2      # Inertia weight
PSO_C1 = 2       # Cognitive coefficient
PSO_C2 = 1       # Social coefficient

# Audio parameters
SAMPLE_RATE = 44100
DURATION = 3.0   # Duration for synthesis (will be adjusted based on target)

# Parallel processing
MAX_WORKERS = None  # Auto-detect

# Output directory
OUTPUT_DIR = "experiment_results_taskA"

# ===========================
# UTILITY FUNCTIONS
# ===========================

def load_target_files(target_folder):
    """
    Load target IR files (random_IR_XXXX.npz) from the target folder.

    The .npz is the official input for Task A (unnormalized displacement
    plus metadata). The sibling .wav is peak-normalized and loses the
    absolute amplitude — which carries mu = rho * h — so it is not
    accepted here.

    Args:
        target_folder: Path to folder containing random_IR_*.npz files

    Returns:
        list: List of tuples (audio, filename)
    """
    target_path = Path(target_folder)

    if not target_path.exists():
        raise ValueError(f"Target folder {target_folder} does not exist")

    # Official input: the scientific .npz (unnormalized IR + metadata).
    npz_files = sorted(target_path.glob("random_IR_[0-9]*.npz"))

    if len(npz_files) == 0:
        raise ValueError(
            f"No random_IR_*.npz files found in {target_folder}. "
            f"Task A requires the unnormalized .npz input; the peak-normalized "
            f".wav cannot be used because it discards the mu = rho*h amplitude "
            f"scale."
        )

    targets = []

    print(f"Loading target files from {target_folder}")

    for npz_file in npz_files:
        try:
            with np.load(str(npz_file)) as data:
                audio = np.asarray(data["ir"], dtype=np.float64)
            targets.append((audio, npz_file.name))
            print(f"  Loaded {npz_file.name}: {len(audio)} samples")
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")

    print(f"Successfully loaded {len(targets)} target files")
    return targets


def synthesize_plate(param_dict, duration, sample_rate=SAMPLE_RATE):
    """
    Synthesize plate audio from parameter dictionary.
    
    Args:
        param_dict: Dictionary of plate parameters
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        np.array: Synthesized audio signal
    """
    return ModalPlate.synthesize_from_params(
        param_dict,
        duration=duration,
        method='ir',  # displacement, matches the dataset
        sample_rate=sample_rate,
        normalize=False  # keep absolute amplitude so mu is identifiable
    )


def save_parameters_csv(params_dict, filepath):
    """Save parameters to CSV file."""
    df = pd.DataFrame([params_dict])
    df.to_csv(filepath, index=False)


# Six identifiable parameters S(P) = {mu, D/mu, T0/mu, Ly, op_x, op_y}.
# Raw (rho, h, E, T0) map through (mu, D/mu, T0/mu), which is the
# identifiable parameterisation (Eq. 15 of the paper).
DERIVED_KEYS = ["mu", "D_mu", "T0_mu", "Ly", "op_x", "op_y"]


def raw_to_derived(raw_params):
    """Return the 6-column S(P) dict from a full raw-parameter dict."""
    rho = float(raw_params["rho"])
    h = float(raw_params["h"])
    E = float(raw_params["E"])
    T0 = float(raw_params["T0"])
    nu = float(raw_params.get("nu", 0.25))
    mu = rho * h
    return {
        "mu": mu,
        "D_mu": E * h * h / (12.0 * (1 - nu ** 2) * rho),
        "T0_mu": T0 / mu,
        "Ly": float(raw_params["Ly"]),
        "op_x": float(raw_params["op_x"]),
        "op_y": float(raw_params["op_y"]),
    }


# Global variables for cost function
current_target_audio = None
current_target_duration = None

def cost_function(variable_params):
    """
    Global cost function for PSO optimization.
    Uses multi-scale spectral loss for comparing target and candidate audio.
    Uses global variables for target audio and duration.
    
    Args:
        variable_params: Array of variable parameter values only
    """
    # Convert variable params to full parameter dictionary
    param_dict = variable_params_to_full_params(variable_params)
    
    # Synthesize candidate
    candidate = synthesize_plate(param_dict, duration=current_target_duration)
    
    # Compute loss
    return multi_scale_spectral_loss(current_target_audio, candidate, STFT_CONFIGS)

# ===========================
# MAIN OPTIMIZATION FUNCTION
# ===========================

def run_baseline_experiment(target_folder="random-IR-10-1.0s"):
    """
    Run baseline PSO experiment on target folder.
    
    Args:
        target_folder: Path to folder containing target IR files
    """
    print("=" * 60)
    print("BASELINE PSO EXPERIMENT")
    print("=" * 60)
    
    # Setup
    if platform.system() == "Darwin":
        # macOS uses 'spawn' for multiprocessing; avoid processes to ensure shared state
        max_workers = 1
        print("macOS detected; using a single worker (no multiprocessing) to avoid spawn issues.")
    elif MAX_WORKERS is None:
        cpu_count = multiprocessing.cpu_count()
        max_workers = min(cpu_count, 16)
        print(f"Auto-detected {max_workers} workers from {cpu_count} CPU cores")
    else:
        max_workers = MAX_WORKERS
        print(f"Using {max_workers} workers")
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    # Initialize logging
    logger.initialize_logging(OUTPUT_DIR)
    
    # Load target files
    targets = load_target_files(target_folder)
    
    if len(targets) == 0:
        print("Error: No valid target files found")
        return
    
    # Results storage
    results = []
    
    # Process each target
    for i, (target_audio, filename) in enumerate(targets):
        print(f"\n" + "=" * 40)
        print(f"Processing target {i+1}/{len(targets)}: {filename}")
        print("=" * 40)
        
        # Adjust duration based on target audio length
        target_duration = len(target_audio) / SAMPLE_RATE
        print(f"Target duration: {target_duration:.3f}s")
        
        # Set global variables for cost function
        global current_target_audio, current_target_duration
        current_target_audio = target_audio
        current_target_duration = target_duration
        
        # Setup PSO
        variable_params = get_variable_params()
        bounds = [(v.low, v.high) for v in variable_params.values()]
        
        print(f"\nStarting PSO optimization...")
        print(f"Parameters: {PARTICLES} particles, {ITERATIONS} iterations")
        print(f"PSO settings: w={PSO_W}, c1={PSO_C1}, c2={PSO_C2}")
        print(f"Loss function: Multi-Scale Spectral Loss (MSS)")
        print(f"STFT configurations: {STFT_CONFIGS}")
        print(f"Optimizing {len(variable_params)} variable parameters (out of {len(plate_params)} total)")
        
        # Show which parameters are being optimized
        print(f"Variable parameters:")
        for k, v in variable_params.items():
            print(f"  {k}: [{v.low}, {v.high}]")
        
        fixed_params = get_fixed_params()
        print(f"Fixed parameters:")
        for k, v in fixed_params.items():
            print(f"  {k}: {v}")
        
        # Run optimization
        optimizer = PSO(
            cost_function, bounds, 
            num_particles=PARTICLES, 
            max_iter=ITERATIONS,
            w=PSO_W, c1=PSO_C1, c2=PSO_C2, 
            max_workers=max_workers
        )
        
        best_variable_params, best_loss, min_explored, max_explored, total_time, mean_time_per_iter = optimizer.optimize()
        
        # Convert best variable params to full dictionary
        best_params = variable_params_to_full_params(best_variable_params)
        
        print(f"\nOptimization completed!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Best loss: {best_loss:.6f}")
        print(f"Best parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        
        # Synthesize best match
        best_audio = synthesize_plate(best_params, duration=target_duration)
        
        # Generate output filename based on input filename
        # Remove extension and add index if needed
        base_name = Path(filename).stem
        if base_name.startswith('random_IR_'):
            # Extract index for consistency
            target_index = base_name.split('_')[-1]
        else:
            # Use sequential numbering
            target_index = f"{i+1:04d}"
        
        # Save best parameters in the six-column S(P) form required by the eval.
        # PSO still runs in the raw 7-D space (unchanged); we only convert the
        # final result before writing, since any point on the raw-parameter
        # fibre maps to the same S(P).
        best_params_file = output_path / f"best_params_{target_index}.csv"
        save_parameters_csv(raw_to_derived(best_params), best_params_file)

        # Save best audio: unnormalized npz (official) + peak-normalized wav (listening).
        best_peak = float(np.max(np.abs(best_audio)))
        best_norm = best_peak if best_peak > 0 else 1.0
        best_audio_file = output_path / f"best_audio_{target_index}.wav"
        sf.write(str(best_audio_file), (best_audio / best_norm).astype(np.float32),
                 SAMPLE_RATE, subtype='FLOAT')
        np.savez(
            str(output_path / f"best_audio_{target_index}.npz"),
            ir=best_audio.astype(np.float64),
            sample_rate=np.int32(SAMPLE_RATE),
            duration_s=np.float64(target_duration),
            normalization_factor=np.float64(best_norm),
        )
        
        # Store results
        iterations_required = PARTICLES * ITERATIONS
        result = {
            'target_file': filename,
            'target_index': target_index,
            'duration': target_duration,
            'optimization_time': total_time,
            'best_loss': best_loss,
            'iterations': iterations_required,
            'best_params': best_params
        }
        results.append(result)
        
        print(f"Saved results for {filename}")
    
    # ===========================
    # SUMMARY
    # ===========================
    
    print(f"\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    # Compute overall statistics
    best_losses = [r['best_loss'] for r in results]
    optimization_times = [r['optimization_time'] for r in results]
    
    print(f"Processed {len(results)} targets")
    print(f"Best losses (Multi-Scale Spectral Loss):")
    print(f"  Mean: {np.mean(best_losses):.6f}")
    print(f"  Std:  {np.std(best_losses):.6f}")
    print(f"  Min:  {np.min(best_losses):.6f}")
    print(f"  Max:  {np.max(best_losses):.6f}")
    
    print(f"Optimization time:")
    print(f"  Mean: {np.mean(optimization_times):.2f}s")
    print(f"  Total: {np.sum(optimization_times):.2f}s")
    
    # Save summary CSV
    summary_data = []
    for result in results:
        summary_data.append({
            'target_file': result['target_file'],
            'target_index': result['target_index'],
            'duration': result['duration'],
            'optimization_time': round(result['optimization_time'], 6),
            'best_loss': result['best_loss'],
            'iterations': result.get('iterations', PARTICLES * ITERATIONS)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_path / "experiment_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved experiment summary to: {summary_file}")
    
    print(f"\nAll results saved to: {output_path.absolute()}")
    print(f"Log file: {logger.get_log_file_path()}")
    
    return results


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Baseline PSO experiment for plate parameter estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python baseline.py                           # Use default folder (random-IR-10-1.0s)
    python baseline.py /path/to/wav/files        # Use custom folder
        """
    )
    
    parser.add_argument(
        'target_folder',
        nargs='?',
        default='random-IR-10-1.0s',
        help='Path to folder containing target IR files (default: random-IR-10-1.0s)'
    )
    
    args = parser.parse_args()
    
    try:
        results = run_baseline_experiment(args.target_folder)
        print("\nBaseline experiment completed successfully!")
        
    except ValueError as e:
        print(f"\nError: {e}")
        print("\n" + "="*60)
        parser.print_help()
        sys.exit(1)
        
    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
