#!/usr/bin/env python3
"""
Random Plate Dataset Generator
==============================

Generates a dataset of random plate impulse responses using uniform distribution
within the parameter ranges defined in ParamRange.py.

Usage:
    python generate_random_dataset.py [--number NUM] [--duration DURATION]
    
Arguments:
    --number:   Number of impulse responses to generate (default: 10)
    --duration: Duration of each IR in seconds (default: 1.0)
    
Output:
    Creates folder "random-IR-<num>-<duration>" containing:
    - random_IR_XXXX.wav: Audio files
    - random_IR_params_XXXX.csv: Parameter files
    - random_IR_modes_XXXX.csv: Modal vector files (G1, G2, P)
    - generation_summary.txt: Summary of generation process
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from ModalPlate.ModalPlate import ModalPlate
from ModalPlate.ParamRange import params as plate_params
import soundfile as sf


def generate_random_parameters(num_sets=10):
    """
    Generate random parameter sets within the parameter ranges.
    
    Args:
        num_sets: Number of random parameter sets to generate
        
    Returns:
        list: List of parameter dictionaries
    """
    random_params_list = []
    
    print(f"Generating {num_sets} random parameter sets...")
    
    for i in range(num_sets):
        param_dict = {}
        for param_name, param_range in plate_params.items():
            if param_range.low == param_range.high:
                # Fixed parameter
                param_dict[param_name] = param_range.low
            else:
                # Random parameter within range
                param_dict[param_name] = np.random.uniform(param_range.low, param_range.high)
        
        random_params_list.append(param_dict)
        
        if (i + 1) % 10 == 0 or i == 0 or i == num_sets - 1:
            print(f"  Generated parameter set {i + 1}/{num_sets}")
    
    return random_params_list


def synthesize_plate_and_data(param_dict, duration=5.0, sample_rate=44100):
    """
    Synthesize plate audio from parameter dictionary and extract modal vectors.
    
    Args:
        param_dict: Dictionary of plate parameters
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        tuple: (audio_signal, modal_vectors_dict)
            - audio_signal: np.array of synthesized audio
            - modal_vectors_dict: dict with 'P', 'sigma', 'f0' keys containing the modal vectors
    """
    # Create ModalPlate instance
    plate = ModalPlate(sample_rate=sample_rate, plate_params=param_dict)
    
    # Synthesize audio
    audio = plate.synthesize_ir_method(duration=duration, velCalc=True, normalize=True)
    
    # Extract modal vectors
    modal_vectors = {
        'P': plate.Pvec.copy(),
        'sigma': plate.sigma_vec.copy(),
        'f0': plate.f0_vec.copy()
    }
    
    return audio, modal_vectors


def save_modal_vectors_csv(modal_vectors, filepath):
    """
    Save modal vectors to CSV file.
    
    Args:
        modal_vectors: Dictionary with 'P', 'sigma', 'f0' keys containing modal vectors
        filepath: Path to save CSV file
    """
    # Create DataFrame with the modal vectors
    df = pd.DataFrame({
        'P': modal_vectors['P'],
        'sigma': modal_vectors['sigma'],
        'f0': modal_vectors['f0']
    })
    df.to_csv(filepath, index=False)


def save_parameters_csv(param_dict, filepath):
    """
    Save parameter dictionary to CSV file.
    
    Args:
        param_dict: Dictionary of plate parameters
        filepath: Path to save CSV file
    """
    # Convert to DataFrame for easy CSV saving
    df = pd.DataFrame([param_dict])
    df.to_csv(filepath, index=False)


def generate_dataset(num_ir=10, duration=5.0, sample_rate=44100):
    """
    Generate complete random plate dataset.
    
    Args:
        num_ir: Number of impulse responses to generate
        duration: Duration of each IR in seconds
        sample_rate: Sample rate in Hz
    """
    print(f"=== Random Plate Dataset Generator ===")
    print(f"Number of IRs: {num_ir}")
    print(f"Duration: {duration}s")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output folder
    folder_name = f"random-IR-{num_ir}-{duration:.1f}s"
    output_dir = Path(folder_name)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir.absolute()}")
    
    # Generate random parameters
    parameter_sets = generate_random_parameters(num_ir)
    
    # Statistics tracking
    successful_generations = 0
    failed_generations = 0
    generation_times = []
    
    # Generate audio files and save parameters
    print(f"\nGenerating audio files...")
    
    for i, params in enumerate(parameter_sets):
        try:
            start_time = datetime.now()
            
            # Generate audio and extract modal vectors
            audio, modal_vectors = synthesize_plate_and_data(params, duration=duration, sample_rate=sample_rate)
            
            # Generate filenames with zero-padded numbers
            file_index = f"{i+1:04d}"
            audio_filename = f"random_IR_{file_index}.wav"
            params_filename = f"random_IR_params_{file_index}.csv"
            modes_filename = f"random_IR_modes_{file_index}.csv"
            
            audio_path = output_dir / audio_filename
            params_path = output_dir / params_filename
            modes_path = output_dir / modes_filename
            
            # Save audio file
            sf.write(str(audio_path), audio, sample_rate)
            
            # Save parameters CSV
            save_parameters_csv(params, params_path)
            
            # Save modal vectors CSV
            save_modal_vectors_csv(modal_vectors, modes_path)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            generation_times.append(generation_time)
            successful_generations += 1
            
            print(f"  Generated {i+1}/{num_ir}: {audio_filename} + params + modes ({generation_time:.2f}s)")
            
        except Exception as e:
            failed_generations += 1
            print(f"  ERROR generating {i+1}/{num_ir}: {e}")
    
    # Generate summary file
    summary_path = output_dir / "generation_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("Random Plate Dataset Generation Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Generation timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of IRs requested: {num_ir}\n")
        f.write(f"Duration per IR: {duration}s\n")
        f.write(f"Sample rate: {sample_rate} Hz\n")
        f.write(f"Output directory: {output_dir.absolute()}\n\n")
        
        f.write("Parameter Ranges Used:\n")
        f.write("-" * 20 + "\n")
        for param_name, param_range in plate_params.items():
            if param_range.low == param_range.high:
                f.write(f"{param_name}: {param_range.low} (fixed)\n")
            else:
                f.write(f"{param_name}: [{param_range.low}, {param_range.high}]\n")
        f.write("\n")
        
        f.write("Generation Results:\n")
        f.write("-" * 18 + "\n")
        f.write(f"Successful generations: {successful_generations}\n")
        f.write(f"Failed generations: {failed_generations}\n")
        f.write(f"Success rate: {successful_generations/num_ir*100:.1f}%\n")
        
        if generation_times:
            f.write(f"Average generation time: {np.mean(generation_times):.2f}s\n")
            f.write(f"Total generation time: {sum(generation_times):.2f}s\n")
            f.write(f"Min generation time: {min(generation_times):.2f}s\n")
            f.write(f"Max generation time: {max(generation_times):.2f}s\n")
        f.write("\n")
        
        f.write("Files Generated:\n")
        f.write("-" * 15 + "\n")
        for i in range(successful_generations):
            file_index = f"{i+1:04d}"
            f.write(f"random_IR_{file_index}.wav\n")
            f.write(f"random_IR_params_{file_index}.csv\n")
            f.write(f"random_IR_modes_{file_index}.csv\n")
    
    print(f"\n=== Generation Complete ===")
    print(f"Successful: {successful_generations}/{num_ir} files")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Summary saved to: {summary_path}")
    
    if generation_times:
        print(f"Average generation time: {np.mean(generation_times):.2f}s per file")
        print(f"Total time: {sum(generation_times):.2f}s")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate random plate impulse response dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_random_dataset.py                          # 10 IRs, 5s each
  python generate_random_dataset.py --number 20              # 20 IRs, 5s each  
  python generate_random_dataset.py --duration 2.5           # 10 IRs, 2.5s each
        """
    )
    
    parser.add_argument(
        '--number', 
        type=int, 
        default=10,
        help='Number of impulse responses to generate (default: 10)'
    )
    
    parser.add_argument(
        '--duration', 
        type=float, 
        default=1.0,
        help='Duration of each IR in seconds (default: 1.0)'
    )
    
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=44100,
        help='Sample rate in Hz (default: 44100)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducible results'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.number <= 0:
        print("Error: Number of IRs must be positive")
        sys.exit(1)
    
    if args.duration <= 0:
        print("Error: Duration must be positive")
        sys.exit(1)
    
    if args.sample_rate <= 0:
        print("Error: Sample rate must be positive") 
        sys.exit(1)
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Generate dataset
    try:
        generate_dataset(
            num_ir=args.number,
            duration=args.duration,
            sample_rate=args.sample_rate
        )
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()