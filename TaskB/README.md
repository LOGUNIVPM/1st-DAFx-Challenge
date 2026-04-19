# Task B — Modal Parameter Estimation

Estimate the modal parameters (frequency, decay rate, and amplitude) of a metal plate from its impulse response. Each mode corresponds to a second-order all-pole resonator as described in the [challenge paper](../DAFxChallengeDetails.pdf) (Section III-B, Eq. 16).

This folder contains two scripts:

| Script         | Purpose                                              |
| :------------- | :--------------------------------------------------- |
| `baseline.py`  | Identify modal parameters directly from the IR files |
| `eval.py`      | Evaluate identified parameters against ground truth  |

The baseline reads only the IR `.npz` files (unnormalized displacement +
metadata) — no plate parameters — in accordance with the "no a priori
modal knowledge" rule of the challenge. The sibling `.wav` is peak-
normalized and is not used by this task, since the normalization
discards the `b_m` amplitude scale.

---

## Baseline

### Usage

Run from the project root (the folder containing both `TaskB/` and your data folder):

```bash
python TaskB/baseline.py --folder <folder_name> --fmin <Hz> --fmax <Hz> [--root <path>]
```

### Command-line options

| Option     | Required | Description                                                            |
| :--------- | :------: | :--------------------------------------------------------------------- |
| `--folder` |   yes    | Folder containing the input `random_IR_*.npz` files.                   |
| `--fmin`   |   yes    | Lower frequency bound of the modal band to analyse (Hz).               |
| `--fmax`   |   yes    | Upper frequency bound of the modal band to analyse (Hz).               |
| `--root`   |    no    | Project root path, if running from elsewhere (default: `.`).           |
| `--prom-db`, `--min-dist-hz`, `--prom-win-hz` | no | Peak-picker knobs (defaults: 6 dB, 2 Hz, same). |

### Example

```bash
python TaskB/baseline.py --folder random-IR-10-1.0s --fmin 50 --fmax 10000
```

This will:

1. Read all `random_IR_params_*.csv` files in `./random-IR-10-1.0s/`.
2. Identify modes whose natural frequencies fall within [50, 10000] Hz.
3. Write per-file results to `./experiment_results_TaskB/random_IR_identifiedModes_XXXX.csv`.
4. Write an index file at `./experiment_results_TaskB/_index_TaskB.json`.

### Input format

Each input file is a `random_IR_XXXX.npz` containing the scientific,
unnormalized displacement IR (float64) together with `sample_rate`,
`duration_s`, and `normalization_factor` metadata. The sibling
`random_IR_XXXX.wav` is peak-normalized and cannot be used for Task B:
the normalization discards the `b_m` amplitude scale.

### Output format

Each result CSV contains three columns:

| Column       | Description                     |
| :----------- | :------------------------------ |
| `f0_ident`   | Identified modal frequency (Hz) |
| `sigma_ident`| Identified decay constant (s⁻¹) |
| `gain_ident` | Identified modal gain           |

---

## Evaluation

### Metrics

The evaluation implements the metrics from the challenge paper (Section IV-C),
with two amendments over the original formulas: per-mode errors are clipped
to 1.0, and the mode-count term is normalised by M and clipped so the final
metric is strictly bounded.

| Symbol | Formula                                          | Description                        |
| :----- | :----------------------------------------------- | :--------------------------------- |
| re_Ω   | min(1, \|Ω_est − Ω_ref\| / Ω_ref) per mode       | Clipped per-mode frequency error   |
| re_σ   | min(1, \|σ_est − σ_ref\| / σ_ref) per mode       | Clipped per-mode decay error       |
| re_b   | min(1, \|b_est − b_ref\| / b_ref) per mode       | Clipped per-mode gain error        |
| RE_Ω, RE_σ, RE_b | (1/M) Σ re_* over all actual modes     | Mean per-plate errors              |
| RE0    | (RE_Ω + RE_σ + RE_b) / 3                        | Combined relative error (∈ [0,1])  |
| ΔM     | \|M − M̃\|                                       | Mode-count mismatch                 |
| **RE** | **RE0 + λ · min(1, ΔM/M)**                       | **Final ranking metric (∈ [0, 1+λ], default λ = 1)** |

Identified and actual modes are matched one-to-one by optimal (Hungarian)
assignment on the log-frequency distance, with a reject threshold of half
an octave by default. Pairs beyond the threshold are treated as unmatched:
the actual mode contributes 1.0 to each per-component error, and the
identified mode contributes to ΔM as a spurious mode.

### Usage

```bash
python TaskB/eval.py --experiment_folder <path> --target_folder <path> [--fmin <Hz>] [--fmax <Hz>]
```

### Command-line options

| Option                 | Required | Description                                                       |
| :--------------------- | :------: | :---------------------------------------------------------------- |
| `--experiment_folder`  |    no    | Folder with identified-mode CSVs (default: `experiment_results_TaskB`) |
| `--target_folder`      |    no    | Folder with ground-truth mode CSVs (default: `random-IR-10-1.0s`)     |
| `--fmin`               |    no    | Lower frequency bound for evaluation, Hz (default: `0`)               |
| `--fmax`               |    no    | Upper frequency bound for evaluation, Hz (default: `inf`)             |
| `--output`             |    no    | Custom path for the results CSV                                       |

The `--fmin` and `--fmax` values should match the band used when running the baseline or your method.

### Example

```bash
# Generate a dataset
python3 -m ModalPlate.DatasetGen --number 10 --duration 1.0

# Run the baseline
python TaskB/baseline.py --folder random-IR-10-1.0s --fmin 50 --fmax 10000

# Evaluate
python TaskB/eval.py --experiment_folder experiment_results_TaskB \
                     --target_folder random-IR-10-1.0s \
                     --fmin 50 --fmax 10000
```

### File matching

The script pairs files by their numeric ID: `random_IR_identifiedModes_0001.csv` in the experiment folder is matched against `random_IR_modes_0001.csv` in the target folder. The ground-truth mode files are produced automatically by `ModalPlate.DatasetGen` and contain columns `f0`, `sigma`, `gain`.

### Output

All outputs are written to the experiment folder:

| File                              | Contents                                              |
| :-------------------------------- | :---------------------------------------------------- |
| `evaluation_results_TaskB.csv`    | Per-file metrics (RE_Ω, RE_σ, RE_b, RE0, ΔM, RE)     |
| `evaluation_summary_TaskB.csv`    | Aggregated statistics (mean, std, min, max, median)   |
| `taskB_RE_histogram.png`          | Distribution of the final RE metric                   |
| `taskB_per_component_errors.png`  | Bar chart of RE_Ω, RE_σ, RE_b per file                |
| `taskB_mode_counts.png`           | Actual vs identified mode counts per file             |
