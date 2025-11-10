### README.md

````markdown
# TaskB

**TaskB** identifies plate modal parameters from CSV-defined plate data generated via the ModalPlate class.  
It reads plate and material parameters from CSV files and attempts to identify all modes whose natural frequencies fall within a user-specified frequency band.

---

## Usage

Run the script from your project root (the folder that contains both `TaskB/` and your data folder):

```bash
python TaskB/baseline.py --folder <folder_name> --fmin <Hz> --fmax <Hz> [--root <path_to_project_root>]
````

### Command-line options

| Option                   | Required | Description                                                                                         |
| :----------------------- |:--------:| :-------------------------------------------------------------------------------------------------- |
| `--folder <folder_name>` |   yes    | Name of the folder containing your input CSV files (for example `random-IR-10-10.0s`).              |
| `--fmin <Hz>`            |   yes    | Lower frequency bound of the modal band to analyze (in Hz).                                         |
| `--fmax <Hz>`            |   yes    | Upper frequency bound of the modal band to analyze (in Hz).                                         |
| `--root <path>`          |    no    | Path to the project root if you are running the script from elsewhere (default: current directory). |

---

## Example

```bash
python TaskB/baseline.py --folder random-IR-10-10.0s --fmin 20 --fmax 10000
```

This will:

1. Read all `random_IR_params_*.csv` files in `./random-IR-10-10.0s/`.
2. Identify all plate modes whose natural frequencies fall between 20 Hz and 10000 Hz.
3. Save the results to:

   ```
   ./experiment_results_TaskB/<input_csv_basename>.csv
   ```
4. Create an index file listing processed CSVs:

   ```
   ./experiment_results_TaskB/_index_TaskB.json
   ```

---

## Input Format

Each input CSV must be named as:

```
random_IR_params_*.csv
```

and contain plate/material/geometry parameters such as:

* `Lx`, `Ly`, `h`, `T0`, `rho`, `E`, `nu`
* `SR`, `DURATION_S`, `fmax`
* `T60_F0`, `T60_F1`, `loss_F1`
* `fp_x`, `fp_y`, `op_x`, `op_y`
* `velCalc`
* Optional peak-picking parameters: `PROM_DB`, `MIN_DIST_HZ`, `PROM_WIN_HZ`

Missing fields automatically use baseline default values.

---

## Output

Each processed CSV produces an output file with columns:

| Column         | Description                      |
| :------------- | :------------------------------- |
| `mode_index`   | Mode number in processing order  |
| `f0_actual`    | Theoretical modal frequency (Hz) |
| `f0_ident`     | Identified modal frequency (Hz)  |
| `sigma_actual` | Theoretical damping coefficient  |
| `sigma_ident`  | Identified damping coefficient   |
| `gain_actual`  | Theoretical modal gain           |
| `gain_ident`   | Identified modal gain            |

A summary index file (`_index_TaskB.json`) lists all processed inputs and their corresponding outputs.

---

## Notes

* The script is self-contained and mirrors the baseline modal identification process.
* All CSV column names are **case-insensitive**.
* The folder passed to `--folder` is taken **as-is**; it is not assumed to be inside another folder.
* Default simulation and material constants are defined inside `baselineTaskB.py` under `BASE_DEFAULTS`.

```
