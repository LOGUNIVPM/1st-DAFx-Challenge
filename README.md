# 1st DAFx Parameter Estimation Challenge

**Official repository for the 1st DAFx Parameter Estimation Challenge**
Hosted at [DAFx26](http://dafx26.mit.edu/), Boston, 1–4 September 2026.

## UPDATE: DATASET RELEASED + CONTACT FORM

The DAFx Challenge is **now officially open**: the dataset has been released publicly!
The dataset includes the synthetic IRs in numpy format (npz) generated with our modal plate model. These are the files to be used for the identification tasks, and they contain the actual displacement wave. Additionally float32 normalized wav files are included for quick listening, these are peak-amplitude-normalized. The dataset does not include the ground truth, which is kept secret—safely locked away in a 🐊 crocodile-surrounded castle ;)
Only the parameter ranges are provided, among which parameters have been randomly selected.
The ground truth will be used by the Challenge organizers to evaluate your results. You must provide an estimate for each IR, the results will be averaged among all IRs.
Please find the dataset under the folder 2026-DATASET-STRIPPED

Anyone who wishes to participate to the challenge or that is at least thinking about it, can send their contact email via [this form](https://forms.cloud.microsoft/e/cnEsR7ZFgY).
Filling the form is not mandatory to be part of the challenge but allows the organizers to send timely updates if needed. The deadline for sending the results to the organizers is May 31st. Good luck!

---

## Introduction

The **DAFx Parameter Estimation Challenge** is an academic competition designed to bring researchers together to address a scientific problem of interest to the DAFx community.

For this first edition, the challenge focuses on the **estimation of parameters** of a **metal plate physical model** used for sound synthesis — similar to those employed in **plate reverbs**.

### Tasks Overview

- **Task A:** Estimate the physical parameters of the model from its impulse response (IR).  
- **Task B:** Estimate the modal parameters (frequency, decay, and amplitude).

Further details about the model, mathematical background, and evaluation metrics can be found in **`ChallengeDetails.pdf`**.

---

## ⚙️ Get Started

The plate model is provided as Python code.  
First install Python 3 and required libraries using

```bash
pip install -r requirements.txt
```

To test the IR generation, run:

```bash
python3 -m ModalPlate.ModalPlate
```

(the `__main__` of that module contains a minimum working example with default
parameters).

To generate a dataset of IRs:

```bash
python3 -m ModalPlate.DatasetGen --number 10 --duration 1.0
```

For each IR, `DatasetGen` writes four files (sampling rate 44 100 Hz throughout).

| File | Purpose |
| --- | --- |
| `random_IR_XXXX.npz` | **Scientific IR** — unnormalized displacement plus metadata. **This is the official input for the challenge tasks.** |
| `random_IR_XXXX.wav` | Peak-normalized, float32 — for **human listening only**. |
| `random_IR_params_XXXX.csv` | Raw plate parameters used (ground truth for Task A). |
| `random_IR_modes_XXXX.csv` | Per-mode `(f0, sigma, gain)` ground truth (for Task B). |

**Important:** the peak-normalized WAV discards the absolute amplitude of the
IR, which carries physical information (μ for Task A, $b_m$ for Task B).
Do **not** use the WAV for the challenge tasks — read the NPZ instead:

### Running the baselines

```bash
# Task A: PSO over S(P) = {mu, D/mu, T0/mu, Ly, op_x, op_y}
python3 TaskA/baseline.py random-IR-10-1.0s

# Task B: modal identification from the IR
python3 TaskB/baseline.py --folder random-IR-10-1.0s --fmin 50 --fmax 10000
```

### Evaluating the results

```bash
python3 TaskA/eval.py --experiment_folder experiment_results_taskA --target_folder random-IR-10-1.0s
python3 TaskB/eval.py --experiment_folder experiment_results_TaskB \
                     --target_folder random-IR-10-1.0s \
                     --fmin 50 --fmax 10000
```

---

## How to Participate

Participation is open to everyone: individual researchers, academic research groups, and teams from private companies.

**Important**: all perspective participants are kindly asked to send a contact email to the organizers even if they are unsure whether they will submit their results or not. This is to ensure that we can bulk send timely notifications via email to anyone who is interested in the challenge.

Each team may submit **up to two proposals per task**. Proposals should not be just slightly different (e.g. two identical Deep Learning architectures trained with different hyperparameters) but have significant differences, justifying the need for a second short paper.

### Each proposal must include:
1. A **short paper** (~2 pages) describing the proposed algorithms.  
   - The paper **should not include results**; these will be computed by the organizers.  
2. A **ZIP archive** containing the estimated data in the required format.

The organizers will evaluate all submissions and compile a **ranking** of the results.

---

## Timeline and deadlines

Challenge opens on: November 5th 2025  
Target dataset will be uploaded by April 17th 2026  
Experiments must be sent to the organizers by May 31st 2026  
Results will be notified at DAFx26 (1-4 Sept. 2026)!  

---

## Target Dataset

The folder `TargetDataset` will contain several impulse responses (IRs) to be identified.  
They are valid for both tasks and provided as `.wav` files only.

The **ground truth parameters** for these files are known to the organizers and will be used to compute the final metrics.

The dataset will be provided at a later time, see the deadlines.

---

## 🧩 Task A — Physical Parameter Estimation

For this task, participants must estimate a subset of the physical parameters of the metal plate from the given impulse responses (IRs).

Specifically, the following parameter set must be inferred:

S(P) = { μ := ρh, D/μ, T₀/μ, Ly, xo, yo }

These six parameters uniquely define the plate’s impulse response and modal distribution.

### Parameter ranges

| Parameter | Symbol | Range | Description |
|----------|--------|--------|-------------|
| Density–thickness ratio | μ = ρ h | derived | Volume density times thickness |
| Rigidity ratio | D/μ | derived | Flexural rigidity normalized by μ |
| Tension ratio | T₀/μ | derived | Tension normalized by μ |
| Plate height | Ly | [1.1, 4.0] m | Plate vertical dimension |
| Output position (x) | xo | [0.51, 1.0] | Output transducer x-position (fraction of Lx) |
| Output position (y) | yo | [0.51, 1.0] | Output transducer y-position (fraction of Ly) |

### Fixed parameters (not to be estimated)

The following parameters are fixed for all simulations:

- Lx = 1.0 m (plate width)
- ν = 0.25 (Poisson’s ratio)
- τ₀ = 6 s (decay time at DC)
- τ₁ = 2 s (decay time at 500 Hz)
- f₁ = 500 Hz (reference loss frequency)
- xi = 0.335 · Lx (input position x)
- yi = 0.467 · Ly (input position y)

### Important note

The parameters ρ, h, E, and T₀ are not estimated directly. Instead, they are combined into the derived parameters μ, D/μ, and T₀/μ, resulting in a total of six estimated parameters.

---

## 🧩 Task B — Modal Parameter Estimation

In this task, the plate’s IR can be modeled as a **bank of 2nd-order all-pole resonators**, each defined by:

- **Center frequency**
- **Decay rate**
- **Amplitude**

The **number of modes is unknown**, and evaluation metrics will penalize missing or extra modes.

More details are provided in **`DAFxChallengeDetails.pdf`**.

---

## 🚀 Allowed Methods

All methods are allowed **except brute-force approaches**.

### ❌ Not allowed
- Random search, grid search.
- Any iterative method that does not exploit problem knowledge or the loss
  surface.

### ✅ Allowed
- Metaheuristic algorithms (e.g. PSO, GSA).
- Deep-learning approaches.
- DDSP-based models.
- Optimization techniques using problem-informed strategies.

**Important**: A priori knowledge of the plate's modal distribution — such as
the analytical frequency formula (Eq. 8) or the decay behaviour described in
Section II-C of the paper — must **not** be used. Only the provided synthetic
impulse responses should be employed, as if they were obtained from actual
measurements.

If unsure whether your method qualifies, contact the organisers.

---

## 📦 Submission Requirements

Each proposal must include:

- A **short paper (PDF)** describing the algorithm  
- A **ZIP archive** containing:
  - The **estimated data** (formatted as prescribed by the organizers)  
  - The **number of iterations/trials/epochs** (use `0` if the method is non-iterative)  
  - The **total compute time** measured on your hardware, along with a short **hardware description**

---

## 📫 Organizers Contact

For questions or clarifications, please contact the organisers by email:
- Leonardo Gabrielli <l.gabrielli@staff.univpm.it>
- Michele Ducceschi <michele.ducceschi@unibo.it>
