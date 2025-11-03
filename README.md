# 1st DAFx Parameter Estimation Challenge

**Official repository for the 1st DAFx Parameter Estimation Challenge**  
Hosted at the next [DAFx](https://dafx.de/) conference: [DAFx26]()http://dafx26.mit.edu/)

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

To test the IR generation just run ModalPlate.py (the main function contains a minimum working example with default parameters.)

To generate a dataset of IRs run ModalPlate as a module like this:

```bash
python3 -m ModalPlate.DatasetGen
```

Each IR (wav file) is accompanied by a csv containing the plate parameters and the modal parameters. The sampling frequency will be 44100 throughout the entire challenge.

If you want to see how the baseline models perform over the generated dataset you can run these scripts

```bash
python TaskA/baseline.py
```
or

```bash
python TaskB/baseline.py
```

To evaluate the estimated results:

```bash
python TaskA/eval.py
```

The script handle different arguments, consult their help by calling them with the `-h` or `--help` argument.

The baseline scripts format the results as CSV files in the format we need to evaluate your proposals. You should use the same functions to get consistent output. The formatting includes information about the run time and the number of iterations performed (if any) by your algorithm. The run time is the time your algorithm takes to estimate the parameters once the IR is given: for data-driven algorithms that require a training, the run time is intended only as the time it takes for inference (provided that the training is done just once for any later inference).

---

## How to Participate

Participation is open to everyone: individual researchers, academic research groups, and teams from private companies.

Each team may submit **up to two proposals per task**. Proposals should not be just slightly different (e.g. two identical Deep Learning architectures trained with different hyperparameters) but have significant differences, justifying the need for a second short paper.

### Each proposal must include:
1. A **short paper** (~2 pages) describing the proposed algorithms.  
   - The paper **should not include results**; these will be computed by the organizers.  
2. A **ZIP archive** containing the estimated data in the required format.

The organizers will evaluate all submissions and compile a **ranking** of the results.

---

## Timeline and deadlines

Challenge opens on: November 5th 2025  
Target dataset will be uploaded by April 6th 2026  
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

For this task, participants must estimate the **physical parameters** of the metal plate from the given IRs.

| Parameter | Range | Description |
|------------|--------|-------------|
| `Ly` | [1.1, 4.0] | Plate height |
| `h` | [0.001, 0.005] | Plate thickness |
| `T0` | [0.01, 1000.0] | Tension |
| `rho` | [2430.0, 21230.0] | Density |
| `E` | [6.7e10, 22.0e10] | Young’s modulus |
| `T60_DC` | [6.0, 10.0] | Decay time at DC |
| `T60_F1` | [1.0, 5.0] | Decay time at frequency F1 |
| `op_x` | [0.51, 1.0] | Output transducer position x |
| `op_y` | [0.51, 1.0] | Output transducer position y |

**Fixed parameters (not to be estimated):**
- Lx: plate width (1m)
- Poisson’s ratio, \( \nu \ )
- `loss_F1`: frequency at which `T60_F1` is defined
- fp_x, fp_y = (0.335, 0.467), the plate input transducer position relative to Lx

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

### ❌ Brute-force methods (not allowed):
- Random search
- Grid search
- Any iterative method that does not exploit problem knowledge or the loss surface

### ✅ Allowed methods include:
- Metaheuristic algorithms (e.g., PSO, GSA, etc.)
- Deep learning approaches
- DDSP-based models
- Optimization techniques using problem-informed strategies

If unsure whether your method qualifies, contact the organizers.

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

For questions or clarifications, please contact the organizers via email:
Leonardo Gabrielli <l.gabrielli@staff.univpm.it>
Michele Ducceschi <michele.ducceschi@unibo.it>



