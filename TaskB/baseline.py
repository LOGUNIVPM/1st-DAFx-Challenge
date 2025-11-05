#!/usr/bin/env python3
""" 
TaskB: Identify plate modal parameters from CSV-defined plate data (frequency-range based). 
 
Usage: 
  python taskB.py --folder <folder_name> --fmin <Hz> --fmax <Hz> [--root <path_to_project_root>] 
 
Description: 
  This script processes all CSV files matching "random_IR_params_*.csv" inside the folder specified 
  with --folder. The folder name itself (e.g., "random-IR-10-10.0s/") is provided directly and is 
  not assumed to be a subfolder within another path. 
 
  For each CSV file, it: 
    - Reads plate/material/IO/loss parameters (falling back to baseline defaults where missing) 
    - Runs the same modal parameter identification method as baselineModalParameters.py 
    - Selects all modes whose natural frequencies fall within [FMIN, FMAX] 
    - Saves the identified results to: 
          experiment_results_TaskB/<input_csv_basename>.csv 
 
Options: 
  --folder   Name of the folder containing the random_IR_params_*.csv files. 
  --fmin     Lower bound (Hz) of the modal frequency band to identify. 
  --fmax     Upper bound (Hz) of the modal frequency band to identify. 
  --root     (Optional) Root path of the project (default: current directory). 
 
Example: 
  python taskB.py --folder random-IR-10-10.0s --fmin 20 --fmax 50 
 
Notes: 
  - This script is self-contained and reproduces the identification method from 
    baselineModalParameters.py. 
  - All columns are read case-insensitively. Missing parameters use baseline defaults. 
  - Expected CSV keys include: 
        Lx, Ly, h, T0, rho, E, nu, 
        SR, DURATION_S, fmax, 
        T60_F0, T60_F1, loss_F1, 
        fp_x, fp_y, op_x, op_y, 
        velCalc, 
        FMIN, FMAX, PROM_DB, MIN_DIST_HZ, PROM_WIN_HZ 
"""

import argparse
import glob
import os
import sys
import math
import numpy as np
import pandas as pd

# ------------------------- Baseline defaults (from baselineModalParameters.py) -------------------------
BASE_DEFAULTS = dict(
    SR=48_000.0,
    DURATION_S=5.0,
    fmax=10_000.0,
    Lx=2.2,
    Ly=1.03,
    h=0.002,
    T0=100.0,
    rho=8000.0,
    E=2e11,
    nu=0.3,
    T60_F0=10.0,
    T60_F1=8.0,
    loss_F1=500.0,
    fp_x=0.521,
    fp_y=0.422,
    op_x=0.23,
    op_y=0.643,
    MODE_COUNT=5,
    MODE_START_RANK=0,
    velCalc=False,
    FMIN=20.0,
    FMAX=60.0,
    PROM_DB=0.1,
    MIN_DIST_HZ=0.05,
    PROM_WIN_HZ=None,  # if None -> MIN_DIST_HZ
)

# ---------------------------- Core math copied from baseline ----------------------------
def modal_params_calc(Lx, Ly, T0, D, rho, h, maxOm):
    DDx = int(np.floor(Lx / np.pi * np.sqrt((-T0 + np.sqrt(T0 ** 2 + 4 * maxOm ** 2 * rho * h * D)) / (2 * D))))
    DDy = int(np.floor(Ly / np.pi * np.sqrt((-T0 + np.sqrt(T0 ** 2 + 4 * maxOm ** 2 * rho * h * D)) / (2 * D))))

    ov = np.zeros((max(DDx,1) * max(DDy,1), 3))
    ind = 0
    for m in range(1, max(DDx,1) + 1):
        for n in range(1, max(DDy,1) + 1):
            g1 = (m * np.pi / Lx) ** 2 + (n * np.pi / Ly) ** 2
            g2 = g1 * g1
            gf = T0 / (rho * h) * g1 + D / (rho * h) * g2
            gf = np.sqrt(max(gf, 0.0))
            ov[ind, :] = [gf, m, n]
            ind += 1

    # Remove very-low modes (<~20 Hz) by pushing them beyond maxOm, then sort/filter
    ov[:, 0] = np.where(ov[:, 0] < 20 * 2 * np.pi, maxOm + 1000, ov[:, 0])
    ov = ov[np.argsort(ov[:, 0])]
    ov = ov[ov[:, 0] <= maxOm]
    return ov

def select_modes_rank(ov, count=1, start_rank=0):
    if ov.ndim != 2 or ov.shape[1] != 3:
        raise ValueError("ov must be a (N,3) array [omega, m, n].")
    end = min(start_rank + count, len(ov))
    sel = ov[start_rank:end, :]
    return sel

def modal_arrays_calc(fp_x, fp_y, op_x, op_y, ov, alpha, beta, ms, k):
    DIM = ov.shape[0]
    G1vec, G2vec, Pvec = np.zeros(DIM), np.zeros(DIM), np.zeros(DIM)

    for m in range(DIM):
        omref, mind, nind = ov[m]
        InWeight  = np.cos(fp_x * np.pi * mind) * np.cos(fp_y * np.pi * nind)
        OutWeight = np.cos(op_x * np.pi * mind) * np.cos(op_y * np.pi * nind)
        b   = OutWeight * InWeight
        sig = alpha + beta * omref ** 2
        G1vec[m] = 2 * np.cos(omref * k) * np.exp(-sig * k)
        G2vec[m] = np.exp(-2 * sig * k)
        Pvec[m]  = b * k**2 * np.exp(-sig * k) / ms

    return G1vec, G2vec, Pvec

def IR_time_int(G1vec, G2vec, Pvec, Ts, k, velCalc):
    DIM = len(G1vec)
    q1, q2 = np.zeros(DIM), np.zeros(DIM)
    y = np.zeros(Ts)
    yPrev = 0.0
    for n in range(Ts):
        fin = 1.0 if n == 0 else 0.0
        q   = G1vec * q1 - G2vec * q2 + Pvec * fin
        yCur = np.sum(q1)
        y[n] = (yCur - yPrev) / k if velCalc else yCur
        q2, q1, yPrev = q1, q, yCur
    return y

# ---------------------------- Helpers ----------------------------
def get_val(row, keys, default):
    """Fetch value from a pandas Series using first matching key (case-insensitive)."""
    row_lower = {k.lower(): v for k, v in row.items()}
    for k in keys:
        if k.lower() in row_lower and pd.notna(row_lower[k.lower()]):
            return row_lower[k.lower()]
    return default

def compute_single(csv_path, defaults: dict):
    # Load 1-row or multi-row CSV; if multi-row, use the first row for parameters
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} is empty.")
    row = df.iloc[0].to_dict()

    # Pull parameters (fallback to defaults)
    params = dict(defaults)  # copy
    mapping = {
        "SR": ["SR", "sample_rate", "fs", "Fs"],
        "DURATION_S": ["DURATION_S", "duration_s", "duration", "T"],
        "fmax": ["fmax", "FMAXMODAL"],
        "Lx": ["Lx", "L_x"],
        "Ly": ["Ly", "L_y"],
        "h": ["h", "thickness"],
        "T0": ["T0", "tension", "T"],
        "rho": ["rho", "density"],
        "E": ["E", "youngs_modulus", "Young"],
        "nu": ["nu", "poisson"],
        "T60_F0": ["T60_F0", "T60F0", "t60_f0"],
        "T60_F1": ["T60_F1", "T60F1", "t60_f1"],
        "loss_F1": ["loss_F1", "lossf1"],
        "fp_x": ["fp_x", "in_x"],
        "fp_y": ["fp_y", "in_y"],
        "op_x": ["op_x", "out_x"],
        "op_y": ["op_y", "out_y"],
                "velCalc": ["velCalc", "velocity_output"],
        "FMIN": ["FMIN", "band_fmin", "fmin_band"],
        "FMAX": ["FMAX", "band_fmax", "fmax_band"],
        "PROM_DB": ["PROM_DB", "prom_db", "prominence_db"],
        "MIN_DIST_HZ": ["MIN_DIST_HZ", "min_dist_hz"],
        "PROM_WIN_HZ": ["PROM_WIN_HZ", "prom_win_hz"],
    }
    for key, aliases in mapping.items():
        params[key] = get_val(row, aliases, params[key])

    # Ensure types
    params["velCalc"] = bool(params["velCalc"])

    # Derivations
    SR = float(params["SR"]); DURATION_S = float(params["DURATION_S"]); fmax = float(params["fmax"])
    Lx = float(params["Lx"]); Ly = float(params["Ly"]); h = float(params["h"])
    T0 = float(params["T0"]); rho = float(params["rho"]); E = float(params["E"]); nu = float(params["nu"])
    T60_F0 = float(params["T60_F0"]); T60_F1 = float(params["T60_F1"]); loss_F1 = float(params["loss_F1"])
    fp_x = float(params["fp_x"])
    fp_y = float(params["fp_y"])
    op_x = float(params["op_x"])
    op_y = float(params["op_y"])
    FMIN = float(params["FMIN"])
    FMAX = float(params["FMAX"])
    PROM_DB = float(params["PROM_DB"]); MIN_DIST_HZ = float(params["MIN_DIST_HZ"])
    PROM_WIN_HZ = float(params["PROM_WIN_HZ"]) if params["PROM_WIN_HZ"] is not None and not (isinstance(params["PROM_WIN_HZ"], float) and math.isnan(params["PROM_WIN_HZ"])) else MIN_DIST_HZ

    Ts    = int(SR * DURATION_S)
    D     = E * h ** 3 / (12 * (1 - nu ** 2))
    ms    = 0.25 * rho * h * Lx * Ly
    k     = 1.0 / SR
    maxOm = fmax * 2 * np.pi
    nv    = np.arange(Ts)
    fv    = nv * SR / Ts
    df    = SR / Ts

    # Rayleigh damping coefficients (same as baseline)
    OmDamp1 = 0.0
    OmDamp2 = 2 * np.pi * loss_F1
    dOmSq   = OmDamp2 ** 2 - OmDamp1 ** 2
    alpha   = 3 * np.log(10) / dOmSq * (OmDamp2 ** 2 / T60_F0 - OmDamp1 ** 2 / T60_F1)
    beta    = 3 * np.log(10) / dOmSq * (1 / T60_F1 - 1 / T60_F0)

    # 1) Modal list
    ov_full = np.array(modal_params_calc(Lx, Ly, T0, D, rho, h, maxOm))

    # 2) Select ALL modes whose natural frequencies fall within [FMIN, FMAX]
    freq_modes = ov_full[:, 0] / (2 * np.pi)
    band_mask = (freq_modes >= FMIN) & (freq_modes <= FMAX)
    ov = ov_full[band_mask]
    DIM = ov.shape[0]

    # 3) Modal arrays
    G1vec, G2vec, Pvec = modal_arrays_calc(fp_x, fp_y, op_x, op_y, ov, alpha, beta, ms, k)

    # 4) Time integration -> time-domain impulse response
    outInt = IR_time_int(G1vec, G2vec, Pvec, Ts, k, params["velCalc"])

    # 5) Spectrum
    fftout = np.fft.fft(outInt)

    # Peak picking & identification (copied/adapted from baseline, without prints)
    N     = Ts
    freq  = fv[:N // 2 + 1]
    spec  = fftout[:N // 2 + 1]
    mag   = np.abs(spec)
    mag_db = 20 * np.log10(np.maximum(mag, 1e-20))

    band = (freq >= FMIN) & (freq <= FMAX)
    bi   = np.flatnonzero(band)

    results = []
    if len(bi) >= 3:
        i0, i1 = bi[0], bi[-1]
        prom_halfw    = max(1, int(round(PROM_WIN_HZ / df)))
        min_dist_bins = max(1, int(round(MIN_DIST_HZ / df)))

        def parabolic_interp(y_m1, y0, y_p1):
            denom = (y_m1 - 2 * y0 + y_p1)
            if abs(denom) < 1e-20:
                return 0.0, y0
            d = 0.5 * (y_m1 - y_p1) / denom
            d = float(np.clip(d, -0.5, 0.5))
            peak = y0 - 0.25 * (y_m1 - y_p1) * d
            return d, float(peak)

        cands = []
        for k_pk in range(max(i0 + 1, 1), min(i1 - 1, len(mag_db) - 1)):
            if mag_db[k_pk] > mag_db[k_pk - 1] and mag_db[k_pk] > mag_db[k_pk + 1]:
                L = max(i0, k_pk - prom_halfw)
                R = min(i1, k_pk + prom_halfw)
                left_min  = np.min(mag_db[L:k_pk]) if k_pk > L else mag_db[k_pk] - 1e9
                right_min = np.min(mag_db[k_pk + 1:R + 1]) if R > k_pk else mag_db[k_pk] - 1e9
                prom = mag_db[k_pk] - max(left_min, right_min)
                if prom >= PROM_DB:
                    d, pk_db_ref = parabolic_interp(mag_db[k_pk - 1], mag_db[k_pk], mag_db[k_pk + 1])
                    f0 = freq[k_pk] + d * df
                    pk_lin = 10 ** (pk_db_ref / 20)
                    cands.append((k_pk, d, f0, pk_db_ref, pk_lin, prom))

        cands.sort(key=lambda t: t[4], reverse=True)
        selected = []
        taken = np.zeros_like(mag_db, dtype=bool)
        for k_pk, d, f0, pk_db_ref, pk_lin, prom in cands:
            left  = max(0, k_pk - min_dist_bins)
            right = min(len(taken), k_pk + min_dist_bins + 1)
            if taken[left:right].any():
                continue
            selected.append((k_pk, d, f0, pk_db_ref, pk_lin, prom))
            taken[left:right] = True

        def interp_cross(f, y, k_left, target):
            y1, y2 = y[k_left], y[k_left + 1]
            if (y1 - target) * (y2 - target) > 0:
                return None
            t = (target - y1) / (y2 - y1 + 1e-20)
            return f[k_left] + t * (f[k_left + 1] - f[k_left])

        def lagrange_quad_complex(y_m1, y0, y_p1, d):
            l_m1 = 0.5 * d * (d - 1.0)
            l_0  = 1.0 - d**2
            l_p1 = 0.5 * d * (d + 1.0)
            return l_m1*y_m1 + l_0*y0 + l_p1*y_p1

        for k_pk, d, f0, pk_db_ref, pk_lin, prom in selected:
            target = pk_lin / np.sqrt(2.0)
            f1 = None
            for kk in range(k_pk - 1, i0, -1):
                f1 = interp_cross(freq, mag, kk, target)
                if f1 is not None:
                    break
            f2 = None
            for kk in range(k_pk, i1):
                f2 = interp_cross(freq, mag, kk, target)
                if f2 is not None:
                    break

            if (f1 is None) or (f2 is None) or (f2 <= f1):
                bw = np.nan
                sigma = np.nan
            else:
                bw = f2 - f1
                sigma = np.pi * bw

            km1 = max(k_pk - 1, 0)
            kp1 = min(k_pk + 1, len(spec) - 1)
            H_hat = lagrange_quad_complex(spec[km1], spec[k_pk], spec[kp1], d)
            ImH = np.imag(H_hat)

            Omega = 2.0 * np.pi * f0
            gain  = -2 * ImH * sigma * k * np.sin(Omega * k) if np.isfinite(sigma) else np.nan

            results.append(dict(
                f0_ident=f0,
                pk_db_ref=pk_db_ref,
                prominence_db=prom,
                f1=f1, f2=f2, bw=bw, sigma=sigma, ImH=ImH, gain_ident=gain
            ))

    # Assemble per-mode table combining "actual" and "identified" quantities
    # Sort identified by f0; pad/truncate to DIM to align with modal ranks.
    results = sorted(results, key=lambda r: r["f0_ident"]) if results else []
    f0_ident_list    = [r["f0_ident"] for r in results]
    sigma_ident_list = [r["sigma"] for r in results]
    gain_ident_list  = [r["gain_ident"] for r in results]

    f0_actual_list    = ov[:, 0] / (2 * np.pi) if DIM > 0 else np.array([])
    sigma_actual_list = (3 * np.log(10) / (np.array([r["bw"] for r in results]) * np.pi)) if False else (alpha + beta * ov[:, 0]**2 if DIM>0 else np.array([]))
    Pvec_actual_list  = G1vec*0 + Pvec  # same length as DIM

    # Build rows
    rows = []
    for i in range(max(DIM, len(results))):
        def safe(lst, idx, fill=np.nan):
            try:
                return lst[idx]
            except Exception:
                return fill
        rows.append(dict(
            f0_ident=safe(f0_ident_list, i),
            sigma_ident=safe(sigma_ident_list, i),
            gain_ident=safe(gain_ident_list, i)
        ))

    out_df = pd.DataFrame(rows)
    meta = dict(
        csv_source=os.path.basename(csv_path),
        params=params
    )
    return out_df, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Subfolder name under random-IR-10-10.0s/ containing the CSVs.")
    ap.add_argument("--root", default=".", help="Path to the project root containing random-IR-10-10.0s/.")
    ap.add_argument("--fmin", type=float, required=True, help="Lower bound of frequency band (Hz).")
    ap.add_argument("--fmax", type=float, required=True, help="Upper bound of frequency band (Hz).")
    args = ap.parse_args()

    csv_dir = os.path.join(args.root, args.folder)
    if not os.path.isdir(csv_dir):
        print(f"[ERROR] CSV directory not found: {csv_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.join(args.root, "experiment_results_TaskB")
    os.makedirs(out_dir, exist_ok=True)

    csv_paths = sorted(glob.glob(os.path.join(csv_dir, "random_IR_params_*.csv")))
    if not csv_paths:
        print(f"[WARN] No CSVs matching random_IR_params_*.csv in {csv_dir}")
        sys.exit(0)

    # Allow CLI overrides of band limits (mandatory)
    defaults = dict(BASE_DEFAULTS)
    defaults["FMIN"] = float(args.fmin)
    defaults["FMAX"] = float(args.fmax)

    # Process
    summary = []
    for p in csv_paths:
        try:
            df, meta = compute_single(p, defaults)
            import re
            base = os.path.splitext(os.path.basename(p))[0]
            out_base = re.sub(r'^random_IR_params_', 'random_IR_identifiedModes_', base)
            out_csv = os.path.join(out_dir, f"{out_base}.csv")
            df.to_csv(out_csv, index=False)
            summary.append(dict(source=base, rows=len(df), out=out_csv))
            print(f"[OK] {base} -> {out_csv} ({len(df)} rows)")
        except Exception as e:
            print(f"[FAIL] {os.path.basename(p)}: {e}", file=sys.stderr)

    # Also write a small JSON index for convenience
    index_path = os.path.join(out_dir, "_index_TaskB.json")
    with open(index_path, "w", encoding="utf-8") as f:
        import json
        json.dump(summary, f, indent=2)
    print(f"\nWrote: {index_path}")

if __name__ == "__main__":
    main()
