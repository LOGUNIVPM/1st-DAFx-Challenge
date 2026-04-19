#!/usr/bin/env python3
"""
TaskB: identify plate modal parameters from impulse-response audio.

Usage:
  python TaskB/baseline.py --folder <folder_name> --fmin <Hz> --fmax <Hz> [--root <path>]

Description:
  For each random_IR_XXXX.npz in the folder, this script:
    - loads the unnormalized displacement IR from the .npz (the sibling
      .wav is peak-normalized and not suitable: it discards the b_m
      amplitude scale),
    - picks spectral peaks in [FMIN, FMAX] on |H(f)| (parabolic sub-bin refinement),
    - estimates sigma via the half-power-bandwidth rule,
    - estimates the modal gain b_m by probing the imaginary part of H at the
      peak via Eq. 18 of the paper,
  and writes experiment_results_TaskB/random_IR_identifiedModes_XXXX.csv with
  columns f0_ident, sigma_ident, gain_ident.

  No plate parameters are read and no IR is regenerated: per the challenge
  rules, the only input is the supplied IR.

Options:
  --folder   Folder containing the random_IR_*.npz files.
  --fmin     Lower bound (Hz) of the modal frequency band to identify.
  --fmax     Upper bound (Hz) of the modal frequency band to identify.
  --root     (Optional) Root path (default: current directory).

Example:
  python TaskB/baseline.py --folder random-IR-10-1.0s --fmin 50 --fmax 10000
"""

import argparse
import glob
import os
import re
import sys
import numpy as np
import pandas as pd


# ---------------------------- IR loader ----------------------------
def load_ir(path):
    """Load an IR from a random_IR_XXXX.npz file.
    Returns (ir_float64, sample_rate_int).

    The .npz is the official input for Task B (unnormalized displacement
    plus metadata). The sibling .wav is peak-normalized and loses the
    absolute amplitude — which carries the b_m scale — so it is not
    accepted here.
    """
    if not path.lower().endswith('.npz'):
        raise ValueError(
            f"Task B expects random_IR_XXXX.npz input (unnormalized IR). "
            f"Got {path!r}. The peak-normalized .wav cannot be used because "
            f"it discards the b_m amplitude scale.")
    with np.load(path) as data:
        ir = np.asarray(data["ir"], dtype=np.float64)
        sr = int(data["sample_rate"])
    return ir, sr

# ---------------------------- Helpers ----------------------------
def get_val(row, keys, default):
    """Fetch value from a pandas Series using first matching key (case-insensitive)."""
    row_lower = {k.lower(): v for k, v in row.items()}
    for k in keys:
        if k.lower() in row_lower and pd.notna(row_lower[k.lower()]):
            return row_lower[k.lower()]
    return default

def compute_single(ir_path, FMIN, FMAX, PROM_DB, MIN_DIST_HZ, PROM_WIN_HZ=None):
    """Identify modes directly from an impulse response.

    Loads the IR (displacement, unnormalized), takes its FFT, and runs the
    same two-stage peak-picking + Eq.-18 gain-inversion identification that
    the original baseline used — no plate parameters, no regeneration.
    """
    # 1) Load the IR
    ir, SR = load_ir(ir_path)
    N = len(ir)
    if N < 3:
        raise ValueError(f"{ir_path}: IR too short ({N} samples).")
    k = 1.0 / SR
    df = SR / N

    # Handle PROM_WIN_HZ default
    if PROM_WIN_HZ is None:
        PROM_WIN_HZ = MIN_DIST_HZ

    # 2) One-sided spectrum (rfft == fft[:N//2+1] on real inputs, faster)
    spec = np.fft.rfft(ir)
    freq = np.fft.rfftfreq(N, d=k)
    mag = np.abs(spec)
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

    # Sort identified modes by frequency and emit the three-column CSV.
    results = sorted(results, key=lambda r: r["f0_ident"]) if results else []
    rows = [dict(
        f0_ident=r["f0_ident"],
        sigma_ident=r["sigma"],
        gain_ident=r["gain_ident"],
    ) for r in results]

    out_df = pd.DataFrame(rows, columns=["f0_ident", "sigma_ident", "gain_ident"])
    meta = dict(ir_source=os.path.basename(ir_path))
    return out_df, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True,
                    help="Folder containing random_IR_*.npz files.")
    ap.add_argument("--root", default=".",
                    help="Path to the project root (default: current directory).")
    ap.add_argument("--fmin", type=float, required=True,
                    help="Lower bound of frequency band (Hz).")
    ap.add_argument("--fmax", type=float, required=True,
                    help="Upper bound of frequency band (Hz).")
    ap.add_argument("--prom-db", type=float, default=6.0, dest="prom_db",
                    help="Minimum peak prominence in dB (default: 6).")
    ap.add_argument("--min-dist-hz", type=float, default=2.0, dest="min_dist_hz",
                    help="Minimum spacing between peaks, Hz (default: 2).")
    ap.add_argument("--prom-win-hz", type=float, default=None, dest="prom_win_hz",
                    help="Half-window for prominence, Hz (default: same as --min-dist-hz).")
    args = ap.parse_args()

    data_dir = os.path.join(args.root, args.folder)
    if not os.path.isdir(data_dir):
        print(f"[ERROR] Folder not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.join(args.root, "experiment_results_TaskB")
    os.makedirs(out_dir, exist_ok=True)

    # Official input: the scientific .npz (unnormalized IR + metadata).
    ir_paths = sorted(glob.glob(os.path.join(data_dir, "random_IR_[0-9]*.npz")))
    if not ir_paths:
        print(f"[WARN] No random_IR_*.npz in {data_dir}. "
              f"(The peak-normalized .wav cannot be used for Task B: it "
              f"loses the b_m amplitude scale.)", file=sys.stderr)
        sys.exit(0)

    # Process
    summary = []
    for p in ir_paths:
        try:
            df, meta = compute_single(
                p, args.fmin, args.fmax,
                args.prom_db, args.min_dist_hz, args.prom_win_hz,
            )
            base = os.path.splitext(os.path.basename(p))[0]
            out_base = re.sub(r'^random_IR_', 'random_IR_identifiedModes_', base)
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
