#!/usr/bin/env python3
# computation_time_plot_debug.py
# Debug-friendly loader + coercer + prints + plots for (oos_weight, N, epsilon) -> time_min

import argparse
import ast
import math
import re
import sys
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def normalize_colname(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


SYNONYMS = {
    "oos_weight": ["oos_weight", "oos_wt", "oos", "oosweight", "weight_oos", "oos__weight", "oos__wt"],
    "N": ["n", "num_samples", "samples", "n_class", "n_classes", "n_value", "n_grid"],
    "epsilon": ["epsilon", "eps", "e", "ϵ", "epsilon_value", "eps_value"],
    "time_min": [
        "time_min", "time_minute", "time_minutes", "runtime_min", "run_time_min",
        "elapsed_min", "computation_time_min", "comp_time_min", "time__min",
        "time_sec", "runtime_sec", "elapsed_sec", "computation_time", "runtime",
        "time", "total_time", "solve_time", "wall_time"
    ],
}


def guess_columns(df: pd.DataFrame) -> Dict[str, str]:
    norm_map = {c: normalize_colname(c) for c in df.columns}
    inv_map = {}
    for orig, nrm in norm_map.items():
        inv_map.setdefault(nrm, []).append(orig)
    chosen: Dict[str, str] = {}
    for target, options in SYNONYMS.items():
        for opt in options:
            if opt in inv_map:
                chosen[target] = inv_map[opt][0]
                break
    return chosen


def _first_number(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    try:
        return float(s)
    except Exception:
        pass
    try:
        v = ast.literal_eval(s)
        if isinstance(v, dict) and len(v):
            return float(next(iter(v.values())))
        if isinstance(v, (list, tuple, set)) and len(v):
            return float(next(iter(v)))
        if isinstance(v, (int, float)):
            return float(v)
    except Exception:
        pass
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return np.nan
    return np.nan


def coerce_numeric(series: pd.Series) -> pd.Series:
    y = pd.to_numeric(series, errors="coerce")
    if y.isna().mean() > 0.2:
        y = series.apply(_first_number)
    return y.astype(float)


def maybe_convert_time_to_min(time_s: pd.Series, original_name: str) -> Tuple[pd.Series, str]:
    name_nrm = normalize_colname(original_name)
    looks_like_sec = any(k in name_nrm for k in ["_sec", "seconds"])
    series = time_s.copy()
    unit_note = "min"
    if looks_like_sec:
        series = series / 60.0
        unit_note = "min (converted from sec)"
    else:
        med = series.dropna().median() if len(series.dropna()) else np.nan
        if pd.notna(med) and 120 < med < 1e6 and "min" not in name_nrm:
            series = series / 60.0
            unit_note = "min (heuristic from sec)"
    return series, unit_note


def load_excel(path: str, sheet: Optional[str]) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    if sheet is None:
        frames = []
        for sh in xl.sheet_names:
            df = xl.parse(sh)
            df["__sheet__"] = sh
            frames.append(df)
        out = pd.concat(frames, ignore_index=True)
    else:
        out = xl.parse(sheet)
        out["__sheet__"] = sheet
    return out


def debug_print(df: pd.DataFrame, chosen: Dict[str, str]) -> None:
    print("\n=== Columns ===")
    print(list(df.columns))
    print("\n=== Head(8) ===")
    print(df.head(8).to_string(index=False))
    print("\n=== Chosen mapping (target -> original) ===")
    for k in ("oos_weight", "N", "epsilon", "time_min"):
        print(f"{k:>12}: {chosen.get(k, '<NOT FOUND>')}")
    print("\n=== NA counts (raw) ===")
    print(df.isna().sum().sort_values(ascending=False).head(20))
    print("\n=== Unique sample values (first up to 10) ===")
    for k, orig in chosen.items():
        try:
            print(f"- {k} ({orig}):", df[orig].unique()[:10])
        except Exception:
            pass


def make_plots(agg: pd.DataFrame, time_unit_note: str) -> None:
    if agg.empty:
        print("\n[WARN] No rows to plot after cleaning/aggregation.")
        return

    # Print all values used for plotting
    print("\n=== Aggregated values (for plotting) ===")
    print(agg.to_string(index=False))

    # 3D scatter
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
        agg["oos_weight_c"], agg["N_c"], agg["epsilon_c"],
        c=agg["time_min"], cmap="viridis", s=120, depthshade=True, edgecolor="k"
    )
    ax.set_xlabel("oos_weight")
    ax.set_ylabel("N")
    ax.set_zlabel("epsilon")
    cb = plt.colorbar(sc, ax=ax, pad=0.1)
    cb.set_label(f"time_min [{time_unit_note}]")
    ax.set_title("OOS weight–N–epsilon vs time_min (mean over duplicates)")
    plt.tight_layout()
    plt.show()

    # Heatmaps per oos_weight (axes: N x epsilon) + print their cell values
    uniq_w = np.array(sorted(agg["oos_weight_c"].unique()))
    if len(uniq_w) > 3:
        idx = np.linspace(0, len(uniq_w) - 1, 3).round().astype(int)
        uniq_w = uniq_w[idx]

    for w in uniq_w:
        sub = agg[agg["oos_weight_c"] == w].copy()
        pivot = sub.pivot_table(index="N_c", columns="epsilon_c",
                                values="time_min", aggfunc="mean")
        pivot = pivot.sort_index().sort_index(axis=1)

        print(f"\n--- Heatmap values for oos_weight={w:g} ---")
        if pivot.size:
            print(pivot.to_string(float_format=lambda x: f"{x:6.2f}"))
        else:
            print("[empty slice]")

        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_title(f"time_min heatmap | oos_weight={w:g}")
        ax.set_xlabel("epsilon")
        ax.set_ylabel("N")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels([f"{c:g}" for c in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels([f"{r:g}" for r in pivot.index])
        
        # Add value annotations to each cell
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.values[i, j]
                if not np.isnan(value):
                    # Choose text color based on cell color (dark background -> white text, light -> black)
                    text_color = 'white' if value > np.nanmean(pivot.values) else 'black'
                    ax.text(j, i, f'{value:.1f}', ha='center', va='center', 
                           color=text_color, fontweight='bold', fontsize=10)
        
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(f"time_min [{time_unit_note}]")
        plt.tight_layout()
        plt.show()


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    p = argparse.ArgumentParser()
    p.add_argument("--excel", default="results_RS_vs_noRS-00.xlsx", help="Path to Excel file")
    p.add_argument("--sheet", default=None, help="Sheet name (default: all sheets concatenated)")
    p.add_argument("--round", type=int, default=6, help="Rounding digits for class merging")
    args = p.parse_args()

    try:
        df = load_excel(args.excel, args.sheet)
    except Exception as e:
        print(f"[ERROR] Failed to load '{args.excel}': {e}")
        sys.exit(1)

    chosen = guess_columns(df)
    debug_print(df, chosen)

    missing = [k for k in ("oos_weight", "N", "epsilon", "time_min") if k not in chosen]
    if missing:
        print(f"\n[ERROR] Could not detect required columns: {missing}")
        print("Tip: manually rename, e.g.:")
        print("  df = df.rename(columns={'OOS_Weight':'oos_weight','time (min)':'time_min','N':'N','epsilon':'epsilon'})")
        sys.exit(2)

    rename_map = {
        chosen["oos_weight"]: "oos_weight",
        chosen["N"]: "N",
        chosen["epsilon"]: "epsilon",
        chosen["time_min"]: "time_min_raw",
    }
    df = df.rename(columns=rename_map)

    for c in ("oos_weight", "N", "epsilon"):
        df[c] = coerce_numeric(df[c])

    time_series = coerce_numeric(df["time_min_raw"])
    time_series, unit_note = maybe_convert_time_to_min(time_series, chosen["time_min"])
    df["time_min"] = time_series

    print("\n=== After coercion (describe) ===")
    print(df[["oos_weight", "N", "epsilon", "time_min"]].describe(include="all"))

    clean = df.dropna(subset=["oos_weight", "N", "epsilon", "time_min"]).copy()
    print(f"\nRows before dropna: {len(df)} | after dropna: {len(clean)}")
    if clean.empty:
        print("\n[ERROR] Dataset empty after coercion. Check raw values above.")
        for c in ("oos_weight", "N", "epsilon", "time_min_raw"):
            print(f"\nSample raw values for {c}:")
            print(df[c].astype(str).head(10).to_string(index=False))
        sys.exit(3)

    r = max(0, int(args.round))
    clean["oos_weight_c"] = clean["oos_weight"].round(r)
    clean["N_c"] = clean["N"].round(r)
    clean["epsilon_c"] = clean["epsilon"].round(r)

    agg = (
        clean.groupby(["oos_weight_c", "N_c", "epsilon_c"], as_index=False)
             .agg(time_min=("time_min", "mean"))
    )
    if agg.empty:
        print("\n[ERROR] Aggregated grid is empty. Try smaller rounding (e.g., --round 3).")
        print("Unique oos_weight:", sorted(clean['oos_weight'].unique())[:10])
        print("Unique N:", sorted(clean['N'].unique())[:10])
        print("Unique epsilon:", sorted(clean['epsilon'].unique())[:10])
        sys.exit(4)

    print("\n=== Aggregated sample (head) ===")
    print(agg.head(15).to_string(index=False))

    make_plots(agg, unit_note)


if __name__ == "__main__":
    main()
