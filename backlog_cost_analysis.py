#!/usr/bin/env python3
# oos_backlog_analysis.py
# Analyze oos_backlog against N and epsilon (with optional faceting by oos_weight/mode)

import argparse, ast, math, re, sys, warnings
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- helpers ----------
def normalize_colname(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

SYNONYMS = {
    "N": ["n", "num_samples", "samples", "n_value", "n_class", "n_classes"],
    "oos_backlog": ["oos_backlog", "backlog", "oos_backlog_cost", "backlog_cost"],
    "oos_weight": ["oos_weight", "weight", "mix_weight", "mixture_weight"],
    "mode": ["mode"],
    "epsilon": ["epsilon", "eps", "e", "ϵ"],
}

def guess_columns(df: pd.DataFrame) -> Dict[str, str]:
    nrm = {c: normalize_colname(c) for c in df.columns}
    inv = {}
    for orig, key in nrm.items(): inv.setdefault(key, []).append(orig)
    chosen = {}
    for tgt, opts in SYNONYMS.items():
        for k in opts:
            if k in inv: chosen[tgt] = inv[k][0]; break
    return chosen

def first_number(x):
    if x is None or (isinstance(x, float) and math.isnan(x)): return np.nan
    if isinstance(x, (int, float, np.number)): return float(x)
    s = str(x).strip()
    try: return float(s)
    except: pass
    try:
        v = ast.literal_eval(s)
        if isinstance(v, dict) and v: return float(next(iter(v.values())))
        if isinstance(v, (list, tuple, set)) and v: return float(next(iter(v)))
        if isinstance(v, (int, float)): return float(v)
    except: pass
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else np.nan

def coerce_num(s: pd.Series) -> pd.Series:
    y = pd.to_numeric(s, errors="coerce")
    if y.isna().mean() > 0.2: y = s.apply(first_number)
    return y.astype(float)

def parse_epsilon_scalar(eps_cell, reduce="mean") -> float:
    """
    eps_cell may be a scalar or a dict-like string: '{0: 10, 1: 30}'
    Reduce across regimes with mean|max|min|sum.
    """
    if isinstance(eps_cell, (int, float, np.number)): vals = [float(eps_cell)]
    else:
        s = str(eps_cell).strip()
        try:
            v = ast.literal_eval(s)
            if isinstance(v, dict): vals = [float(x) for x in v.values()]
            elif isinstance(v, (list, tuple, set)): vals = [float(x) for x in v]
            else: vals = [first_number(s)]
        except: vals = [first_number(s)]
    vals = [x for x in vals if x is not None and not math.isnan(x)]
    if not vals: return np.nan
    if reduce == "max": return float(np.max(vals))
    if reduce == "min": return float(np.min(vals))
    if reduce == "sum": return float(np.sum(vals))
    return float(np.mean(vals))  # default mean

def load_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

def print_head(df: pd.DataFrame, chosen: Dict[str, str]):
    print("\n=== Columns ===", list(df.columns), sep="\n")
    print("\n=== Head(8) ===")
    print(df.head(8).to_string(index=False))
    print("\n=== Chosen mapping (target -> original) ===")
    for k in ("N","epsilon","oos_backlog","oos_weight","mode"):
        print(f"{k:>12}: {chosen.get(k, '<NOT FOUND>')}")

# ---------- plotting ----------
def make_plots(agg: pd.DataFrame, time_title: str, facet_by_mode: bool):
    if agg.empty:
        print("\n[WARN] Nothing to plot after cleaning.")
        return

    # 3D scatter: axes (N, epsilon_scalar, oos_weight), color by oos_backlog
    print("\n=== Aggregated values used in plots ===")
    print(agg.to_string(index=False))

    if facet_by_mode:
        modes = list(agg["mode"].unique())
    else:
        modes = [None]

    for m in modes:
        sub = agg if m is None else agg[agg["mode"]==m]
        if sub.empty: continue
        ttl = "oos_backlog vs N–epsilon" + (f" | mode={m}" if m else "")

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(sub["N_c"], sub["epsilon_c"], sub["oos_weight_c"],
                        c=sub["oos_backlog"], cmap="viridis", s=120, depthshade=True, edgecolor="k")
        ax.set_xlabel("N"); ax.set_ylabel("epsilon"); ax.set_zlabel("oos_weight")
        cb = plt.colorbar(sc, ax=ax, pad=0.1); cb.set_label("oos_backlog")
        ax.set_title(ttl)
        plt.tight_layout(); plt.show()

        # Heatmaps: fix oos_weight (up to 3)
        uniq_w = np.array(sorted(sub["oos_weight_c"].unique()))
        if len(uniq_w) > 3:
            idx = np.linspace(0, len(uniq_w)-1, 3).round().astype(int)
            uniq_w = uniq_w[idx]

        for w in uniq_w:
            sl = sub[sub["oos_weight_c"]==w]
            pv = sl.pivot_table(index="N_c", columns="epsilon_c", values="oos_backlog", aggfunc="mean")
            pv = pv.sort_index().sort_index(axis=1)
            print(f"\n--- Heatmap grid (oos_backlog) for oos_weight={w:g}" + (f", mode={m}" if m else "") + " ---")
            if pv.size: print(pv.to_string(float_format=lambda x: f"{x:6.2f}"))
            else: print("[empty]")

            fig, ax = plt.subplots(figsize=(6,4.5))
            im = ax.imshow(pv.values, aspect="auto")
            ax.set_title(f"oos_backlog heatmap | oos_weight={w:g}" + (f", mode={m}" if m else ""))
            ax.set_xlabel("epsilon"); ax.set_ylabel("N")
            ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels([f"{c:g}" for c in pv.columns], rotation=45, ha="right")
            ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels([f"{r:g}" for r in pv.index])
            cb = fig.colorbar(im, ax=ax); cb.set_label("oos_backlog")
            plt.tight_layout(); plt.show()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", default="results_RS_vs_noRS.xlsx", help="Path to results Excel")
    ap.add_argument("--mode", choices=["RS","noRS","both"], default="both", help="Filter by mode")
    ap.add_argument("--eps_reduce", choices=["mean","max","min","sum"], default="mean",
                    help="How to reduce multi-state epsilon dict to scalar")
    ap.add_argument("--round", type=int, default=6, help="Rounding digits for class merging")
    args = ap.parse_args()

    df = load_excel(args.excel)
    chosen = guess_columns(df)
    print_head(df, chosen)

    # Ensure required columns exist
    req = ["N","epsilon","oos_backlog"]
    missing = [r for r in req if r not in chosen]
    if missing:
        print(f"\n[ERROR] Missing required columns: {missing}")
        print("Please rename your columns or update SYNONYMS in this script.")
        sys.exit(2)

    # Rename to canonical
    rename = {chosen["N"]:"N", chosen["epsilon"]:"epsilon_raw", chosen["oos_backlog"]:"oos_backlog"}
    if "oos_weight" in chosen: rename[chosen["oos_weight"]] = "oos_weight"
    if "mode" in chosen: rename[chosen["mode"]] = "mode"
    df = df.rename(columns=rename)

    # Parse epsilon -> scalar; coerce numbers
    df["epsilon"] = df["epsilon_raw"].apply(lambda x: parse_epsilon_scalar(x, args.eps_reduce))
    df["N"] = coerce_num(df["N"])
    df["oos_backlog"] = coerce_num(df["oos_backlog"])
    if "oos_weight" in df.columns: df["oos_weight"] = coerce_num(df["oos_weight"])
    else: df["oos_weight"] = 0.0  # single slice if absent
    if "mode" not in df.columns: df["mode"] = "both"

    # Filter mode
    if args.mode != "both":
        df = df[df["mode"] == args.mode]

    # Drop NA rows
    clean = df.dropna(subset=["N","epsilon","oos_backlog","oos_weight"]).copy()
    if clean.empty:
        print("\n[ERROR] No rows after cleaning. Inspect your input file.")
        sys.exit(3)

    # Class rounding (helps merge float fuzz)
    r = max(0, int(args.round))
    clean["N_c"] = clean["N"].round(r)
    clean["epsilon_c"] = clean["epsilon"].round(r)
    clean["oos_weight_c"] = clean["oos_weight"].round(r)
    if "mode" not in clean.columns: clean["mode"] = "both"

    # Aggregate duplicates
    agg = (clean.groupby(["mode","N_c","epsilon_c","oos_weight_c"], as_index=False)
                 .agg(oos_backlog=("oos_backlog","mean"),
                      cnt=("oos_backlog","size")))

    print("\n=== Describe (clean) ===")
    print(clean[["N","epsilon","oos_backlog","oos_weight"]].describe())

    # Plots + printed grids
    make_plots(agg, "oos_backlog", facet_by_mode=(args.mode=="both"))

if __name__ == "__main__":
    main()
