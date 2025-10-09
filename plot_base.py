# plot_base.py
# Visualization for RS vs noRS experiments:
# - Histograms of total and component costs (per-sample)
# - Overlay histograms (RS vs noRS)
# - Mean ± std bar charts (RS vs noRS)
# - CV heatmaps (K=2) and CV curves (K=1)

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------- CLI ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results_RS_vs_noRS.xlsx",
                   help="Summary results file produced by rs_base.py")
    p.add_argument("--oos", default="oos_per_sample.parquet",
                   help="Per-sample OoS parquet produced by rs_base.py")
    p.add_argument("--cv", default="cv_traces.xlsx",
                   help="Optional CV trace file produced by rs_base.py")
    p.add_argument("--outdir", default="plots",
                   help="Directory to save plots")
    p.add_argument("--bins", type=int, default=40, help="Histogram bins")
    return p.parse_args()

# ----------------------- util / I/O helpers -------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def try_read_parquet(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        # fallback to CSV with same base name
        alt = os.path.splitext(path)[0] + ".csv"
        if os.path.exists(alt):
            return pd.read_csv(alt)
        return None

def try_read_excel(path):
    if os.path.exists(path):
        try:
            return pd.read_excel(path)
        except Exception:
            pass
    # also allow CSV with same base
    alt = os.path.splitext(path)[0] + ".csv"
    if os.path.exists(alt):
        try:
            return pd.read_csv(alt)
        except Exception:
            pass
    return None

# ------------------------- plotting ---------------------------
def save_hist(series, title, xlabel, outfile, bins=40):
    plt.figure()
    series.plot.hist(bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def save_overlay_hist(s1, s2, labels, title, xlabel, outfile, bins=40):
    # common bin edges for fair overlay
    lo = np.min([s1.min(), s2.min()])
    hi = np.max([s1.max(), s2.max()])
    edges = np.linspace(lo, hi, bins + 1)

    plt.figure()
    plt.hist(s1, bins=edges, alpha=0.55, density=False, label=labels[0])
    plt.hist(s2, bins=edges, alpha=0.55, density=False, label=labels[1])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def save_mean_std_bars(df_grp, title, ylabel, outfile):
    # df_grp columns must include: mode, metric; and index-specific info embedded in title/outfile
    modes = ["noRS", "RS"]
    g = df_grp.set_index("mode").reindex(modes)
    means = g["mean"].values
    stds  = g["std"].values

    plt.figure()
    x = np.arange(len(modes))
    plt.bar(x, means, yerr=stds, capsize=6)
    plt.xticks(x, modes)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def save_cv_heatmap(cv_sub, title, outfile):
    table = cv_sub.pivot(index="eps_state0", columns="eps_state1", values="avg_val_cost")
    plt.figure()
    plt.imshow(table.values, aspect="auto", origin="lower")
    plt.xticks(range(table.shape[1]), table.columns)
    plt.yticks(range(table.shape[0]), table.index)
    plt.colorbar(label="Mean CV cost")
    plt.title(title)
    plt.xlabel("eps_state1")
    plt.ylabel("eps_state0")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def save_cv_curve(cv_sub, title, outfile):
    # For K=1: expect column "eps_state0"
    agg = (cv_sub
           .groupby("eps_state0", as_index=False)["avg_val_cost"]
           .mean()
           .sort_values("eps_state0"))
    plt.figure()
    plt.plot(agg["eps_state0"], agg["avg_val_cost"], marker="o")
    plt.title(title)
    plt.xlabel("epsilon")
    plt.ylabel("Mean CV cost")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

# ---------------------------- main ----------------------------
def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    # Load data
    oos = try_read_parquet(args.oos)
    results = try_read_excel(args.results)
    cv = try_read_excel(args.cv)

    if results is None:
        print(f"[WARN] Could not read results file: {args.results}")
    if oos is None:
        print(f"[WARN] Could not read per-sample OoS file: {args.oos}")
    if (results is None) and (oos is None) and (cv is None):
        print("[ERROR] Nothing to plot.")
        return

    # ----------------- 1) Per-sample histograms -----------------
    if oos is not None and all(c in oos.columns for c in ["mode","N","oos_weight","total","purchase","inventory","backlog"]):
        # Individual histograms per (mode, N, weight)
        for (mode, N, w), g in oos.groupby(["mode", "N", "oos_weight"]):
            base = f"{outdir}/N{N}_w{w}_{mode}"
            save_hist(g["total"], f"Total OoS Cost — mode={mode}, N={N}, mix={w}",
                      "Total cost", f"{base}_hist_total.png", bins=args.bins)
            for comp in ["purchase", "inventory", "backlog"]:
                save_hist(g[comp], f"{comp.capitalize()} OoS — mode={mode}, N={N}, mix={w}",
                          f"{comp.capitalize()} cost", f"{base}_hist_{comp}.png", bins=args.bins)

        # Overlay histograms: RS vs noRS per (N, weight)
        for (N, w), gg in oos.groupby(["N", "oos_weight"]):
            if set(gg["mode"].unique()) >= {"RS", "noRS"}:
                a = gg[gg["mode"]=="noRS"]
                b = gg[gg["mode"]=="RS"]
                base = f"{outdir}/N{N}_w{w}_overlay"
                save_overlay_hist(a["total"], b["total"], ["noRS", "RS"],
                                  f"Total OoS Cost — RS vs noRS (N={N}, mix={w})",
                                  "Total cost", f"{base}_hist_total_overlay.png", bins=args.bins)
                for comp in ["purchase","inventory","backlog"]:
                    save_overlay_hist(a[comp], b[comp], ["noRS","RS"],
                                      f"{comp.capitalize()} OoS — RS vs noRS (N={N}, mix={w})",
                                      f"{comp.capitalize()} cost", f"{base}_hist_{comp}_overlay.png", bins=args.bins)

    # ----------------- 2) Mean ± std bar charts -----------------
    if results is not None and {"mode","N","oos_weight","oos_mean","oos_std"}.issubset(results.columns):
        for (N, w), g in results.groupby(["N","oos_weight"]):
            # calculate mean of means if multiple rows exist (should typically be one per mode)
            agg = (g.groupby("mode")
                    .agg(mean=("oos_mean","mean"), std=("oos_std","mean"))
                    .reset_index())
            base = f"{outdir}/N{N}_w{w}_bars"
            save_mean_std_bars(
                agg,
                title=f"OoS Mean ± Std — RS vs noRS (N={N}, mix={w})",
                ylabel="OoS total cost",
                outfile=f"{base}.png"
            )

        # Overall delta table as a CSV for quick inspection
        pivot = (results.pivot_table(index=["N","oos_weight"], columns="mode", values="oos_mean")
                 .reset_index())
        if {"RS","noRS"}.issubset(pivot.columns):
            pivot["delta_abs"] = pivot["noRS"] - pivot["RS"]
            pivot["delta_pct"] = 100.0 * pivot["delta_abs"] / pivot["noRS"]
            pivot.to_csv(f"{outdir}/comparison_rs_vs_nors.csv", index=False)

    # ----------------- 3) CV visualizations -----------------
    if cv is not None and {"K","mode","N","avg_val_cost"}.issubset(cv.columns):
        # K=2: heatmaps over eps_state0 x eps_state1
        if {"eps_state0","eps_state1"}.issubset(cv.columns):
            k2 = cv[cv["K"] == 2]
            if not k2.empty:
                agg = (k2.groupby(["mode","N","eps_state0","eps_state1"], as_index=False)
                          ["avg_val_cost"].mean())
                for (mode, N), g in agg.groupby(["mode","N"]):
                    save_cv_heatmap(g, title=f"CV heatmap (K=2) — mode={mode}, N={N}",
                                    outfile=f"{outdir}/cv_heatmap_K2_{mode}_N{N}.png")

        # K=1: curves of avg CV cost vs epsilon
        if "eps_state0" in cv.columns:
            k1 = cv[cv["K"] == 1]
            if not k1.empty:
                for (mode, N), g in k1.groupby(["mode","N"]):
                    save_cv_curve(g, title=f"CV curve (K=1) — mode={mode}, N={N}",
                                  outfile=f"{outdir}/cv_curve_K1_{mode}_N{N}.png")

    print(f"Done. Plots saved to: {outdir}")

if __name__ == "__main__":
    main()
