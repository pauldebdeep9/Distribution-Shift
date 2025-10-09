# visualization_v5.py
# Two 3x3 figures:
#  A) RS vs noRS values (zoomed y-axis ~±10% band) with y-axis ticks scaled by ÷1e5
#  B) % reduction = 100 * (noRS - RS) / noRS (no bar annotations)
#
# Run directly in VSCode (F5). Requires: pandas, numpy, matplotlib.

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---------------- Config ----------------
WRAP_METRIC_LABEL = True
ANNOTATE_VALUES   = False     # annotate bars in Fig A with absolute values (2 decimals)
ANNOTATE_PCT      = False    # DO NOT annotate bars in Fig B (per user request)
VALUE_AXIS_SCALE  = 1e5      # y-tick scaling for Fig A (divide by this)
# ---------------------------------------

def _nrm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s

def _find_col(df: pd.DataFrame, candidates) -> str:
    cols = {_nrm(c): c for c in df.columns}
    for cand in candidates:
        k = _nrm(cand)
        if k in cols:
            return cols[k]
    # partial fallback
    for cand in candidates:
        key = _nrm(cand)
        for k, orig in cols.items():
            if key in k:
                return orig
    raise KeyError(f"Could not find any of {candidates}. Available: {list(df.columns)}")

def _norm_variant(v) -> str:
    s = _nrm(str(v))
    if "no" in s and "rs" in s: return "noRS"
    if s in {"nors","no-rs","no rs","baseline","plain"}: return "noRS"
    if "rs" in s: return "RS"
    return "RS" if s in {"1","true","yes"} else "noRS" if s in {"0","false","no"} else str(v)

def _wrap_label(text: str) -> str:
    if not WRAP_METRIC_LABEL:
        return text
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 <= 18:
            line = (line + " " + w).strip()
        else:
            out.append(line); line = w
    out.append(line)
    return "\n".join(out[:2])

def _fmt_vals_scaled(v, pos=None):  # Fig A y-ticks: divide by 1e5, show 2 decimals
    return f"{(v / VALUE_AXIS_SCALE):,.2f}"

def _fmt_pct(v, pos=None):          # Fig B y-ticks: percent with 2 decimals
    return f"{v:.2f}%"

def _zoom_limits_10pct(vals: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return (0.0, 1.0)
    vmin, vmax = float(np.min(finite)), float(np.max(finite))
    if vmin >= 0:
        lo, hi = 0.9 * vmin, 1.1 * vmax
    elif vmax <= 0:
        lo, hi = 1.1 * vmin, 0.9 * vmax
    else:
        lo, hi = vmin - 0.1 * abs(vmin), vmax + 0.1 * abs(vmax)
    if abs(hi - lo) < 1e-9:
        span = max(1.0, abs(hi) * 0.1)
        lo, hi = lo - span / 2, hi + span / 2
    return lo, hi

def _percent_reduction(rs_vals: np.ndarray, nors_vals: np.ndarray) -> np.ndarray:
    """100 * (noRS - RS) / noRS ; positive => RS reduces the metric."""
    rs = np.asarray(rs_vals, dtype=float)
    nr = np.asarray(nors_vals, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (nr - rs) / nr * 100.0
    return pct

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(here, "results_RS_vs_noRS.xlsx")
    df = pd.read_excel(file_path)

    # Identify columns (your sheet headers include: mode, N, oos_weight, obj, oos_mean, purchase, oos_inv, oos_backlog, ...)
    col_variant = _find_col(df, ["mode","variant","rs_flag","rs","method","model","algo"])
    col_N       = _find_col(df, ["N","n","samples","num-samples"])
    col_w       = _find_col(df, ["oos_weight","oos-weight","weight","oos w"])

    # Rename metrics per spec
    rename_map = {
        _find_col(df, ["obj"]):         "objective",
        _find_col(df, ["oos_mean"]):    "Total",
        _find_col(df, ["purchase"]):    "Proc",
        _find_col(df, ["oos_inv"]):     "Storage",
        _find_col(df, ["oos_backlog"]): "Backlog",
    }
    df = df.rename(columns=rename_map)

    metrics = [
        "objective",
        "Total",
        "Proc",
        "Storage",
        "Backlog",
    ]

    # Normalize RS/noRS
    df["_variant"] = df[col_variant].apply(_norm_variant)

    # Use first 3 unique values for each axis (as before)
    uniq_w = sorted(pd.unique(df[col_w]))[:3]
    uniq_N = sorted(pd.unique(df[col_N]))[:3]

    x = np.arange(len(metrics))
    width = 0.38

    # -------------------- Figure A: values (zoomed y-axis, y-ticks ÷1e5) --------------------
    figA, axesA = plt.subplots(3, 3, figsize=(17, 11))
    figA.suptitle(f"Values (RS vs noRS) — zoomed y-axis (±10%), y-ticks ÷ {int(VALUE_AXIS_SCALE):,}", fontsize=16, y=0.98)

    legend_handles = None
    for r, w_val in enumerate(uniq_w):
        for c, n_val in enumerate(uniq_N):
            ax = axesA[r, c]
            sub = df[(df[col_w] == w_val) & (df[col_N] == n_val)]
            if sub.empty:
                ax.axis("off"); ax.set_title(f"w={w_val}, N={n_val}\n(no data)"); continue

            agg = sub.groupby("_variant")[metrics].mean()
            for v in ("RS","noRS"):
                if v not in agg.index:
                    agg.loc[v] = [np.nan] * len(metrics)
            agg = agg.loc[["RS","noRS"]]

            rs_vals, nors_vals = agg.iloc[0].values, agg.iloc[1].values

            b1 = ax.bar(x - width/2, rs_vals,   width=width, label="RS")
            b2 = ax.bar(x + width/2, nors_vals, width=width, label="noRS",
                        hatch="///", edgecolor="black", linewidth=0.5)

            if legend_handles is None:
                legend_handles = (b1[0], b2[0])

            lo, hi = _zoom_limits_10pct(np.concatenate([rs_vals, nors_vals]))
            ax.set_ylim(lo, hi)
            ax.yaxis.set_major_formatter(FuncFormatter(_fmt_vals_scaled))
            if c == 0:
                ax.set_ylabel(f"Value (÷{int(VALUE_AXIS_SCALE):,})")
            ax.set_title(f"w={w_val}, N={n_val}")
            ax.set_xticks(x)
            ax.set_xticklabels([_wrap_label(m) for m in metrics], fontsize=9)
            ax.grid(axis="y", linestyle=":", alpha=0.4)

            if ANNOTATE_VALUES:
                for bars in (b1, b2):
                    for rect in bars:
                        h = rect.get_height()
                        if np.isfinite(h):
                            ax.annotate(f"{h:,.2f}",
                                        xy=(rect.get_x() + rect.get_width()/2, h),
                                        xytext=(0, 3),
                                        textcoords="offset points",
                                        ha="center", va="bottom", fontsize=8)

            ax.annotate("zoomed y-axis (±10%)", xy=(0.99, 0.02), xycoords="axes fraction",
                        ha="right", va="bottom", fontsize=7, color="dimgray")

    if legend_handles:
        figA.legend(legend_handles, ["RS", "noRS"], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # -------------------- Figure B: % reduction = 100*(noRS - RS)/noRS --------------------
    figB, axesB = plt.subplots(3, 3, figsize=(17, 11), sharey=True)
    figB.suptitle("% Reduction with RS vs noRS  (100 × (noRS − RS) / noRS; + = RS reduces)", fontsize=16, y=0.98)

    for r, w_val in enumerate(uniq_w):
        for c, n_val in enumerate(uniq_N):
            ax = axesB[r, c]
            sub = df[(df[col_w] == w_val) & (df[col_N] == n_val)]
            if sub.empty:
                ax.axis("off"); ax.set_title(f"w={w_val}, N={n_val}\n(no data)"); continue

            agg = sub.groupby("_variant")[metrics].mean()
            for v in ("RS","noRS"):
                if v not in agg.index:
                    agg.loc[v] = [np.nan] * len(metrics)
            agg = agg.loc[["RS","noRS"]]

            rs_vals, nors_vals = agg.iloc[0].values, agg.iloc[1].values
            pct = _percent_reduction(rs_vals, nors_vals)  # 100*(noRS - RS)/noRS

            bars = ax.bar(x, pct)  # single bar per metric
            ax.axhline(0.0, color="black", linewidth=1)
            ax.set_title(f"w={w_val}, N={n_val}")
            ax.set_xticks(x)
            ax.set_xticklabels([_wrap_label(m) for m in metrics], fontsize=9)
            ax.yaxis.set_major_formatter(FuncFormatter(_fmt_pct))
            ax.grid(axis="y", linestyle=":", alpha=0.4)

            # No bar annotations in Fig B (per request)
            if ANNOTATE_PCT:
                for rect, p in zip(bars, pct):
                    if np.isfinite(p):
                        ax.annotate(f"{p:.2f}%",
                                    xy=(rect.get_x() + rect.get_width()/2, p),
                                    xytext=(0, 3 if p >= 0 else -10),
                                    textcoords="offset points",
                                    ha="center", va="bottom" if p >= 0 else "top",
                                    fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()
