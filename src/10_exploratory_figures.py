"""Exploratory figures: covariate shift between events and empirical landslide-frequency
response curves (motivation for latent event offsets)."""
import os, sys, importlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
X = importlib.import_module("04_experiments")
from events import EVENTS
ROOT = os.path.join(os.path.dirname(__file__), ".."); F = os.path.join(ROOT, "figures")
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
LAB = {k: v["label"] for k, v in EVENTS.items()}
CM = plt.get_cmap("tab10"); COL = {e: CM(i) for i, e in enumerate(EVENTS)}


def wquant(x, w, qs):
    o = np.argsort(x); x, w = x[o], w[o]; c = np.cumsum(w) / w.sum()
    return np.interp(qs, c, x)


def fig_shift(df):
    vars_ = [("log_pgv", "log$_{10}$ PGV (cm s$^{-1}$)"), ("slope", "Slope (°)"), ("log_newmark", "log$_{10}$($D_N$ + 0.1 cm)")]
    ev = list(EVENTS)
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 3.2), sharey=True)
    for ax, (v, lab) in zip(axs, vars_):
        for i, e in enumerate(ev):
            d = df[df.event == e]
            a = wquant(d[v].values, d.w.values, [0.05, 0.25, 0.5, 0.75, 0.95])     # all cells (domain)
            l = d[d.y == 1][v].values; b = np.quantile(l, [0.05, 0.25, 0.5, 0.75, 0.95])
            ax.plot([a[0], a[4]], [i + 0.17] * 2, color="0.6", lw=1)
            ax.plot([a[1], a[3]], [i + 0.17] * 2, color="0.6", lw=4, solid_capstyle="butt")
            ax.plot(a[2], i + 0.17, "|", color="k", ms=6)
            ax.plot([b[0], b[4]], [i - 0.17] * 2, color=COL[e], lw=1)
            ax.plot([b[1], b[3]], [i - 0.17] * 2, color=COL[e], lw=4, solid_capstyle="butt")
            ax.plot(b[2], i - 0.17, "|", color="k", ms=6)
        ax.set_xlabel(lab); ax.grid(axis="x", lw=0.3, alpha=0.5)
    axs[0].set_yticks(range(len(ev))); axs[0].set_yticklabels([LAB[e] for e in ev]); axs[0].invert_yaxis()
    axs[0].plot([], [], color="0.6", lw=4, label="all cells in domain"); axs[0].plot([], [], color="C0", lw=4, label="landslide cells (event colour)")
    for a, t in zip(axs, "abc"): a.set_title(f"({t})", loc="left", fontweight="bold")
    fig.tight_layout()
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.6, 0.0))
    fig.savefig(os.path.join(F, "figS1_covariate_shift.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(F, "figS1_covariate_shift.png"), bbox_inches="tight")


def fig_response(df):
    """Weighted landslide frequency vs Newmark displacement and vs PGA per event."""
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.0))
    for ax, (v, lab, bins) in zip(axs, [("log_newmark", "log$_{10}$($D_N$ + 0.1 cm)", np.linspace(-1, 2.6, 13)),
                                         ("log_pga", "log$_{10}$ PGA (g)", np.linspace(-1.4, 0.2, 13))]):
        for e in EVENTS:
            d = df[df.event == e]
            idx = np.digitize(d[v].values, bins)
            xs, ys = [], []
            for k in range(1, len(bins)):
                m = idx == k
                if d.w.values[m].sum() < 2000 or d.y.values[m].sum() < 5:
                    continue
                xs.append(0.5 * (bins[k - 1] + bins[k])); ys.append(np.average(d.y.values[m], weights=d.w.values[m]))
            if xs:
                ax.plot(xs, ys, marker="o", ms=2.5, lw=1.2, color=COL[e], label=LAB[e])
        ax.set_yscale("log"); ax.set_xlabel(lab); ax.set_ylabel("Landslide frequency (fraction of cells)")
    axs[0].set_title("(a) vs Newmark displacement", loc="left", fontweight="bold")
    axs[1].set_title("(b) vs ShakeMap PGA", loc="left", fontweight="bold")
    axs[1].legend(frameon=False, fontsize=6, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig_response_curves.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(F, "fig_response_curves.png"), bbox_inches="tight")


if __name__ == "__main__":
    df = X.load()
    fig_shift(df); fig_response(df)
