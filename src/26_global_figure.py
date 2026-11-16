"""Figure for the global extension: (a) LOEO AUC of HiPhyBoost vs PC-GBM for 38 earthquakes;
(b) event offsets (all-event fit) against antecedent precipitation ratio, by inventory type;
(c) zero-shot 90% interval width and coverage for unconditional and covariate-conditioned priors."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), ".."); R = os.path.join(ROOT, "results"); F = os.path.join(ROOT, "figures")
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
a = pd.read_csv(os.path.join(R, "global", "global_auc.csv"))
cov = pd.read_csv(os.path.join(ROOT, "data", "global", "event_covariates.csv")).set_index("event")
off = pd.read_csv(os.path.join(R, "global_loeo", "ALL_offsets.csv")).set_index("train_event").b
s = pd.read_csv(os.path.join(R, "global", "offset_regression_summary.csv")).set_index("model")
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
ax = axs[0]
for pac, mk, lab in [(True, "o", "Pacific Rim (25)"), (False, "^", "other (13)")]:
    g = a[a.pacific == pac]
    ax.scatter(g.auc_pcgbm, g.auc_hiphy, marker=mk, s=16, color="#e15759" if pac else "#4e79a7", label=lab, lw=0)
ax.plot([0.5, 1], [0.5, 1], "k--", lw=0.6); ax.set_xlim(0.53, 0.97); ax.set_ylim(0.53, 0.97)
ax.set_xlabel("AUC, PC-GBM"); ax.set_ylabel("AUC, HiPhyBoost"); ax.legend(frameon=False, fontsize=6.5, loc="upper left")
ax.set_title("(a) 29/38 above diagonal", loc="left", fontsize=7.5, fontweight="bold")
ax = axs[1]
c = cov.loc[off.index]
for pt, mk, col, lab in [(False, "o", "#59a14f", "polygon inventory"), (True, "s", "#f28e2b", "point catalogue")]:
    m = c.point_inventory == pt
    ax.scatter(c.ppt_ratio[m], off[m], marker=mk, s=16, color=col, label=lab, lw=0)
ax.set_xscale("log"); ax.set_xlabel("Antecedent precipitation / climatology"); ax.set_ylabel("Event offset $b_e$ (log-odds)")
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
ax.xaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2])); ax.xaxis.set_major_formatter(FixedFormatter(["0.25", "0.5", "1", "2"]))
ax.xaxis.set_minor_locator(NullLocator())
ax.axhline(0, color="0.6", lw=0.5); ax.legend(frameon=False, fontsize=6.5, loc="upper left")
ax.set_title("(b) Offsets and event covariates", loc="left", fontsize=7.5, fontweight="bold")
ax = axs[2]
names = ["M0 unconditional", "M1 inventory/mapping", "M2 shaking/ShakeMap", "M3 climate/wetness", "M4 all"]
short = ["none", "inventory", "ShakeMap", "climate", "all"]
x = np.arange(len(names))
ax.bar(x, s.loc[names, "width_median"], color="#bab0ac")
ax.set_xticks(x); ax.set_xticklabels(short, rotation=30); ax.set_ylabel("Median width of 90% interval (×)")
ax2 = ax.twinx(); ax2.plot(x, s.loc[names, "coverage"], "o-", color="#e15759"); ax2.set_ylim(0.5, 1.0); ax2.axhline(0.9, color="#e15759", ls=":", lw=0.6)
ax2.set_ylabel("Coverage", color="#e15759"); ax2.spines["right"].set_visible(True)
ax.set_title("(c) Zero-shot prior covariates", loc="left", fontsize=7.5, fontweight="bold")
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(os.path.join(F, f"fig11_global.{ext}"), bbox_inches="tight")
