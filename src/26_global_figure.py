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
ax.set_title("(a) Transfer AUC, 38 events", loc="left", fontsize=7.5, fontweight="bold")
ax.text(0.95, 0.05, "29 of 38 above diagonal", transform=ax.transAxes, ha="right", fontsize=6.5)
ax = axs[1]
import json, glob
area = {}
for f in glob.glob(os.path.join(ROOT, "data", "processed*", "*_info.json")):
    d = json.load(open(f)); area[d["event"]] = d["area_km2"]
c = cov.loc[off.index]; A = pd.Series(area).loc[off.index]
for pt, mk, col, lab in [(False, "o", "#59a14f", "polygon inventory"), (True, "s", "#f28e2b", "point catalogue")]:
    m = (c.point_inventory == pt).values
    ax.scatter(A[m], off[m], marker=mk, s=16, color=col, label=lab, lw=0)
ax.set_xscale("log"); ax.set_xlabel("Study-domain area (km$^2$)"); ax.set_ylabel("Event offset $b_e$ (log-odds)")
ax.axhline(0, color="0.6", lw=0.5); ax.legend(frameon=False, fontsize=6.5, loc="lower left")
ax.set_title("(b) Offsets vs domain size", loc="left", fontsize=7.5, fontweight="bold")
ax = axs[2]
names = ["M0 unconditional", "F1 shaking (forecast-time)", "F2 climate (forecast-time)", "F3 all forecast-time", "P2 inventory + domain (post-event)"]
short = ["none", "F1 shaking", "F2 climate", "F3 F1+F2", "P2 post-event"]
x = np.arange(len(names))
rr = pd.read_csv(os.path.join(R, "global", "offset_regression.csv"))
from statsmodels.stats.proportion import proportion_confint
rng = np.random.default_rng(1)
wl, wh, cl, ch, cv, wm = [], [], [], [], [], []
for n in names:
    g = rr[rr.model == n]
    k, N = int(g.covered.sum()), len(g); lo, hi = proportion_confint(k, N, alpha=0.05, method="wilson")
    bw = [np.median(rng.choice(g.width.values, N)) for _ in range(2000)]
    wm.append(np.median(g.width)); wl.append(np.percentile(bw, 2.5)); wh.append(np.percentile(bw, 97.5))
    cv.append(k / N); cl.append(lo); ch.append(hi)
wm, wl, wh, cv, cl, ch = map(np.array, (wm, wl, wh, cv, cl, ch))
ax.bar(x, wm, yerr=[wm - wl, wh - wm], capsize=2, error_kw=dict(lw=0.6), color=["#bab0ac", "#4e79a7", "#4e79a7", "#4e79a7", "#f28e2b"])
ax.set_xticks(x); ax.set_xticklabels(short, rotation=35, ha="right"); ax.set_ylabel("Median width of 90% interval (×)")
ax2 = ax.twinx(); ax2.errorbar(x + 0.12, cv, yerr=[cv - cl, ch - cv], fmt="o-", color="#e15759", capsize=2, lw=1, ms=4)
ax2.set_ylim(0.5, 1.02); ax2.axhline(0.9, color="#e15759", ls=":", lw=0.6)
ax2.set_ylabel("Coverage (of 38 events)", color="#e15759"); ax2.spines["right"].set_visible(True)
pd.DataFrame(dict(model=names, width_median=wm, width_lo=wl, width_hi=wh, coverage=cv, cov_lo=cl, cov_hi=ch)).to_csv(os.path.join(R, "global", "fig11c_intervals.csv"), index=False)
ax.set_title("(c) Zero-shot prior covariates", loc="left", fontsize=7.5, fontweight="bold")
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(os.path.join(F, f"fig11_global.{ext}"), bbox_inches="tight")
