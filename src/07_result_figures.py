"""Figures for experiments E1-E3 and the Kaikoura transfer map."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
sys.path.insert(0, os.path.dirname(__file__))
from events import EVENTS
import importlib
X = importlib.import_module("04_experiments")

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(ROOT, "results"); F = os.path.join(ROOT, "figures"); os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "font.family": "DejaVu Sans"})
LAB = {k: v["label"] for k, v in EVENTS.items()}
MODELS = ["Physics (Jibson 2000)", "Physics-logit", "Global LR", "GBM", "PC-GBM"]
COL = {"Physics (Jibson 2000)": "#8c8c8c", "Physics-logit": "#b07aa1", "Global LR": "#4e79a7",
       "GBM": "#f28e2b", "PC-GBM": "#e15759", "Local": "#000000"}
MK = {"Physics (Jibson 2000)": "s", "Physics-logit": "D", "Global LR": "o", "GBM": "^", "PC-GBM": "*", "Local": "|"}


def save(fig, name):
    fig.savefig(os.path.join(F, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(F, name + ".png"), bbox_inches="tight")
    plt.close(fig)


def fig_loeo():
    r = pd.read_csv(os.path.join(R, "e1_loeo_metrics.csv"))
    order = (r[(r.model == "PC-GBM") & (r.setting == "LOEO transfer")].sort_values("auc").event.tolist())
    y = {e: i for i, e in enumerate(order)}
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 3.3), sharey=True)
    metrics = [("auc", "ROC AUC", None), ("ap_lift", "AP lift (× random)", "log"), ("e_over_o", "Expected / observed landslide cells", "log")]
    off = np.linspace(-0.28, 0.28, len(MODELS))
    for ax, (m, lab, sc) in zip(axs, metrics):
        for j, mo in enumerate(MODELS):
            d = r[(r.model == mo) & (r.setting == "LOEO transfer")]
            ax.scatter(d[m], [y[e] + off[j] for e in d.event], marker=MK[mo], color=COL[mo], s=22 if mo != "PC-GBM" else 40,
                       label=mo, zorder=3, edgecolor="none")
        d = r[r.setting == "Local spatial CV"]
        ax.scatter(d[m], [y[e] for e in d.event], marker="|", color="k", s=120, label="Local PC-GBM (spatial CV)", zorder=4)
        if sc: ax.set_xscale(sc)
        if m == "e_over_o": ax.axvline(1, color="k", lw=0.6, ls="--")
        if m == "auc": ax.axvline(0.5, color="k", lw=0.6, ls=":")
        ax.set_xlabel(lab); ax.grid(axis="x", lw=0.3, alpha=0.5)
        for i in range(len(order)): ax.axhspan(i - 0.45, i + 0.45, color="0.95" if i % 2 else "white", zorder=0)
    axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([LAB[e] for e in order])
    for a, t in zip(axs, "abc"): a.set_title(f"({t})", loc="left", fontweight="bold")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.0), frameon=False)
    save(fig, "fig3_loeo")


def fig_reliability(df):
    """Reliability diagrams of transferred Global LR and PC-GBM."""
    fig, axs = plt.subplots(3, 3, figsize=(7.1, 6.6), sharex=True, sharey=True)
    preds = {}
    for ax, te in zip(axs.ravel(), EVENTS):
        tr, ts = df[df.event != te], df[df.event == te]
        w = X.event_balanced_weights(tr)
        for mo in ["Global LR", "PC-GBM"]:
            p = X.MODELS[mo](tr, w)(ts); preds[(te, mo)] = p
            bins = np.quantile(np.log10(np.clip(p, 1e-7, 1)), np.linspace(0, 1, 13), method="linear")
            b = np.clip(np.digitize(np.log10(np.clip(p, 1e-7, 1)), bins[1:-1]), 0, 11)
            pp = np.array([np.average(p[b == i], weights=ts.w.values[b == i]) for i in range(12) if (b == i).any()])
            oo = np.array([np.average(ts.y.values[b == i], weights=ts.w.values[b == i]) for i in range(12) if (b == i).any()])
            ax.plot(pp, np.clip(oo, 1e-6, None), marker=MK[mo], color=COL[mo], ms=3.5, lw=1, label=mo)
        ax.plot([1e-6, 1], [1e-6, 1], "k--", lw=0.6)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-6, 1); ax.set_ylim(1e-6, 1)
        ax.set_title(LAB[te], fontsize=8)
    for a in axs[-1]: a.set_xlabel("Predicted probability")
    for a in axs[:, 0]: a.set_ylabel("Observed frequency")
    axs[0, 0].legend(frameon=False, fontsize=7)
    save(fig, "fig4_reliability")
    out = []
    for (te, mo), p in preds.items():
        out.append(pd.DataFrame({"event": te, "model": mo, "p": p}))
    pd.concat(out).to_parquet(os.path.join(R, "loeo_preds_lr_pcgbm.parquet"))


def fig_fewshot():
    r = pd.read_csv(os.path.join(R, "e2_fewshot.csv"))
    meth = ["Transfer (no recalibration)", "Transfer + intercept recalibration", "Transfer + Platt recalibration", "Local PC-GBM (mapped tiles only)"]
    c = ["#8c8c8c", "#e15759", "#4e79a7", "#59a14f"]
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.6))
    # per-event median over repetitions, then median and IQR over events
    g = r.groupby(["method", "k", "event"]).agg(auc=("auc", "median"), bss=("bss", "median"),
                                                 eo=("e_over_o", lambda x: np.median(np.abs(np.log2(x))))).reset_index()
    for ax, (m, lab) in zip(axs, [("auc", "ROC AUC"), ("bss", "Brier skill score"), ("eo", "|log$_2$(E/O)|")]):
        for mm, cc in zip(meth, c):
            d = g[g.method == mm].groupby("k")[m]
            med, lo, hi = d.median(), d.quantile(0.25), d.quantile(0.75)
            ax.plot(med.index, med.values, marker="o", ms=3, color=cc, label=mm)
            ax.fill_between(med.index, lo.values, hi.values, color=cc, alpha=0.15, lw=0)
        ax.set_xscale("log"); ax.set_xlabel("Mapped 3 × 3 km tiles (k)"); ax.set_ylabel(lab)
        ax.set_xticks([1, 2, 5, 10, 20, 40]); ax.set_xticklabels([1, 2, 5, 10, 20, 40])
    for a, t in zip(axs, "abc"): a.set_title(f"({t})", loc="left", fontweight="bold")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.tight_layout()
    save(fig, "fig5_fewshot")


def fig_shap():
    imp = pd.read_csv(os.path.join(R, "e3_shap_importance.csv"))
    cur = pd.read_csv(os.path.join(R, "e3_shap_pga_curves.csv"))
    fig = plt.figure(figsize=(7.1, 3.2))
    ax = fig.add_axes([0.08, 0.15, 0.48, 0.75])
    d = imp[imp.model == "PC-GBM"].pivot(index="event", columns="feature", values="share")
    feats = d.mean().sort_values(ascending=False).index
    d = d[feats].loc[list(EVENTS)]
    im = ax.imshow(d.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=60, ha="right")
    ax.set_yticks(range(len(d))); ax.set_yticklabels([LAB[e] for e in d.index])
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            ax.text(j, i, f"{100*d.values[i, j]:.0f}", ha="center", va="center", fontsize=5.5,
                    color="w" if d.values[i, j] < 0.6 * d.values.max() else "k")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Share of mean |SHAP|")
    ax.set_title("(a)", loc="left", fontweight="bold")
    ax2 = fig.add_axes([0.70, 0.15, 0.28, 0.75])
    cmap = plt.get_cmap("tab10")
    for i, e in enumerate(EVENTS):
        c = cur[cur.event == e].sort_values("log_pga")
        if len(c) == 0: continue
        bins = np.linspace(c.log_pga.min(), c.log_pga.max(), 15)
        bc = pd.cut(c.log_pga, bins); mm = c.groupby(bc, observed=True).agg(x=("log_pga", "median"), y=("shap_pga", "median"))
        ax2.plot(10 ** mm.x, mm.y, color=cmap(i), lw=1.2, label=LAB[e].split(" (")[0])
    ax2.set_xscale("log"); ax2.set_xlabel("ShakeMap PGA (g)"); ax2.set_ylabel("SHAP of log PGA (log-odds)")
    ax2.legend(frameon=False, fontsize=5.5, ncol=1, loc="upper left")
    ax2.set_title("(b)", loc="left", fontweight="bold")
    save(fig, "fig6_shap")


def fig_kaikoura():
    p = pd.read_parquet(os.path.join(R, "kaikoura_transfer_pred.parquet"))
    s = 2000.0
    p["bx"] = np.floor(p.x / s); p["by"] = np.floor(p.y_utm / s)
    p["pw"] = p.p_transfer * p.w; p["yw"] = p.y * p.w
    g = p.groupby(["bx", "by"]).agg(pred=("pw", "sum"), obs=("yw", "sum"), n=("w", "sum")).reset_index()
    a_km2 = (s / 1000) ** 2 * g.n / (s / 30) ** 2   # valid area
    g["pred_d"] = g.pred / ((s / 30) ** 2) * 100; g["obs_d"] = g.obs / ((s / 30) ** 2) * 100   # % of cells
    x0, y0 = g.bx.min(), g.by.min()
    nx, ny = int(g.bx.max() - x0 + 1), int(g.by.max() - y0 + 1)
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.9))
    vmax = np.nanpercentile(g.obs_d[g.obs_d > 0], 99)
    for ax, col, t in zip(axs[:2], ["obs_d", "pred_d"], ["(a) Observed (Massey et al. inventory)", "(b) PC-GBM trained on 8 other events"]):
        A = np.full((ny, nx), np.nan)
        A[(g.by - y0).astype(int), (g.bx - x0).astype(int)] = g[col]
        im = ax.imshow(np.clip(A, 0.01, None), origin="lower", cmap="magma_r", norm=LogNorm(0.01, vmax),
                       extent=[x0 * s / 1e3, (x0 + nx) * s / 1e3, y0 * s / 1e3, (y0 + ny) * s / 1e3])
        ax.set_title(t, loc="left", fontsize=7.5); ax.set_xlabel("Easting (km, UTM 59S)")
    axs[0].set_ylabel("Northing (km)")
    cb = fig.colorbar(im, ax=axs[:2], fraction=0.03, pad=0.02); cb.set_label("Landslide cells (% of 2 km cell)")
    ok = (g.n > 0.5 * (s / 30) ** 2)
    axs[2].scatter(np.clip(g.pred_d[ok], 1e-3, None), np.clip(g.obs_d[ok], 1e-3, None), s=3, alpha=0.4, color="#e15759", lw=0)
    axs[2].plot([1e-3, 100], [1e-3, 100], "k--", lw=0.6)
    axs[2].set_xscale("log"); axs[2].set_yscale("log"); axs[2].set_xlabel("Predicted (%)"); axs[2].set_ylabel("Observed (%)")
    from scipy.stats import spearmanr
    rho = spearmanr(g.pred_d[ok], g.obs_d[ok]).correlation
    axs[2].set_title(f"(c) 2 km cells, Spearman ρ = {rho:.2f}", loc="left", fontsize=7.5)
    json.dump(dict(spearman_2km=rho, n_cells=int(ok.sum()), total_pred=float(p.pw.sum()), total_obs=float(p.yw.sum())),
              open(os.path.join(R, "kaikoura_map_stats.json"), "w"), indent=1)
    save(fig, "fig7_kaikoura_transfer")


if __name__ == "__main__":
    which = sys.argv[1:] or ["loeo", "fewshot", "shap", "kaikoura", "reliability"]
    if "loeo" in which: fig_loeo()
    if "fewshot" in which: fig_fewshot()
    if "shap" in which: fig_shap()
    if "kaikoura" in which: fig_kaikoura()
    if "reliability" in which: fig_reliability(X.load())
