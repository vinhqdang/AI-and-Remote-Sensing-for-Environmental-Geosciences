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
    """E5: few-shot updating, identical tile draws for all methods."""
    r = pd.concat([pd.read_csv(os.path.join(R, "e5_hiphy_fewshot.csv")),
                   pd.read_csv(os.path.join(R, "e6_fewshot_overdispersed.csv"))], ignore_index=True)
    meth = ["PC-GBM + intercept recal.", "PC-GBM + Platt recal.", "HiPhyBoost + MLE offset",
            "HiPhyBoost + Bayes (naive)", "HiPhyBoost + Bayes + overdispersion (no selection term)",
            "HiPhyBoost (full): selection-aware + overdispersion"]
    lab = {"HiPhyBoost + Bayes (naive)": "HiPhyBoost, cell-level Bayes",
           "HiPhyBoost + Bayes + overdispersion (no selection term)": "HiPhyBoost, tile-level Bayes, no selection term",
           "HiPhyBoost (full): selection-aware + overdispersion": "HiPhyBoost (full): tile-level, selection-aware"}
    c = ["#8c8c8c", "#4e79a7", "#b07aa1", "#f28e2b", "#59a14f", "#e15759"]
    r["abs_log2_eo"] = np.abs(np.log2(r.e_over_o))
    g = r.groupby(["method", "k", "event"]).agg(auc=("auc", "median"), eo=("abs_log2_eo", "median"),
                                                 cov=("covered", "mean")).reset_index()
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.6))
    panels = [("auc", "ROC AUC (unmapped area)"), ("eo", "|log$_2$(E/O)| (unmapped area)"), ("cov", "90% interval coverage")]
    for ax, (m, ylab) in zip(axs, panels):
        for mm, cc in zip(meth, c):
            d = g[g.method == mm]
            if m == "cov" and d[m].isna().all():
                continue
            d = d.groupby("k")[m]
            med, lo, hi = d.median(), d.quantile(0.25), d.quantile(0.75)
            if m == "cov":
                med = g[g.method == mm].groupby("k")[m].mean()
                ax.plot(med.index, med.values, marker="o", ms=3, color=cc, label=lab.get(mm, mm))
            else:
                ax.plot(med.index, med.values, marker="o", ms=3, color=cc, label=lab.get(mm, mm))
                ax.fill_between(med.index, lo.values, hi.values, color=cc, alpha=0.12, lw=0)
        ax.set_xscale("log"); ax.set_xlabel("Mapped 3 × 3 km tiles (k)"); ax.set_ylabel(ylab)
        ax.set_xticks([1, 2, 5, 10, 20, 40]); ax.set_xticklabels([1, 2, 5, 10, 20, 40])
    axs[2].axhline(0.9, color="k", lw=0.6, ls="--"); axs[2].set_ylim(0, 1.02)
    axs[1].axhline(1, color="k", lw=0.6, ls=":")
    for a, t in zip(axs, "abc"): a.set_title(f"({t})", loc="left", fontweight="bold")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig5_fewshot")


def fig_offsets():
    """Event offsets b_e of HiPhyBoost (all nine events) and zero-shot 90% intervals of E/O."""
    off = pd.read_csv(os.path.join(R, "hiphy_offsets_all.csv")).set_index("event")
    e4 = pd.read_csv(os.path.join(R, "e4_hiphy_loeo.csv"))
    e1 = pd.read_csv(os.path.join(R, "e1_loeo_metrics.csv"))
    zs = e4[e4.method == "HiPhyBoost (zero-shot Bayes)"].set_index("event")
    pc = e1[(e1.model == "PC-GBM") & (e1.setting == "LOEO transfer")].set_index("event")
    order = off.b_e.sort_values().index.tolist()
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
    yy = np.arange(len(order))
    axs[0].barh(yy, off.loc[order, "b_e"], color=["#4e79a7" if v < 0 else "#e15759" for v in off.loc[order, "b_e"]])
    axs[0].axvline(0, color="k", lw=0.6)
    axs[0].set_yticks(yy); axs[0].set_yticklabels([LAB[e] for e in order])
    axs[0].set_xlabel("Event offset $b_e$ (log-odds)")
    axs[0].set_title("(a) Event landslide propensity", loc="left", fontweight="bold", fontsize=8)
    for i, e in enumerate(order):
        lo, md, hi = zs.loc[e, ["eo_q05", "eo_q50", "eo_q95"]]
        axs[1].plot([lo, hi], [i, i], color="#e15759", lw=2.5, alpha=0.6, solid_capstyle="butt",
                    label="HiPhyBoost zero-shot 90% interval" if i == 0 else None)
        axs[1].scatter(md, i, color="#e15759", s=14, zorder=3)
        axs[1].scatter(pc.loc[e, "e_over_o"], i, marker="*", color="k", s=30, zorder=4,
                       label="PC-GBM point forecast" if i == 0 else None)
    axs[1].axvline(1, color="k", lw=0.6, ls="--"); axs[1].set_xscale("log")
    axs[1].set_xlabel("Predicted / observed landslide cells")
    axs[1].set_title("(b) Zero-shot forecast of landslide total", loc="left", fontweight="bold", fontsize=8)
    axs[1].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    save(fig, "fig8_offsets")


def fig_posterior(target="kaikoura2016", seed=3):
    """Prior and posteriors of the event offset for the target event as mapped tiles accumulate."""
    import hiphyboost as H
    from scipy.special import expit
    from scipy.optimize import brentq
    df = X.load(); ts = df[df.event == target].reset_index(drop=True)
    sc = pd.read_parquet(os.path.join(R, "hiphy_loeo_scores.parquet")); m = sc[sc.event == target].m.values
    off = pd.read_csv(os.path.join(R, "hiphy_loeo_offsets.csv")); off = off[off.target == target].iloc[0]
    mu, tau = off.mu, off.tau
    y, w = ts.y.values.astype(float), ts.w.values.astype(float)
    b_oracle = brentq(lambda b: np.sum(w * (expit(m + b) - y)), -25, 25)
    grid = np.linspace(-20, 20, 1601)
    _, inv = np.unique(X.blocks(ts, 3000), return_inverse=True); inv = inv.reshape(-1)
    pos = np.unique(inv[y == 1])
    import json
    s_u = json.load(open(os.path.join(R, "e6_sigma_u.json")))[target]["sigma_u"]
    tiles_all, _, A_all = H.tile_shift_loglik(m, y, w, inv)
    P0 = H.tile_p0_matrix(A_all, grid, s_u)
    rng = np.random.default_rng(seed); order = rng.permutation(pos)
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.8))
    prior = np.exp(-0.5 * ((grid - mu) / tau) ** 2); prior /= prior.sum()
    for ax in axs:
        ax.fill_between(grid, prior / (grid[1] - grid[0]), color="0.85", label="prior N(μ, τ²)")
        ax.axvline(b_oracle, color="k", ls="--", lw=0.8, label="oracle offset (full inventory)")
    cols = {1: "#f2b880", 5: "#e15759", 20: "#7b1e2b"}
    for k, c in cols.items():
        ch = order[:k]; cal = np.isin(inv, ch)
        LLb, A = H.grid_loglik(m[cal], y[cal], w[cal], inv[cal], grid)
        pn = H.posterior_from_ll(LLb, A, grid, mu, tau, False, True)
        g2, ps = H.posterior_offset_od(m[cal], y[cal], w[cal], inv[cal], ch, (tiles_all, P0), mu, tau, s_u, grid=grid)
        axs[0].plot(grid, pn / (grid[1] - grid[0]), color=c, lw=1.2, label=f"k = {k}")
        axs[1].plot(grid, ps / (grid[1] - grid[0]), color=c, lw=1.2, label=f"k = {k}")
    for ax, t in zip(axs, ["(a) Cell-level Bayesian update", f"(b) HiPhyBoost: tile-level, selection-aware (σ_u = {s_u:.2f})"]):
        ax.set_xlim(mu - 4 * tau, mu + 4 * tau); ax.set_xlabel("Event offset $b$ (log-odds)"); ax.set_ylabel("Density")
        ax.set_title(t, loc="left", fontweight="bold", fontsize=7.5)
    axs[0].set_yscale("symlog", linthresh=1); axs[1].set_ylim(0, None)
    axs[1].legend(frameon=False, fontsize=6.5)
    fig.tight_layout(); save(fig, "fig9_posterior_kaikoura")


def fig_maps_multi(events=("northridge1994", "iwate2008", "porgera2018", "palu2018"), s=2000.0):
    """Observed vs HiPhyBoost zero-shot landslide density (2 km cells) for four held-out events."""
    from numpy.polynomial.hermite_e import hermegauss
    from scipy.special import expit
    from scipy.stats import spearmanr
    df = X.load()
    sc = pd.read_parquet(os.path.join(R, "hiphy_loeo_scores.parquet"))
    offs = pd.read_csv(os.path.join(R, "hiphy_loeo_offsets.csv"))
    xg, wg = hermegauss(40); wg = wg / np.sqrt(2 * np.pi)
    fig, axs = plt.subplots(2, len(events), figsize=(7.1, 3.9))
    rows = []
    for j, e in enumerate(events):
        ts = df[df.event == e].reset_index(drop=True); m = sc[sc.event == e].m.values
        o = offs[offs.target == e].iloc[0]
        p = expit(m + o.mu)                            # point forecast at the prior median offset
        d = pd.DataFrame({"bx": np.floor(ts.x / s), "by": np.floor(ts.y_utm / s), "pw": p * ts.w, "yw": ts.y * ts.w, "n": ts.w})
        g = d.groupby(["bx", "by"]).sum().reset_index()
        g = g[g.n > 0.3 * (s / 30) ** 2]
        g["obs"] = g.yw / g.n * 100; g["pred"] = g.pw / g.n * 100
        rho = spearmanr(g.pred, g.obs).correlation; rows.append(dict(event=e, spearman_2km=rho))
        x0, y0 = g.bx.min(), g.by.min(); nx, ny = int(g.bx.max() - x0 + 1), int(g.by.max() - y0 + 1)
        vmax = max(np.nanpercentile(g.obs[g.obs > 0], 98), 0.05)
        for i, col in enumerate(["obs", "pred"]):
            A = np.full((ny, nx), np.nan); A[(g.by - y0).astype(int), (g.bx - x0).astype(int)] = g[col]
            im = axs[i, j].imshow(np.clip(A, vmax / 300, None), origin="lower", cmap="magma_r", norm=LogNorm(vmax / 300, vmax))
            axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
            for sp in axs[i, j].spines.values(): sp.set_visible(True); sp.set_linewidth(0.4)
        axs[0, j].set_title(LAB[e], fontsize=7.5)
        axs[1, j].set_xlabel(f"Spearman ρ = {rho:.2f}", fontsize=7)
    axs[0, 0].set_ylabel("Observed"); axs[1, 0].set_ylabel("HiPhyBoost\nzero-shot")
    cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01); cb.set_label("Landslide cells (% of 2 km cell)\n(colour scale per event)")
    pd.DataFrame(rows).to_csv(os.path.join(R, "maps_multi_spearman.csv"), index=False)
    save(fig, "fig10_maps_multi")


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
    ax2.legend(frameon=False, fontsize=5.5, ncol=1, loc="lower right")
    ax2.set_title("(b)", loc="left", fontweight="bold")
    save(fig, "fig6_shap")


def fig_kaikoura():
    p = pd.read_parquet(os.path.join(R, "kaikoura_transfer_pred.parquet"))
    title_b = "(b) PC-GBM trained on 8 other events"
    hp = os.path.join(R, "hiphy_loeo_scores.parquet")
    if os.path.exists(hp):
        from numpy.polynomial.hermite_e import hermegauss
        from scipy.special import expit
        sc = pd.read_parquet(hp); m = sc[sc.event == "kaikoura2016"].m.values
        off = pd.read_csv(os.path.join(R, "hiphy_loeo_offsets.csv")); off = off[off.target == "kaikoura2016"].iloc[0]
        xg, wg = hermegauss(40); wg = wg / np.sqrt(2 * np.pi)
        assert len(m) == len(p)
        p["p_transfer"] = expit(m + off.mu)          # point forecast at the prior median offset
        title_b = "(b) HiPhyBoost zero-shot (8 other events)"
    s = 2000.0
    p["bx"] = np.floor(p.x / s); p["by"] = np.floor(p.y_utm / s)
    p["pw"] = p.p_transfer * p.w; p["yw"] = p.y * p.w
    g = p.groupby(["bx", "by"]).agg(pred=("pw", "sum"), obs=("yw", "sum"), n=("w", "sum")).reset_index()
    a_km2 = (s / 1000) ** 2 * g.n / (s / 30) ** 2   # valid area
    g["pred_d"] = g.pred / ((s / 30) ** 2) * 100; g["obs_d"] = g.obs / ((s / 30) ** 2) * 100   # % of cells
    x0, y0 = g.bx.min(), g.by.min()
    nx, ny = int(g.bx.max() - x0 + 1), int(g.by.max() - y0 + 1)
    fig = plt.figure(figsize=(7.1, 3.0))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.05, 0.75, 1.05], wspace=0.12)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 4])]
    cax = fig.add_subplot(gs[0, 2])
    vmax = np.nanpercentile(g.obs_d[g.obs_d > 0], 99)
    ext = [x0 * s / 1e3, (x0 + nx) * s / 1e3, y0 * s / 1e3, (y0 + ny) * s / 1e3]
    for ax, col, t in zip(axs[:2], ["obs_d", "pred_d"], ["(a) Observed (GNS v2)", title_b.replace(" (8 other events)", "")]):
        A = np.full((ny, nx), np.nan)
        A[(g.by - y0).astype(int), (g.bx - x0).astype(int)] = g[col]
        im = ax.imshow(np.clip(A, 0.01, None), origin="lower", cmap="magma_r", norm=LogNorm(0.01, vmax), extent=ext)
        ax.set_title(t, loc="left", fontsize=7); ax.set_xlabel("Easting (km)")
        ax.tick_params(labelsize=6.5)
    axs[0].set_ylabel("Northing (km, UTM 59S)"); axs[1].set_yticklabels([])
    cb = fig.colorbar(im, cax=cax); cb.set_label("Landslide cells (% of 2 km cell)", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    ok = (g.n > 0.5 * (s / 30) ** 2)
    pos = ok & (g.obs_d > 0); zer = ok & (g.obs_d == 0)
    axs[2].scatter(g.pred_d[pos], g.obs_d[pos], s=3, alpha=0.4, color="#e15759", lw=0)
    axs[2].scatter(g.pred_d[zer], np.full(zer.sum(), 3e-3), s=3, alpha=0.25, color="0.4", lw=0)
    axs[2].axhline(6e-3, color="0.7", lw=0.5)
    axs[2].text(1.2e-3, 3.6e-3, "no landslide", fontsize=6, color="0.3")
    axs[2].plot([1e-3, 100], [1e-3, 100], "k--", lw=0.6)
    axs[2].set_xscale("log"); axs[2].set_yscale("log"); axs[2].set_xlim(1e-3, 100); axs[2].set_ylim(2e-3, 100)
    axs[2].set_xlabel("Predicted (%)"); axs[2].set_ylabel("Observed (%)")
    from scipy.stats import spearmanr
    rho = spearmanr(g.pred_d[ok], g.obs_d[ok]).correlation
    axs[2].set_title(f"(c) 2 km cells, Spearman ρ = {rho:.2f}", loc="left", fontsize=7)
    json.dump(dict(spearman_2km=rho, n_cells=int(ok.sum()), total_pred=float(p.pw.sum()), total_obs=float(p.yw.sum())),
              open(os.path.join(R, "kaikoura_map_stats.json"), "w"), indent=1)
    save(fig, "fig7_kaikoura_transfer")


if __name__ == "__main__":
    which = sys.argv[1:] or ["loeo", "fewshot", "shap", "kaikoura", "offsets", "posterior", "maps"]
    if "loeo" in which: fig_loeo()
    if "fewshot" in which: fig_fewshot()
    if "shap" in which: fig_shap()
    if "kaikoura" in which: fig_kaikoura()
    if "offsets" in which: fig_offsets()
    if "posterior" in which: fig_posterior()
    if "maps" in which: fig_maps_multi()
    if "reliability" in which: fig_reliability(X.load())
