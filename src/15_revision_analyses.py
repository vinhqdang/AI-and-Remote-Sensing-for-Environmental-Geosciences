"""Revision analyses (no refitting):
 R1 event-level paired tests of AUC differences (Wilcoxon signed-rank over the nine events);
 R2 zero-shot intervals with a Student-t predictive for an unseen event offset (accounts for
    estimating mu and tau from eight events);
 R3 grouped SHAP attribution stability (terrain / ground motion / Newmark);
 R4 few-shot: PC-GBM with the same empirical-Bayes quasi-posterior as HiPhyBoost (fair baseline),
    and sensitivity of HiPhyBoost's quasi-posterior to the dispersion factor phi.
Tile draws reproduce E5. Usage: python 15_revision_analyses.py [R1R2R3 | fewshot <events...> | merge]"""
import os, sys, importlib, json, glob
import numpy as np, pandas as pd
from scipy.special import expit, logit
from scipy.optimize import brentq
from scipy.stats import wilcoxon, t as tdist, spearmanr
sys.path.insert(0, os.path.dirname(__file__))
X = importlib.import_module("04_experiments")
H = importlib.import_module("hiphyboost")
R = X.RES_DIR
QS = (0.05, 0.5, 0.95); GRID = np.linspace(-20, 20, 1601); KS = (1, 2, 5, 10, 20, 40); TILE = 3000


def r1_r2_r3():
    out = {}
    e1 = pd.read_csv(os.path.join(R, "e1_loeo_metrics.csv")); e4 = pd.read_csv(os.path.join(R, "e4_hiphy_loeo.csv"))
    h = e4[e4.method == "HiPhyBoost (plug-in)"].set_index("event").auc
    ab = e4[e4.method.str.startswith("Ablation")].set_index("event").auc
    for mo in ["PC-GBM", "GBM", "Global LR"]:
        b = e1[(e1.model == mo) & (e1.setting == "LOEO transfer")].set_index("event").auc.loc[h.index]
        d = h - b
        out[f"hiphy_vs_{mo}"] = dict(mean=float(d.mean()), median=float(d.median()), n_pos=int((d > 0).sum()),
                                     p_wilcoxon=float(wilcoxon(d).pvalue))
    d = h - ab; out["hiphy_vs_ablation"] = dict(mean=float(d.mean()), median=float(d.median()), n_pos=int((d > 0).sum()), p_wilcoxon=float(wilcoxon(d).pvalue))
    pc = e1[(e1.model == "PC-GBM") & (e1.setting == "LOEO transfer")].set_index("event").auc.loc[ab.index]
    d = ab - pc; out["ablation_vs_PC-GBM"] = dict(mean=float(d.mean()), median=float(d.median()), n_pos=int((d > 0).sum()), p_wilcoxon=float(wilcoxon(d).pvalue))
    # R2: t-based zero-shot interval
    df = X.load(); sc = pd.read_parquet(os.path.join(R, "hiphy_loeo_scores.parquet")); offs = pd.read_csv(os.path.join(R, "hiphy_loeo_offsets.csv"))
    rows = []
    for e in sorted(df.event.unique()):
        ts = df[df.event == e]; m = sc[sc.event == e].m.values; w = ts.w.values; obs = np.sum(w * ts.y.values)
        o = offs[offs.target == e]; bs = o.b.values; n = len(bs); mu, s = bs.mean(), bs.std(ddof=1)
        tq = tdist.ppf([0.05, 0.5, 0.95], df=n - 1) * s * np.sqrt(1 + 1 / n) + mu
        iv = np.array([np.sum(w * expit(m + b)) for b in tq]) / obs
        rows.append(dict(event=e, eo_q05_t=iv[0], eo_q50_t=iv[1], eo_q95_t=iv[2], covered_t=bool(iv[0] <= 1 <= iv[2]), width_t=iv[2] / iv[0]))
    zt = pd.DataFrame(rows); zt.to_csv(os.path.join(R, "rev_zeroshot_t.csv"), index=False)
    out["zeroshot_t"] = dict(coverage=int(zt.covered_t.sum()), median_width=float(zt.width_t.median()))
    zs = e4[e4.method == "HiPhyBoost (zero-shot Bayes)"]
    out["zeroshot_bayes_mean_EO_median"] = float(zs.e_over_o.median())
    # R3 grouped SHAP stability
    imp = pd.read_csv(os.path.join(R, "e3_shap_importance.csv"))
    grp = {"slope": "terrain", "slope_max5": "terrain", "relief510": "terrain", "tpi510": "terrain", "relpos1k": "terrain",
           "curv_prof": "terrain", "curv_plan": "terrain", "slope_sd3": "terrain", "log_pga": "ground motion",
           "log_pgv": "ground motion", "log_newmark": "Newmark"}
    imp["group"] = imp.feature.map(grp)
    for mo in ["GBM", "PC-GBM"]:
        g = imp[imp.model == mo].groupby(["event", "group"]).share.sum().unstack()
        out[f"shap_group_share_{mo}"] = g.round(3).to_dict()
        out[f"shap_group_share_{mo}_mean"] = g.mean().round(3).to_dict()
        out[f"shap_group_share_{mo}_range"] = {c: [float(g[c].min()), float(g[c].max())] for c in g}
        # null reference: rank agreement expected from random importance vectors
    rng = np.random.default_rng(0)
    null = [np.mean([spearmanr(rng.random(11), rng.random(11)).correlation for _ in range(36)]) for _ in range(2000)]
    out["shap_null_mean_pairwise_spearman_95pct"] = float(np.percentile(null, 95))
    json.dump(out, open(os.path.join(R, "rev_stats.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:3000])


def fewshot(only, reps=30, reps_sens=10, phis=(15.0, 30.0, 100.0)):
    df = X.load().reset_index(drop=True); events = sorted(df.event.unique())
    sc = pd.read_parquet(os.path.join(R, "hiphy_loeo_scores.parquet")); offs = pd.read_csv(os.path.join(R, "hiphy_loeo_offsets.csv"))
    pcp = pd.read_parquet(os.path.join(R, "loeo_preds_lr_pcgbm.parquet")); pcp = pcp[pcp.model == "PC-GBM"]
    phi_used = {k: v["phi"] for k, v in json.load(open(os.path.join(R, "e7_phi.json"))).items()}
    data, pcshift, pcphi = {}, {}, {}
    for e in events:
        ts = df[df.event == e].reset_index(drop=True)
        m = sc[sc.event == e].m.values; y = ts.y.values.astype(float); w = ts.w.values.astype(float)
        s = logit(np.clip(pcp[pcp.event == e].p.values, 1e-7, 1 - 1e-7))
        _, inv = np.unique(X.blocks(ts, TILE), return_inverse=True); inv = inv.reshape(-1)
        a = brentq(lambda aa: np.sum(w * (expit(s + aa) - y)), -25, 25); p = expit(s + a)
        O = np.bincount(inv, weights=w * y); E = np.bincount(inv, weights=w * p); V = np.bincount(inv, weights=w * p * (1 - p))
        pcshift[e] = a; pcphi[e] = float(np.sum((O - E) ** 2) / np.sum(V))
        data[e] = (m, s, y, w, inv)
    rng = np.random.default_rng(7); draws = {}
    for target in events:
        _, _, y_, _, inv_ = data[target]
        ub_ = np.unique(inv_); pt_ = np.unique(inv_[y_ == 1])
        for k in KS:
            if k >= len(ub_) // 2 or len(pt_) == 0:
                continue
            for rep in range(reps):
                draws[(target, k, rep)] = rng.choice(pt_, min(k, len(pt_)), replace=False)
    rows = []
    for target in only:
        m, s, y, w, inv = data[target]
        others = [e for e in events if e != target]
        # PC-GBM empirical-Bayes prior: required log-odds shifts of the other events (out-of-sample)
        sh = np.array([pcshift[e] for e in others]); mu_pc, tau_pc = sh.mean(), sh.std(ddof=1)
        phi_pc = float(np.median([pcphi[e] for e in others]))
        o = offs[offs.target == target].iloc[0]; mu, tau = o.mu, o.tau
        ub = np.unique(inv); pos = np.unique(inv[y == 1])
        for k in KS:
            if k >= len(ub) // 2 or len(pos) == 0:
                continue
            for rep in range(reps):
                chosen = draws[(target, k, rep)]; cal = np.isin(inv, chosen); ev = ~cal
                if y[ev].sum() < 5:
                    continue
                we, ye = w[ev], y[ev]; obs = np.sum(we * ye)
                runs = [("PC-GBM + EB quasi-Bayes", s, mu_pc, tau_pc, phi_pc)]
                if rep < reps_sens:
                    runs += [(f"HiPhyBoost quasi-Bayes, phi={ph:g}", m, mu, tau, ph) for ph in phis]
                for name, sc_, mu_, tau_, ph in runs:
                    LLb, _ = H.grid_loglik(sc_[cal], y[cal], w[cal], inv[cal], GRID)
                    lp = LLb / ph - 0.5 * ((GRID - mu_) / tau_) ** 2; lp -= lp.max(); post = np.exp(lp); post /= post.sum()
                    p = H.mixture_predict(sc_[ev], GRID, post)
                    iv = H.posterior_interval(sc_[ev], we, GRID, post, QS) / obs
                    r = X.wmetrics(ye, p, we)
                    r.update(event=target, k=k, rep=rep, method=name, eo_q05=iv[0], eo_q50=iv[1], eo_q95=iv[2],
                             covered=float(iv[0] <= 1 <= iv[2]), phi=ph)
                    rows.append(r)
        print("R4", target, "done", flush=True)
    os.makedirs(os.path.join(R, "rev_parts"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(R, "rev_parts", "rev_" + "_".join(only) + ".csv"), index=False)


if __name__ == "__main__":
    if sys.argv[1] == "R1R2R3":
        r1_r2_r3()
    elif sys.argv[1] == "merge":
        pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, "rev_parts", "rev_*.csv")))]).to_csv(os.path.join(R, "rev_fewshot.csv"), index=False)
    else:
        fewshot(sys.argv[2:])
