"""Can event-level covariates predict the HiPhyBoost event offset of an unseen earthquake?

For each target event T (global leave-one-event-out):
  oracle offset b*_T : offset that matches the observed landslide total given the transferred scores
  training offsets   : b_e of the other events from the fold that held out T
  prior models       : M0 unconditional N(mean, sd) (current HiPhyBoost); conditional Bayesian ridge
                       regressions b_e = z_e' gamma + eps on pre-specified covariate sets, ridge penalty
                       chosen by leave-one-out within the training events; Student-t predictive.
Metrics: log predictive density of b*_T, 90% interval coverage of the landslide total, interval
width, absolute log2 error of the median forecast of the total. Also: global AUC comparison of
HiPhyBoost vs PC-GBM across all events.
Usage: python 24_offset_regression.py <loeo_out_dir>
"""
import os, sys, glob, json
import numpy as np, pandas as pd
from scipy.special import expit
from scipy.optimize import brentq
from scipy.stats import t as tdist, wilcoxon
from sklearn.metrics import roc_auc_score
ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(ROOT, "results", "global"); os.makedirs(R, exist_ok=True)

SETS = {
    "M1 inventory/mapping": ["point_inventory", "mapping_extent", "log_p5_area", "p5_missing", "year"],
    "M2 shaking/ShakeMap": ["log_stations", "mag", "log_depth"],
    "M3 climate/wetness": ["pdsi_prev", "log_ppt_ratio", "log_map"],
    "M4 all": ["point_inventory", "mapping_extent", "log_p5_area", "p5_missing", "year", "log_stations", "mag",
               "log_depth", "pdsi_prev", "log_ppt_ratio", "log_map"],
}


def covariates():
    c = pd.read_csv(os.path.join(ROOT, "data", "global", "event_covariates.csv")).set_index("event")
    z = pd.DataFrame(index=c.index)
    z["point_inventory"] = c.point_inventory.astype(float); z["mapping_extent"] = c.mapping_extent.astype(float)
    lp = np.log10(c.poly_area_p5_m2); z["p5_missing"] = lp.isna().astype(float); z["log_p5_area"] = lp.fillna(lp.median())
    z["year"] = c.year.astype(float); z["log_stations"] = np.log1p(c.shakemap_stations.clip(lower=0).fillna(0))
    z["mag"] = c.mag; z["log_depth"] = np.log10(c.depth_km.clip(lower=1))
    z["pdsi_prev"] = c.pdsi_prev.fillna(0); z["log_ppt_ratio"] = np.log(c.ppt_ratio.clip(lower=0.05)).fillna(0)
    z["log_map"] = np.log10(c.map_mm.clip(lower=50))
    z["rev"] = (c.mechanism == "reverse").astype(float); z["normal"] = (c.mechanism == "normal").astype(float)
    return z, c


def ridge_predict(Ztr, btr, zte, lams=(0.1, 0.3, 1, 3, 10, 30, 100)):
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-9
    A = (Ztr - mu) / sd; a = (zte - mu) / sd
    n, p = A.shape

    def fit(A_, b_, lam):
        X_ = np.column_stack([np.ones(len(A_)), A_]); P = lam * np.eye(p + 1); P[0, 0] = 0
        M = np.linalg.inv(X_.T @ X_ + P); g = M @ X_.T @ b_
        return g, M, X_
    # leave-one-out choice of lambda
    best, bl = np.inf, lams[0]
    for lam in lams:
        err = 0.0
        for i in range(n):
            m = np.arange(n) != i
            g, _, _ = fit(A[m], btr[m], lam)
            err += (btr[i] - np.r_[1, A[i]] @ g) ** 2
        if err < best:
            best, bl = err, lam
    g, M, X_ = fit(A, btr, bl)
    H = X_ @ M @ X_.T; dof = max(n - np.trace(H), 2.0)
    s2 = np.sum((btr - X_ @ g) ** 2) / dof
    x = np.r_[1, a]
    return float(x @ g), float(np.sqrt(s2 * (1 + x @ M @ x))), dof, bl


def main(loeo_dir):
    z, cov = covariates()
    rows, auc_rows = [], []
    targets = sorted(f.split("/")[-1][:-16] for f in glob.glob(os.path.join(loeo_dir, "*_scores.parquet")))
    for T in targets:
        s = pd.read_parquet(os.path.join(loeo_dir, f"{T}_scores.parquet"))
        o = pd.read_csv(os.path.join(loeo_dir, f"{T}_offsets.csv"))
        m, y, w = s.m.values, s.y.values.astype(float), s.w.values.astype(float)
        obs = np.sum(w * y)
        auc_rows.append(dict(event=T, auc_hiphy=roc_auc_score(y, m, sample_weight=w), auc_pcgbm=roc_auc_score(y, s.p_pcgbm, sample_weight=w)))
        b_star = brentq(lambda b: np.sum(w * (expit(m + b) - y)), -30, 30)
        o = o[o.train_event.isin(z.index)]
        btr = o.b.values; Ztr_all = z.loc[o.train_event]
        preds = {"M0 unconditional": (btr.mean(), btr.std(ddof=1) * np.sqrt(1 + 1 / len(btr)), len(btr) - 1, np.nan)}
        if T in z.index:
            for name, cols in SETS.items():
                preds[name] = ridge_predict(Ztr_all[cols].values.astype(float), btr, z.loc[T, cols].values.astype(float))
        for name, (mu, sd, dof, lam) in preds.items():
            q = mu + sd * tdist.ppf([0.05, 0.5, 0.95], dof)
            tot = np.array([np.sum(w * expit(m + b)) for b in q]) / obs
            rows.append(dict(event=T, model=name, b_star=b_star, b_pred=mu, b_sd=sd, lam=lam,
                             lpd=float(tdist.logpdf((b_star - mu) / sd, dof) - np.log(sd)),
                             eo_q05=tot[0], eo_q50=tot[1], eo_q95=tot[2], covered=bool(tot[0] <= 1 <= tot[2]),
                             width=tot[2] / tot[0], abs_log2_eo=abs(np.log2(tot[1]))))
    r = pd.DataFrame(rows); r.to_csv(os.path.join(R, "offset_regression.csv"), index=False)
    a = pd.DataFrame(auc_rows); a.to_csv(os.path.join(R, "global_auc.csv"), index=False)
    d = a.auc_hiphy - a.auc_pcgbm
    summ = dict(n_events=len(a), auc_hiphy_median=float(a.auc_hiphy.median()), auc_pcgbm_median=float(a.auc_pcgbm.median()),
                n_better=int((d > 0).sum()), mean_diff=float(d.mean()), p_wilcoxon=float(wilcoxon(d).pvalue))
    g = r.groupby("model").agg(lpd_mean=("lpd", "mean"), coverage=("covered", "mean"), width_median=("width", "median"),
                                abs_log2_eo_median=("abs_log2_eo", "median"), n=("event", "size"))
    # paired comparison of predictive density against M0
    base = r[r.model == "M0 unconditional"].set_index("event").lpd
    for mname in g.index:
        if mname.startswith("M0"):
            continue
        dd = r[r.model == mname].set_index("event").lpd - base
        dd = dd.dropna()
        g.loc[mname, "lpd_gain_vs_M0"] = dd.mean(); g.loc[mname, "p_lpd_gain"] = wilcoxon(dd).pvalue if len(dd) > 5 else np.nan
    g.to_csv(os.path.join(R, "offset_regression_summary.csv"))
    json.dump(summ, open(os.path.join(R, "global_auc_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1)); print(g.round(3).to_string())
    # offsets from the all-event fit vs covariates
    allo = pd.read_csv(os.path.join(loeo_dir, "ALL_offsets.csv")).set_index("train_event").b
    zz = z.loc[allo.index]
    corr = {c: float(np.corrcoef(zz[c], allo)[0, 1]) for c in zz.columns if zz[c].std() > 0}
    json.dump(corr, open(os.path.join(R, "offset_covariate_correlations.json"), "w"), indent=1)
    print({k: round(v, 2) for k, v in sorted(corr.items(), key=lambda kv: -abs(kv[1]))})


if __name__ == "__main__":
    main(sys.argv[1])
