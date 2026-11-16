"""Cross-event transfer experiments for coseismic landslide models.

E1  Leave-one-event-out (LOEO) transfer of five models; spatial-block CV within event
    as a local reference.
E2  Few-shot recalibration of transferred models from k rapidly mapped 3 km tiles.
E3  TreeSHAP attribution stability across events (event-specific models) and
    ground-motion response curves.
All metrics are weighted with the inverse sampling probability of negatives, i.e. they
refer to the full study domain at natural landslide prevalence.
"""
import os, sys, json, glob, warnings
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from events import EVENTS
warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
RES_DIR = os.path.join(ROOT, "results"); os.makedirs(RES_DIR, exist_ok=True)
RNG = np.random.default_rng(2026)

TERRAIN = ["slope", "slope_max5", "relief510", "tpi510", "relpos1k", "curv_prof", "curv_plan", "slope_sd3"]
MOTION = ["log_pga", "log_pgv"]
PHYS = ["log_newmark"]
FEATS = TERRAIN + MOTION + PHYS
# physically required monotone responses: stronger shaking, steeper slopes, larger
# Newmark displacement and higher local relief cannot lower failure probability
MONO = {"slope": 1, "slope_max5": 1, "relief510": 1, "log_pga": 1, "log_pgv": 1, "log_newmark": 1}
XGB_PARAMS = dict(max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5, reg_lambda=1.0, objective="binary:logistic",
                  tree_method="hist", nthread=4, eval_metric="logloss")
N_ROUNDS = 400


def load():
    dfs = []
    for k in EVENTS:
        p = os.path.join(PROC, f"{k}.parquet")
        if os.path.exists(p):
            dfs.append(pd.read_parquet(p))
    df = pd.concat(dfs, ignore_index=True)
    df["log_pga"] = np.log10(np.clip(df.pga, 1e-3, None))
    df["log_pgv"] = np.log10(np.clip(df.pgv, 1e-2, None))
    df["log_newmark"] = np.log10(df.newmark_cm + 0.1)
    df = df.dropna(subset=FEATS).reset_index(drop=True)
    return df


def event_balanced_weights(d):
    """Each training event contributes equal total weight; within-event prevalence kept."""
    w = d.w.values.astype(float).copy()
    for e, idx in d.groupby("event").indices.items():
        w[idx] = w[idx] / w[idx].sum()
    return w / w.mean()


def wmetrics(y, p, w):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    prev = np.average(y, weights=w)
    auc = roc_auc_score(y, p, sample_weight=w)
    ap = average_precision_score(y, p, sample_weight=w)
    brier = np.average((p - y) ** 2, weights=w)
    oe = np.sum(w * p) / np.sum(w * y)                      # expected / observed landslide cells
    lr = LogisticRegression(C=1e6).fit(np.log(p / (1 - p))[:, None], y, sample_weight=w)
    # Brier skill vs. climatology (known prevalence)
    bss = 1 - brier / (prev * (1 - prev))
    return dict(auc=auc, ap=ap, ap_lift=ap / prev, brier=brier, bss=bss, e_over_o=oe,
                cal_slope=float(lr.coef_[0, 0]), cal_int=float(lr.intercept_[0]), prevalence=prev)


# ---------------------------------------------------------------- models
def jibson2000(d):
    dn = d.newmark_cm.values
    return np.clip(0.335 * (1 - np.exp(-0.048 * dn ** 1.565)), 1e-6, None)


def lr_features(d):
    s = d.slope.values
    return np.column_stack([d.log_pgv, s, d.log_pgv * s, d.relief510 / 100])


def fit_lr(d, w):
    m = LogisticRegression(C=1.0, max_iter=2000).fit(lr_features(d), d.y, sample_weight=w)
    return lambda x: m.predict_proba(lr_features(x))[:, 1]


def fit_physlogit(d, w):
    X = lambda x: np.column_stack([x.log_newmark, x.slope / 10])
    m = LogisticRegression(C=1.0, max_iter=1000).fit(X(d), d.y, sample_weight=w)
    return lambda x: m.predict_proba(X(x))[:, 1]


def fit_xgb(d, w, feats=FEATS, mono=False, rounds=N_ROUNDS, ret_model=False):
    params = dict(XGB_PARAMS)
    if mono:
        params["monotone_constraints"] = "(" + ",".join(str(MONO.get(f, 0)) for f in feats) + ")"
    dm = xgb.DMatrix(d[feats].values, label=d.y.values, weight=w, feature_names=feats)
    b = xgb.train(params, dm, rounds)
    f = lambda x: b.predict(xgb.DMatrix(x[feats].values, feature_names=feats))
    return (f, b) if ret_model else f


MODELS = {
    "Physics (Jibson 2000)": None,
    "Physics-logit": fit_physlogit,
    "Global LR": fit_lr,
    "GBM": lambda d, w: fit_xgb(d, w, mono=False),
    "PC-GBM": lambda d, w: fit_xgb(d, w, mono=True),
}


def blocks(d, size):
    return (np.floor(d.x.values / size).astype(np.int64) * 100003 + np.floor(d.y_utm.values / size).astype(np.int64))


# ---------------------------------------------------------------- E1
def e1_loeo(df):
    rows, preds = [], {}
    events = df.event.unique()
    for te in events:
        tr, ts = df[df.event != te], df[df.event == te]
        w = event_balanced_weights(tr)
        for name, fit in MODELS.items():
            p = jibson2000(ts) if fit is None else fit(tr, w)(ts)
            m = wmetrics(ts.y.values, p, ts.w.values); m.update(event=te, model=name, setting="LOEO transfer")
            rows.append(m)
            preds[(te, name)] = p
        # local reference: 5-fold spatial block CV (5 km blocks) within the target event
        b = blocks(ts, 5000); ub = np.unique(b); RNG.shuffle(ub)
        fold = dict(zip(ub, np.arange(len(ub)) % 5)); fo = np.array([fold[x] for x in b])
        p = np.zeros(len(ts))
        for k in range(5):
            trk, tsk = ts[fo != k], ts[fo == k]
            p[fo == k] = fit_xgb(trk, trk.w.values / trk.w.mean(), mono=True)(tsk)
        m = wmetrics(ts.y.values, p, ts.w.values); m.update(event=te, model="PC-GBM", setting="Local spatial CV")
        rows.append(m)
        print(te, "done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(RES_DIR, "e1_loeo_metrics.csv"), index=False)
    return r, preds


def block_bootstrap_auc_diff(df, preds, a="PC-GBM", b="Global LR", nboot=200):
    out = []
    for te in df.event.unique():
        ts = df[df.event == te]; bl = blocks(ts, 5000); ub, inv = np.unique(bl, return_inverse=True)
        idx_by_b = [np.flatnonzero(inv == i) for i in range(len(ub))]
        pa, pb = preds[(te, a)], preds[(te, b)]
        diffs = []
        for _ in range(nboot):
            sel = np.concatenate([idx_by_b[i] for i in RNG.integers(0, len(ub), len(ub))])
            y, w = ts.y.values[sel], ts.w.values[sel]
            if y.sum() == 0 or y.sum() == len(y):
                continue
            diffs.append(roc_auc_score(y, pa[sel], sample_weight=w) - roc_auc_score(y, pb[sel], sample_weight=w))
        diffs = np.array(diffs)
        out.append(dict(event=te, diff=np.median(diffs), lo=np.percentile(diffs, 2.5), hi=np.percentile(diffs, 97.5)))
    r = pd.DataFrame(out); r.to_csv(os.path.join(RES_DIR, f"e1_auc_diff_{a}_vs_{b}.csv".replace(" ", "_")), index=False)
    return r


# ---------------------------------------------------------------- E2
def e2_fewshot(df, preds, ks=(1, 2, 5, 10, 20, 40), reps=30, tile=3000):
    rows = []
    for te in df.event.unique():
        ts = df[df.event == te].reset_index(drop=True)
        tb = blocks(ts, tile); ub, inv = np.unique(tb, return_inverse=True)
        pos_tiles = np.unique(inv[ts.y.values == 1])
        s = np.log(np.clip(preds[(te, "PC-GBM")], 1e-7, 1 - 1e-7)); s = s - np.log1p(-np.exp(s))  # logit
        for k in ks:
            if k >= len(ub) // 2:
                continue
            for rep in range(reps):
                # analysts map k tiles chosen at random among tiles that contain at least one landslide
                # (rapid mapping targets visibly affected areas); everything in a mapped tile is labelled
                chosen = RNG.choice(pos_tiles, min(k, len(pos_tiles)), replace=False)
                cal = np.isin(inv, chosen); ev = ~cal
                y, w = ts.y.values, ts.w.values
                if y[ev].sum() < 5:
                    continue
                res = {}
                res["Transfer (no recalibration)"] = preds[(te, "PC-GBM")][ev]
                # intercept-only (prior-shift) recalibration
                from scipy.optimize import brentq
                yc, wc, sc = y[cal], w[cal], s[cal]
                def g(a): return np.sum(wc * (1 / (1 + np.exp(-(sc + a))))) - np.sum(wc * yc)
                try:
                    a = brentq(g, -15, 15)
                except ValueError:
                    a = 0.0
                res["Transfer + intercept recalibration"] = 1 / (1 + np.exp(-(s[ev] + a)))
                lr = LogisticRegression(C=1.0).fit(sc[:, None], yc, sample_weight=wc / wc.mean())
                res["Transfer + Platt recalibration"] = lr.predict_proba(s[ev][:, None])[:, 1]
                # local model trained only on the mapped tiles
                if yc.sum() >= 3:
                    loc = fit_xgb(ts[cal], wc / wc.mean(), mono=True, rounds=200)
                    res["Local PC-GBM (mapped tiles only)"] = loc(ts[ev])
                for name, p in res.items():
                    m = wmetrics(y[ev], p, w[ev]); m.update(event=te, k=k, rep=rep, method=name,
                                                         n_ls_mapped=int(yc.sum()))
                    rows.append(m)
        print("E2", te, flush=True)
    r = pd.DataFrame(rows); r.to_csv(os.path.join(RES_DIR, "e2_fewshot.csv"), index=False)
    return r


# ---------------------------------------------------------------- E3
def e3_shap(df):
    imp_rows, curves = [], []
    for mono in [False, True]:
        tag = "PC-GBM" if mono else "GBM"
        for e in df.event.unique():
            d = df[df.event == e]
            # balanced subsample for speed
            f, b = fit_xgb(d, d.w.values / d.w.mean(), mono=mono, ret_model=True)
            sub = d.sample(min(len(d), 20000), random_state=1)
            sv = b.predict(xgb.DMatrix(sub[FEATS].values, feature_names=FEATS), pred_contribs=True)[:, :-1]
            imp = np.abs(sv).mean(0)
            for fe, v in zip(FEATS, imp):
                imp_rows.append(dict(model=tag, event=e, feature=fe, mean_abs_shap=v, share=v / imp.sum()))
            if mono:
                j = FEATS.index("log_pga")
                for xv, yv in zip(sub.log_pga.values[:3000], sv[:3000, j]):
                    curves.append(dict(event=e, log_pga=xv, shap_pga=yv))
    imp = pd.DataFrame(imp_rows); imp.to_csv(os.path.join(RES_DIR, "e3_shap_importance.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(RES_DIR, "e3_shap_pga_curves.csv"), index=False)
    stab = []
    for tag in ["GBM", "PC-GBM"]:
        piv = imp[imp.model == tag].pivot(index="feature", columns="event", values="mean_abs_shap")
        ev = piv.columns
        rhos = [spearmanr(piv[a], piv[b]).correlation for i, a in enumerate(ev) for b in ev[i + 1:]]
        # Kendall's W (rank concordance across events)
        R = piv.rank(axis=0).sum(axis=1); m_, n_ = len(ev), len(piv)
        Wk = 12 * ((R - R.mean()) ** 2).sum() / (m_ ** 2 * (n_ ** 3 - n_))
        stab.append(dict(model=tag, mean_pairwise_spearman=np.mean(rhos), sd=np.std(rhos), kendall_W=Wk))
    stab = pd.DataFrame(stab); stab.to_csv(os.path.join(RES_DIR, "e3_attribution_stability.csv"), index=False)
    return imp, stab


def e3b_monotonicity_violations(df):
    """Fraction of held-out cells where increasing PGA by 10 % lowers predicted probability."""
    rows = []
    for te in df.event.unique():
        tr, ts = df[df.event != te], df[df.event == te]
        w = event_balanced_weights(tr)
        sub = ts.sample(min(len(ts), 20000), random_state=0)
        up = sub.copy(); up["pga"] *= 1.1; up["pgv"] *= 1.1
        up["log_pga"] = np.log10(np.clip(up.pga, 1e-3, None)); up["log_pgv"] = np.log10(np.clip(up.pgv, 1e-2, None))
        # Newmark displacement recomputed consistently
        a = np.deg2rad(np.clip(up.slope, 0.5, 89)); ac = np.clip(up.ac, 0, None)
        ratio = np.where(up.pga > 0, ac / up.pga, np.inf)
        with np.errstate(all="ignore"):
            dn = np.where((ratio < 1) & (ratio > 0), 10 ** (0.215 + np.log10((1 - ratio) ** 2.341 * ratio ** -1.438)),
                          np.where(ratio <= 0, 10 ** 2.5, 0.0))
        up["newmark_cm"] = dn; up["log_newmark"] = np.log10(dn + 0.1)
        for name in ["GBM", "PC-GBM", "Global LR"]:
            f = MODELS[name](tr, w)
            viol = np.mean(f(up) < f(sub) - 1e-9)
            rows.append(dict(event=te, model=name, frac_violation=viol))
    r = pd.DataFrame(rows); r.to_csv(os.path.join(RES_DIR, "e3_monotonicity_violations.csv"), index=False)
    return r


def predict_kaikoura_map(df):
    """Train PC-GBM on the eight non-NZ events and store transfer predictions for Kaikoura cells."""
    tr = df[df.event != "kaikoura2016"]; ts = df[df.event == "kaikoura2016"]
    f = fit_xgb(tr, event_balanced_weights(tr), mono=True)
    out = ts[["x", "y_utm", "y", "w"]].copy(); out["p_transfer"] = f(ts)
    out.to_parquet(os.path.join(RES_DIR, "kaikoura_transfer_pred.parquet"))


if __name__ == "__main__":
    df = load()
    summ = df.groupby("event").agg(n=("y", "size"), pos=("y", "sum")).reset_index()
    print(summ)
    r1, preds = e1_loeo(df)
    print(r1.groupby(["setting", "model"])[["auc", "ap_lift", "bss", "e_over_o", "cal_slope"]].median())
    bd = block_bootstrap_auc_diff(df, preds); print(bd)
    bd2 = block_bootstrap_auc_diff(df, preds, a="PC-GBM", b="GBM"); print(bd2)
    r2 = e2_fewshot(df, preds)
    print(r2.groupby(["method", "k"])[["auc", "bss", "e_over_o"]].median())
    imp, stab = e3_shap(df); print(stab)
    mv = e3b_monotonicity_violations(df); print(mv.groupby("model").frac_violation.describe())
    predict_kaikoura_map(df)
