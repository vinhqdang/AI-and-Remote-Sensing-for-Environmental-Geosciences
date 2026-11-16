"""HiPhyBoost experiments.

E4  Leave-one-event-out (LOEO) transfer of HiPhyBoost (plug-in, zero-shot Bayes, and the
    no-offset ablation), with 90 % predictive intervals of the total landslide count.
    Also saves the training-event offsets per fold and one all-event offset fit.
E5  Few-shot calibration from k rapidly mapped 3 km tiles, identical tile draws for all methods.

Full run:   python3 src/08_hiphyboost_experiments.py
Quick test: python3 src/08_hiphyboost_experiments.py --quick --nthread 2
            (3 events, 40k rows per event, T=1, rounds=50, reps=2; outputs go to results/quick_hiphy
             unless --out is given)
"""
import os, sys, argparse, warnings, importlib
import numpy as np, pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
X = importlib.import_module("04_experiments")
H = importlib.import_module("hiphyboost")
warnings.filterwarnings("ignore")

QS = (0.05, 0.5, 0.95)
GRID = np.linspace(-20, 20, 1601)     # spacing 0.025 on [-20, 20] (reduced from 4001 points for speed)
KS = (1, 2, 5, 10, 20, 40)
TILE = 3000
E4_METRICS = ["auc", "ap_lift", "bss", "e_over_o", "cal_slope"]


def quick_subsample(df, events, n_per=40000, max_pos=3000, seed=0):
    """Keep up to max_pos positives per event and fill to n_per rows with negatives (no weight change)."""
    parts = []
    for e in events:
        d = df[df.event == e]
        pos, neg = d[d.y == 1], d[d.y == 0]
        if len(pos) > max_pos:
            pos = pos.sample(max_pos, random_state=seed)
        neg = neg.sample(min(max(n_per - len(pos), 0), len(neg)), random_state=seed)
        parts.append(pd.concat([pos, neg]))
    return pd.concat(parts, ignore_index=True)


def load_or_fit_pcgbm(df, target, ts, use_file):
    """LOEO PC-GBM predictions for the target event: from results/loeo_preds_lr_pcgbm.parquet if
    available and aligned with df[df.event == target], otherwise refitted on the other events."""
    p = os.path.join(X.RES_DIR, "loeo_preds_lr_pcgbm.parquet")
    if use_file and os.path.exists(p):
        pr = pd.read_parquet(p)
        sub = pr[(pr.model == "PC-GBM") & (pr.event == target)]
        if len(sub) == len(ts):
            return sub.p.values.astype(float)
        print("  PC-GBM file rows do not match for", target, "- refitting", flush=True)
    tr = df[df.event != target]
    return X.MODELS["PC-GBM"](tr, X.event_balanced_weights(tr))(ts)


# ---------------------------------------------------------------- E4
def e4_loeo(df, events, T, R, nt, out):
    rows, off_rows, score_parts, models = [], [], [], {}
    for target in events:
        tr = df[df.event != target]
        ts = df[df.event == target].reset_index(drop=True)
        y, w = ts.y.values, ts.w.values
        obs = np.sum(w * y)
        mdl = H.HiPhyBoost(T=T, rounds=R, rounds_early=150, nthread=nt).fit(tr)
        models[target] = mdl
        m = mdl.score(ts)
        n_tr = len(mdl.events)
        base = dict(event=target, n_train_events=n_tr, tau=mdl.tau)

        r = X.wmetrics(y, expit(m + mdl.mu), w)
        r.update(base, method="HiPhyBoost (plug-in)")
        rows.append(r)

        p_zs = mdl.predict_zero_shot(ts)
        iv = mdl.total_interval(ts, w, QS) / obs
        r = X.wmetrics(y, p_zs, w)
        r.update(base, method="HiPhyBoost (zero-shot Bayes)", eo_q05=iv[0], eo_q50=iv[1], eo_q95=iv[2],
                 covered=bool(iv[0] <= 1 <= iv[2]))
        rows.append(r)

        mdl0 = H.HiPhyBoost(T=T, rounds=R, rounds_early=150, learn_offsets=False, nthread=nt).fit(tr)
        r = X.wmetrics(y, expit(mdl0.score(ts) + mdl0.mu), w)
        r.update(base, method="Ablation: no offsets (PC-GBM-like)")
        rows.append(r)

        for e, b in mdl.offsets.items():
            off_rows.append(dict(target=target, train_event=e, b=b, tau=mdl.tau, mu=mdl.mu))
        score_parts.append(pd.DataFrame({"event": target, "m": m}))
        print("E4", target, "tau=%.3f" % mdl.tau,
              "offsets:", {e: round(v, 2) for e, v in mdl.offsets.items()}, flush=True)

    pd.concat(score_parts, ignore_index=True).to_parquet(os.path.join(out, "hiphy_loeo_scores.parquet"))
    pd.DataFrame(off_rows).to_csv(os.path.join(out, "hiphy_loeo_offsets.csv"), index=False)
    e4 = pd.DataFrame(rows)
    cols = ["event", "method", "n_train_events", "tau"] + E4_METRICS + \
           ["brier", "ap", "prevalence", "eo_q05", "eo_q50", "eo_q95", "covered"]
    e4 = e4[[c for c in cols if c in e4.columns]]
    e4.to_csv(os.path.join(out, "e4_hiphy_loeo.csv"), index=False)

    mall = H.HiPhyBoost(T=T, rounds=R, rounds_early=150, nthread=nt).fit(df)
    pd.DataFrame({"event": mall.events, "b_e": [mall.offsets[e] for e in mall.events],
                  "tau": mall.tau, "mu": mall.mu}).to_csv(os.path.join(out, "hiphy_offsets_all.csv"), index=False)
    return e4, models


# ---------------------------------------------------------------- E5
def e5_fewshot(df, events, models, reps, use_file, out):
    rng = np.random.default_rng(7)
    rows = []
    for target in events:
        ts = df[df.event == target].reset_index(drop=True)
        mdl = models[target]
        m_all = mdl.score(ts)
        p_pc = np.clip(load_or_fit_pcgbm(df, target, ts, use_file), 1e-7, 1 - 1e-7)
        s = np.log(p_pc / (1 - p_pc))                      # logit of PC-GBM LOEO prediction, as in E2
        y = ts.y.values.astype(float); w = ts.w.values.astype(float)
        _, inv = np.unique(X.blocks(ts, TILE), return_inverse=True); inv = inv.reshape(-1)
        ub = np.unique(inv)
        pos_tiles = np.unique(inv[y == 1])
        tq = H.tile_q_matrix(m_all, w, inv, GRID)          # q_t(b) for every tile, once per event
        for k in KS:
            if k >= len(ub) // 2 or len(pos_tiles) == 0:
                continue
            for rep in range(reps):
                chosen = rng.choice(pos_tiles, min(k, len(pos_tiles)), replace=False)
                cal = np.isin(inv, chosen); ev = ~cal
                if y[ev].sum() < 5:
                    continue
                yc, wc, sc = y[cal], w[cal], s[cal]
                we, ye = w[ev], y[ev]
                obs_e = np.sum(we * ye)
                nls = int(yc.sum())
                preds = {}   # method -> (probabilities, interval/observed or None)

                def gfun(a):
                    return np.sum(wc * expit(sc + a)) - np.sum(wc * yc)
                try:
                    a = brentq(gfun, -15, 15)
                except ValueError:
                    a = 0.0
                preds["PC-GBM + intercept recal."] = (expit(s[ev] + a), None)
                lr = LogisticRegression(C=1.0).fit(sc[:, None], yc, sample_weight=wc / wc.mean())
                preds["PC-GBM + Platt recal."] = (lr.predict_proba(s[ev][:, None])[:, 1], None)

                m_c, m_e = m_all[cal], m_all[ev]
                LLb, A = H.grid_loglik(m_c, yc, wc, inv[cal], GRID)
                sel = H.selection_logterm(tq[0], tq[1], chosen)
                post_sel = H.posterior_from_ll(LLb, A, GRID, mdl.mu, mdl.tau, True, True, sel_term=sel)
                post_nai = H.posterior_from_ll(LLb, A, GRID, mdl.mu, mdl.tau, False, True)
                post_mle = H.posterior_from_ll(LLb, A, GRID, mdl.mu, mdl.tau, False, False)
                for name, post in [("HiPhyBoost + Bayes (selection-corrected)", post_sel),
                                   ("HiPhyBoost + Bayes (naive)", post_nai)]:
                    p = H.mixture_predict(m_e, GRID, post)
                    iv = H.posterior_interval(m_e, we, GRID, post, QS) / obs_e
                    preds[name] = (p, iv)
                b_hat = GRID[np.argmax(post_mle)]
                preds["HiPhyBoost + MLE offset"] = (expit(m_e + b_hat), None)

                for name, (p, iv) in preds.items():
                    r = X.wmetrics(ye, p, we)
                    r.update(event=target, k=k, rep=rep, method=name, n_ls_mapped=nls)
                    if iv is not None:
                        r.update(eo_q05=iv[0], eo_q50=iv[1], eo_q95=iv[2], covered=float(iv[0] <= 1 <= iv[2]))
                    rows.append(r)
        print("E5", target, "done", flush=True)
    e5 = pd.DataFrame(rows)
    front = ["event", "k", "rep", "method", "n_ls_mapped"]
    e5 = e5[front + [c for c in e5.columns if c not in front]]
    e5.to_csv(os.path.join(out, "e5_hiphy_fewshot.csv"), index=False)
    return e5


# ---------------------------------------------------------------- summaries
def summarise(e4, e5):
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print("\n== E4 LOEO: median over events (plug-in / zero-shot / ablation) ==")
    t4 = e4.groupby(["method", "event"])[E4_METRICS].median().groupby(level=0).median()
    print(t4.round(3))
    z = e4[e4.method == "HiPhyBoost (zero-shot Bayes)"]
    if len(z):
        print("zero-shot 90%% interval coverage (share of events): %.2f ; median eo_q05 %.3f, eo_q95 %.3f"
              % (z.covered.mean(), z.eo_q05.median(), z.eo_q95.median()))

    print("\n== E5 few-shot: median over events of per-event medians ==")
    g = e5.groupby(["method", "k", "event"])
    t5 = g[E4_METRICS].median().groupby(level=[0, 1]).median()
    print(t5.round(3))
    cov = e5.dropna(subset=["covered"]).groupby(["method", "k", "event"]).covered.mean()
    if len(cov):
        print("\n90% interval coverage of eval total (median over events of per-event coverage):")
        print(cov.groupby(level=[0, 1]).median().unstack("k").round(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--nthread", type=int, default=4)
    args = ap.parse_args()
    if args.quick:
        T, R, reps = 1, 50, 2
        out = args.out or os.path.join(X.RES_DIR, "quick_hiphy")
    else:
        T, R, reps = 3, X.N_ROUNDS, 30
        out = args.out or X.RES_DIR
    os.makedirs(out, exist_ok=True)

    df = X.load()
    if args.quick:
        events = sorted(df.event.unique())[:3]
        df = quick_subsample(df, events)
    df = df.reset_index(drop=True)
    events = sorted(df.event.unique())
    print("events:", events, "rows:", len(df), "T=%d rounds=%d reps=%d out=%s" % (T, R, reps, out), flush=True)

    e4, models = e4_loeo(df, events, T, R, args.nthread, out)
    e5 = e5_fewshot(df, events, models, reps, use_file=not args.quick, out=out)
    summarise(e4, e5)
    print("\nwritten to", out)


if __name__ == "__main__":
    main()
