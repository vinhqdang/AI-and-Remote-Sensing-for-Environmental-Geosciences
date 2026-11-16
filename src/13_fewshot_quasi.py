"""E7: selection-aware Bayesian updating with a quasi-likelihood overdispersion correction.
The cell-level log-likelihood of the mapped tiles is divided by a dispersion factor phi, the
Pearson dispersion of tile-level landslide counts estimated from the out-of-sample LOEO scores of
the other eight events (median). Point estimates keep the score-equation property of the
cell-level likelihood; only the spread of the posterior is widened. Tile draws reproduce E5."""
import os, sys, importlib, json
import numpy as np, pandas as pd
from scipy.special import expit
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(__file__))
X = importlib.import_module("04_experiments")
H = importlib.import_module("hiphyboost")
R = X.RES_DIR
QS = (0.05, 0.5, 0.95); GRID = np.linspace(-20, 20, 1601); KS = (1, 2, 5, 10, 20, 40); TILE = 3000


def main(reps=30, only=None):
    df = X.load().reset_index(drop=True)
    events = sorted(df.event.unique())
    sc = pd.read_parquet(os.path.join(R, "hiphy_loeo_scores.parquet"))
    offs = pd.read_csv(os.path.join(R, "hiphy_loeo_offsets.csv"))
    data, phi = {}, {}
    for e in events:
        ts = df[df.event == e].reset_index(drop=True)
        m = sc[sc.event == e].m.values; y = ts.y.values.astype(float); w = ts.w.values.astype(float)
        _, inv = np.unique(X.blocks(ts, TILE), return_inverse=True); inv = inv.reshape(-1)
        b = brentq(lambda bb: np.sum(w * (expit(m + bb) - y)), -25, 25); p = expit(m + b)
        O = np.bincount(inv, weights=w * y); E = np.bincount(inv, weights=w * p); V = np.bincount(inv, weights=w * p * (1 - p))
        phi[e] = float(np.sum((O - E) ** 2) / np.sum(V))
        data[e] = (m, y, w, inv, b)
    rng = np.random.default_rng(7); draws = {}
    for target in events:                      # reproduce the E5 random sequence over all events
        _, y_, _, inv_, _ = data[target]
        ub_ = np.unique(inv_); pt_ = np.unique(inv_[y_ == 1])
        for k in KS:
            if k >= len(ub_) // 2 or len(pt_) == 0:
                continue
            for rep in range(reps):
                draws[(target, k, rep)] = rng.choice(pt_, min(k, len(pt_)), replace=False)
    rows, meta = [], {}
    for target in (only or events):
        m_all, y, w, inv, b_or = data[target]
        o = offs[offs.target == target].iloc[0]; mu, tau = o.mu, o.tau
        ph = float(np.median([phi[e] for e in events if e != target]))
        meta[target] = dict(phi=ph, phi_target_oracle=phi[target], oracle_offset=b_or)
        tq = H.tile_q_matrix(m_all, w, inv, GRID)
        ub = np.unique(inv); pos_tiles = np.unique(inv[y == 1])
        for k in KS:
            if k >= len(ub) // 2 or len(pos_tiles) == 0:
                continue
            for rep in range(reps):
                chosen = draws[(target, k, rep)]
                cal = np.isin(inv, chosen); ev = ~cal
                if y[ev].sum() < 5:
                    continue
                we, ye = w[ev], y[ev]; obs_e = np.sum(we * ye)
                LLb, A = H.grid_loglik(m_all[cal], y[cal], w[cal], inv[cal], GRID)
                sel = H.selection_logterm(tq[0], tq[1], chosen)
                prior = -0.5 * ((GRID - mu) / tau) ** 2
                for name, lp in [("HiPhyBoost + quasi-Bayes (no selection term)", LLb / ph + prior),
                                 ("HiPhyBoost (full): selection-aware quasi-Bayes", (LLb - sel) / ph + prior)]:
                    lp = lp - lp.max(); post = np.exp(lp); post /= post.sum()
                    p = H.mixture_predict(m_all[ev], GRID, post)
                    iv = H.posterior_interval(m_all[ev], we, GRID, post, QS) / obs_e
                    r = X.wmetrics(ye, p, we)
                    r.update(event=target, k=k, rep=rep, method=name, n_ls_mapped=int(y[cal].sum()),
                             eo_q05=iv[0], eo_q50=iv[1], eo_q95=iv[2], covered=float(iv[0] <= 1 <= iv[2]),
                             b_post_mean=float(np.sum(GRID * post)), b_oracle=b_or, phi=ph)
                    rows.append(r)
        print("E7", target, "done phi=%.1f" % ph, flush=True)
    tag = "_".join(only) if only else "all"
    os.makedirs(os.path.join(R, "e7_parts"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(R, "e7_parts", f"e7_{tag}.csv"), index=False)
    json.dump(meta, open(os.path.join(R, "e7_parts", f"phi_{tag}.json"), "w"), indent=1)


def merge():
    import glob
    pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, "e7_parts", "e7_*.csv")))],
              ignore_index=True).to_csv(os.path.join(R, "e7_fewshot_quasi.csv"), index=False)
    meta = {}
    for f in sorted(glob.glob(os.path.join(R, "e7_parts", "phi_*.json"))):
        meta.update(json.load(open(f)))
    json.dump(meta, open(os.path.join(R, "e7_phi.json"), "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        merge()
    else:
        main(only=sys.argv[1:] or None)
