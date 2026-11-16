"""E6: selection-aware Bayesian updating with tile-level overdispersion (HiPhyBoost, full model).
Re-uses the saved LOEO HiPhyBoost scores (no refitting) and reproduces the exact tile draws of E5
(same RNG seed and loop order), so results can be merged with e5_hiphy_fewshot.csv.
sigma_u for each target is estimated by maximum marginal likelihood from the out-of-sample LOEO
scores of the other eight events (offset of each such event fixed at its full-inventory value)."""
import os, sys, importlib, json
import numpy as np, pandas as pd
from scipy.special import expit
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(__file__))
X = importlib.import_module("04_experiments")
H = importlib.import_module("hiphyboost")
R = X.RES_DIR
QS = (0.05, 0.5, 0.95); GRID = np.linspace(-20, 20, 1601); KS = (1, 2, 5, 10, 20, 40); TILE = 3000


def total_od(m, w, b, sigma_u, n_gh=21):
    from numpy.polynomial.hermite_e import hermegauss
    x, wt = hermegauss(n_gh); wt = wt / wt.sum()
    return float(np.sum(w * (expit(m[:, None] + b + sigma_u * x[None, :]) @ wt)))


def main(reps=30):
    df = X.load().reset_index(drop=True)
    events = sorted(df.event.unique())
    sc = pd.read_parquet(os.path.join(R, "hiphy_loeo_scores.parquet"))
    offs = pd.read_csv(os.path.join(R, "hiphy_loeo_offsets.csv"))
    data = {}
    for e in events:
        ts = df[df.event == e].reset_index(drop=True)
        m = sc[sc.event == e].m.values; y = ts.y.values.astype(float); w = ts.w.values.astype(float)
        _, inv = np.unique(X.blocks(ts, TILE), return_inverse=True); inv = inv.reshape(-1)
        b_or = brentq(lambda b: np.sum(w * (expit(m + b) - y)), -25, 25)
        data[e] = (m, y, w, inv, b_or)
    SIG = np.arange(0.0, 3.01, 0.25)
    prof = {e: H.sigma_profile(*data[e], sig_grid=SIG) for e in events}
    print("sigma profiles done", flush=True)
    rng = np.random.default_rng(7)
    rows, sig = [], {}
    for target in events:
        m_all, y, w, inv, b_or = data[target]
        o = offs[offs.target == target].iloc[0]; mu, tau = o.mu, o.tau
        tot = sum(prof[e] for e in events if e != target); s_u = float(SIG[np.argmax(tot)])
        sig[target] = dict(sigma_u=s_u, oracle_offset_target=b_or)
        tiles_all, _, A_all = H.tile_shift_loglik(m_all, y, w, inv)
        P0_all = H.tile_p0_matrix(A_all, GRID, s_u)
        ub = np.unique(inv); pos_tiles = np.unique(inv[y == 1])
        print(target, "sigma_u =", s_u, flush=True)
        for k in KS:
            if k >= len(ub) // 2 or len(pos_tiles) == 0:
                continue
            for rep in range(reps):
                chosen = rng.choice(pos_tiles, min(k, len(pos_tiles)), replace=False)
                cal = np.isin(inv, chosen); ev = ~cal
                if y[ev].sum() < 5:
                    continue
                we, ye = w[ev], y[ev]; obs_e = np.sum(we * ye)
                for name, selc in [("HiPhyBoost + Bayes + overdispersion (no selection term)", False),
                                   ("HiPhyBoost (full): selection-aware + overdispersion", True)]:
                    g, post = H.posterior_offset_od(m_all[cal], y[cal], w[cal], inv[cal], chosen, (tiles_all, P0_all),
                                                    mu, tau, s_u, grid=GRID, selection_correction=selc)
                    p = H.predict_od(m_all[ev], g, post, s_u)
                    cdf = np.cumsum(post); bq = np.interp(QS, cdf / cdf[-1], g)
                    iv = np.array([total_od(m_all[ev], we, b, s_u) for b in bq]) / obs_e
                    r = X.wmetrics(ye, p, we)
                    r.update(event=target, k=k, rep=rep, method=name, n_ls_mapped=int(y[cal].sum()),
                             eo_q05=iv[0], eo_q50=iv[1], eo_q95=iv[2], covered=float(iv[0] <= 1 <= iv[2]),
                             b_post_mean=float(np.sum(g * post)), b_oracle=b_or, sigma_u=s_u)
                    rows.append(r)
        print("E6", target, "done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(R, "e6_fewshot_overdispersed.csv"), index=False)
    json.dump(sig, open(os.path.join(R, "e6_sigma_u.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
