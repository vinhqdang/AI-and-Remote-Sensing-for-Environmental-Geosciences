"""HiPhyBoost: physics-anchored boosted logistic model with latent event offsets.

logit p_i = phi(z_i) + g(x_i) + b_{e(i)}

phi(z) = alpha + beta * z (beta >= 0) is the physical Newmark term, g is a monotone-constrained
XGBoost residual and b_e is a latent event intercept (random effect across earthquakes).
Fitting alternates weighted logistic regression for phi, boosting for g and a per-event
root-find for b_e. Event offsets are then used for zero-shot prediction in unseen events
(Gauss-Hermite integration over N(mu, tau^2)) and for Bayesian calibration of the offset from
a few mapped tiles, with an optional selection correction for tiles chosen because they
contain at least one landslide.
"""
import os, sys, importlib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse
from scipy.optimize import minimize, brentq
from scipy.special import expit
from scipy.stats import norm
from numpy.polynomial.hermite_e import hermegauss

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
X = importlib.import_module("04_experiments")

GRID_DEFAULT = np.linspace(-25, 25, 4001)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _mono_string(feats):
    return "(" + ",".join(str(X.MONO.get(f, 0)) for f in feats) + ")"


# ---------------------------------------------------------------- posterior helpers (module level)
def grid_loglik(m, y, w, tile_ids, grid, chunk=100):
    """Grid log-likelihood pieces for the event offset b.

    m        : score (phi + g) of the mapped cells
    tile_ids : tile label per cell
    Returns LLb (G,) = sum_i w_i [y_i log s_i + (1-y_i) log(1-s_i)], s_i = sigmoid(m_i + b),
    and A (T, G) = per-tile sums sum_{i in tile} w_i log(1-s_i) used for the selection term.
    Uses log s_i = log(1-s_i) + m_i + b, so only one logaddexp per cell and grid value.
    """
    m = np.asarray(m, float); y = np.asarray(y, float); w = np.asarray(w, float)
    _, tinv = np.unique(np.asarray(tile_ids), return_inverse=True)
    tinv = tinv.reshape(-1)
    T = int(tinv.max()) + 1
    n = len(m)
    S = sparse.csr_matrix((np.ones(n), (tinv, np.arange(n))), shape=(T, n))
    G = len(grid)
    LLb = np.empty(G); A = np.empty((T, G))
    Wy = np.sum(w * y); Cm = np.sum(w * y * m)
    for s in range(0, G, chunk):
        bg = grid[s:s + chunk]
        l1m = -np.logaddexp(0, m[:, None] + bg[None, :])     # log(1 - s_i(b))
        wl = w[:, None] * l1m
        LLb[s:s + chunk] = wl.sum(0) + bg * Wy + Cm
        A[:, s:s + chunk] = S @ wl
    return LLb, A


def tile_q_matrix(m_all, w_all, tile_ids_all, grid, chunk=100):
    """Per-tile probability q_t(b) that tile t contains >= 1 landslide, for every grid value.
    Returns (unique tile ids, Q (T, G)). Computed once per event from all domain cells."""
    tiles, tinv = np.unique(np.asarray(tile_ids_all), return_inverse=True)
    tinv = tinv.reshape(-1)
    n = len(m_all)
    S = sparse.csr_matrix((np.ones(n), (tinv, np.arange(n))), shape=(len(tiles), n))
    Q = np.empty((len(tiles), len(grid)))
    m_all = np.asarray(m_all, float); w_all = np.asarray(w_all, float)
    for s in range(0, len(grid), chunk):
        bg = grid[s:s + chunk]
        A = S @ (w_all[:, None] * -np.logaddexp(0, m_all[:, None] + bg[None, :]))
        Q[:, s:s + chunk] = -np.expm1(np.minimum(A, 0.0))
    return tiles, Q


def selection_logterm(tiles, Q, chosen):
    """log C(N(b), k) with N(b) = k + sum_{t not chosen} q_t(b): the probability of drawing the
    observed k tiles when an analyst samples k tiles uniformly among all landslide-bearing tiles
    is 1 / C(N_pos, k); N_pos is replaced by its expectation under b."""
    from scipy.special import gammaln
    chosen = np.unique(np.asarray(chosen))
    k = len(chosen)
    mask = np.isin(tiles, chosen)
    n = k + Q[~mask].sum(0)
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def posterior_from_ll(LLb, A, grid, mu, tau, selection_correction=True, prior=True, sel_term=None):
    """Normalised posterior (softmax) over the grid of b.
    selection_correction: subtract sel_term(b) = log C(N(b), k) (uniform choice among positive
    tiles, see selection_logterm). If sel_term is None the per-tile conditional correction
    sum_t log P(tile t positive | b) is used instead (tiles forced to be positive)."""
    lp = np.array(LLb, dtype=float)
    if selection_correction and sel_term is not None:
        lp -= sel_term
    elif selection_correction and A.shape[0] > 0:
        a = np.minimum(A, -1e-12)
        lp -= np.sum(np.log(-np.expm1(a)), axis=0)
    if prior:
        s = max(float(tau), 1e-6)
        lp += -0.5 * ((grid - mu) / s) ** 2
    lp -= lp.max()
    p = np.exp(lp)
    return p / p.sum()


def mixture_predict(m, grid, post, prune=1e-10, chunk=50000):
    """p_i = sum_g post_g sigmoid(m_i + grid_g)."""
    m = np.asarray(m, float)
    keep = post > prune * post.max()
    g, pw = grid[keep], post[keep] / post[keep].sum()
    out = np.empty(len(m))
    for s in range(0, len(m), chunk):
        out[s:s + chunk] = expit(m[s:s + chunk, None] + g[None, :]) @ pw
    return out


def posterior_interval(m, w, grid, post, qs=(0.05, 0.5, 0.95)):
    """Quantiles of the total T(b) = sum_i w_i sigmoid(m_i + b) under the posterior of b.
    T is monotone in b, so T(quantile of b) gives the quantile of T."""
    m = np.asarray(m, float); w = np.asarray(w, float)
    cdf = np.cumsum(post); cdf = cdf / cdf[-1]
    bq = np.interp(qs, cdf, grid)
    return np.array([np.sum(w * expit(m + b)) for b in bq])


# ---------------------------------------------------------------- model
class HiPhyBoost:
    def __init__(self, T=3, rounds=X.N_ROUNDS, rounds_early=150, learn_offsets=True,
                 nthread=None, feats=None, seed=0):
        self.T = T
        self.rounds = rounds
        self.rounds_early = rounds_early
        self.learn_offsets = learn_offsets
        self.nthread = nthread
        self.feats = list(feats) if feats is not None else list(X.FEATS)
        self.seed = seed

    # ---- internals
    def _params(self):
        p = dict(X.XGB_PARAMS)
        p.update(monotone_constraints=_mono_string(self.feats), base_score=0.5, seed=self.seed)
        if self.nthread:
            p["nthread"] = self.nthread
        return p

    def _fit_booster(self, Xf, y, w, base, rounds):
        dm = xgb.DMatrix(Xf, label=y, weight=w, base_margin=base, feature_names=self.feats)
        return xgb.train(self._params(), dm, rounds)

    def _margin(self, bst, Xf):
        """Pure g(x): base_score is 0.5 (logit 0), so the output margin has no bias term."""
        return bst.predict(xgb.DMatrix(Xf, feature_names=self.feats), output_margin=True)

    @staticmethod
    def _fit_phi(z, y, w, o, a0, b0):
        def f(th):
            a, bb = th
            eta = a + bb * z + o
            nll = np.sum(w * (np.logaddexp(0, eta) - y * eta))
            r = w * (expit(eta) - y)
            return nll, np.array([r.sum(), (r * z).sum()])
        res = minimize(f, x0=np.array([a0, b0]), jac=True, method="L-BFGS-B",
                       bounds=[(None, None), (0, None)])
        return float(res.x[0]), float(res.x[1]), float(res.fun)

    @staticmethod
    def _solve_offsets(eta_base, y, w, eidx, K, lo=-25.0, hi=25.0):
        out = np.zeros(K)
        for k in range(K):
            idx = eidx == k
            ee, yy, ww = eta_base[idx], y[idx], w[idx]
            f = lambda bb: np.sum(ww * (expit(ee + bb) - yy))
            if f(lo) >= 0:
                out[k] = lo
            elif f(hi) <= 0:
                out[k] = hi
            else:
                out[k] = brentq(f, lo, hi)
        return out

    # ---- fitting
    def fit(self, d):
        d = d.reset_index(drop=True)
        w = X.event_balanced_weights(d).astype(float)
        y = d.y.values.astype(float)
        z = d.log_newmark.values.astype(float)
        Xf = d[self.feats].values.astype(float)
        events = sorted(d.event.unique())
        K = len(events)
        eidx = pd.Categorical(d.event, categories=events).codes.astype(int)

        prev = np.array([np.sum(w[eidx == k] * y[eidx == k]) / np.sum(w[eidx == k]) for k in range(K)])
        b = _logit(prev); b = b - b.mean()
        if not self.learn_offsets:
            b = np.zeros(K)
        g = np.zeros(len(d))
        alpha, beta = 0.0, 0.0
        self.history = []
        bst = None
        for t in range(self.T):
            o = g + b[eidx]
            alpha, beta, nll = self._fit_phi(z, y, w, o, alpha, beta)
            phi = alpha + beta * z
            rounds = self.rounds if t == self.T - 1 else min(self.rounds_early, self.rounds)
            bst = self._fit_booster(Xf, y, w, phi + b[eidx], rounds)
            g = self._margin(bst, Xf)
            rec = dict(iter=t, nll_phi=nll, alpha=alpha, beta=beta)
            if self.learn_offsets:
                b = self._solve_offsets(phi + g, y, w, eidx, K)
                c = b.mean()
                b = b - c
                alpha += c
            rec.update({f"b_{e}": v for e, v in zip(events, b)})
            self.history.append(rec)

        self.booster = bst
        self.alpha, self.beta = float(alpha), float(beta)
        self.events = events
        self.offsets = {e: float(v) for e, v in zip(events, b)}
        self.mu = float(np.mean(b))
        self.tau = float(np.std(b, ddof=1)) if K > 1 else 0.0
        return self

    # ---- scoring
    def score(self, d):
        """m_i = phi(z_i) + g(x_i): log-odds without the event offset."""
        Xf = d[self.feats].values.astype(float)
        z = d.log_newmark.values.astype(float)
        return self.alpha + self.beta * z + self._margin(self.booster, Xf)

    def predict_zero_shot(self, d, n_gh=40):
        m = self.score(d)
        x, wts = hermegauss(n_gh)
        wts = wts / np.sqrt(2 * np.pi)
        b = self.mu + self.tau * x
        return mixture_predict(m, b, wts, prune=0.0)

    def offset_quantiles(self, qs):
        return self.mu + self.tau * norm.ppf(np.asarray(qs, float))

    def total_interval(self, d, w, qs=(0.05, 0.5, 0.95)):
        m = self.score(d)
        w = np.asarray(w, float)
        return np.array([np.sum(w * expit(m + bq)) for bq in self.offset_quantiles(qs)])

    def posterior_offset(self, d_cal, tile_ids, selection_correction=True, prior=True,
                         grid=GRID_DEFAULT, tileQ=None, m_cal=None):
        """tileQ = (tiles, Q) from tile_q_matrix over the whole target domain enables the
        uniform-selection correction; otherwise the per-tile conditional correction is used."""
        m = self.score(d_cal) if m_cal is None else m_cal
        LLb, A = grid_loglik(m, d_cal.y.values, d_cal.w.values, tile_ids, grid)
        sel = selection_logterm(tileQ[0], tileQ[1], tile_ids) if (selection_correction and tileQ is not None) else None
        post = posterior_from_ll(LLb, A, grid, self.mu, self.tau, selection_correction, prior, sel)
        return grid, post

    def predict_with_posterior(self, d, grid, post):
        return mixture_predict(self.score(d), grid, post)

    def total_interval_posterior(self, d, w, grid, post, qs=(0.05, 0.5, 0.95)):
        return posterior_interval(self.score(d), w, grid, post, qs)


def fit_predict_loeo(df, target, **kw):
    tr = df[df.event != target]; ts = df[df.event == target]
    mdl = HiPhyBoost(**kw).fit(tr)
    return mdl, mdl.score(ts)


# ---------------------------------------------------------------- self-test
def _selftest(seed=11, n_per_event=60000, rounds=200, T=2, nreps=60, c0=-5.3, tile=1000.0):
    rng = np.random.default_rng(seed)
    truth = {"ev_A": -2.0, "ev_B": -0.5, "ev_C": 0.5, "ev_D": 2.0}   # training events (centred)
    held = ("ev_H", 1.0)                                              # held-out event
    L = 30000.0                                                       # 30 km square; 1 km tiles (~1 landslide per tile)

    def simulate(name, b_true, n):
        d = pd.DataFrame(index=range(n))
        for f in X.FEATS:
            d[f] = rng.normal(size=n)
        d["slope"] = rng.uniform(0, 45, n)
        d["relief510"] = rng.gamma(2.0, 100.0, n)
        d["log_pga"] = rng.normal(-0.3, 0.25, n)
        d["log_pgv"] = rng.normal(0.0, 0.3, n)
        d["log_newmark"] = rng.normal(0.0, 0.7, n)
        eta = (c0 + 0.9 * d.log_newmark + 0.5 * (d.slope - 20) / 10 + 0.6 * d.log_pga / 0.25 * 0.3
               + 0.4 * np.tanh((d.relief510 - 200) / 100) + b_true)
        d["y"] = (rng.random(n) < expit(eta)).astype(int)
        d["event"] = name
        d["x"] = rng.uniform(0, L, n); d["y_utm"] = rng.uniform(0, L, n)
        d["w"] = 1.0
        return d

    tr = pd.concat([simulate(e, b, n_per_event) for e, b in truth.items()], ignore_index=True)
    ho = simulate(held[0], held[1], n_per_event)
    print("prevalence per event:", {e: round(float(tr[tr.event == e].y.mean()), 4) for e in truth},
          "held-out:", round(float(ho.y.mean()), 4))

    # 1. recovery of event offsets
    mdl = HiPhyBoost(T=T, rounds=rounds, nthread=2).fit(tr)
    est = np.array([mdl.offsets[e] for e in truth]); tru = np.array(list(truth.values()))
    r = np.corrcoef(est, tru)[0, 1]
    print("recovered offsets:", np.round(est, 3), "truth:", tru, "corr=%.4f" % r,
          "mu=%.4f tau=%.3f" % (mdl.mu, mdl.tau))
    assert r > 0.95, "offset recovery failed"

    # 2. base-margin identity: predict(output_margin) on base_margin m equals m + g
    Xf = tr[mdl.feats].values.astype(float)
    mm = rng.normal(0, 1, 500)
    import xgboost as _xgb
    g1 = mdl._margin(mdl.booster, Xf[:500])
    g2 = mdl.booster.predict(_xgb.DMatrix(Xf[:500], base_margin=mm, feature_names=mdl.feats), output_margin=True)
    print("base-margin identity max|err| = %.2e" % np.abs(g2 - mm - g1).max())
    assert np.abs(g2 - mm - g1).max() < 1e-4

    # 3. zero-shot prediction and interval sanity
    pz = mdl.predict_zero_shot(ho)
    print("zero-shot p range [%.5f, %.5f], mean %.5f (observed %.5f)" % (pz.min(), pz.max(), pz.mean(), ho.y.mean()))
    assert np.all((pz > 0) & (pz < 1))
    iv = mdl.total_interval(ho, ho.w.values)
    print("total interval (q05,q50,q95) / observed:", np.round(iv / ho.y.sum(), 3))
    assert np.all(np.diff(iv) >= 0)

    # 4. few-shot offset recovery on held-out event: 5 tiles containing >= 1 landslide.
    #    Two true held-out offsets: 0 (no shrinkage towards the prior mean) and 1.
    for hb in (0.0, 1.0):
        ho = simulate(held[0], hb, n_per_event)
        inv = np.unique(X.blocks(ho, tile), return_inverse=True)[1].reshape(-1)
        pos_t = np.unique(inv[ho.y.values == 1])
        tq = tile_q_matrix(mdl.score(ho), ho.w.values, inv, GRID_DEFAULT)
        res = {"sel (prior)": [], "naive (prior)": [], "sel (no prior)": [], "naive/MLE (no prior)": []}
        cover = {"sel (prior)": [], "naive (prior)": []}
        for rep in range(nreps):
            ch = rng.choice(pos_t, 5, replace=False)
            cal = np.isin(inv, ch)
            dc = ho[cal]
            de = ho[~cal]
            obs_e = de.y.sum() * 1.0
            g_sel, p_sel = mdl.posterior_offset(dc, inv[cal], selection_correction=True, prior=True, tileQ=tq)
            g_nai, p_nai = mdl.posterior_offset(dc, inv[cal], selection_correction=False, prior=True)
            g_sn, p_sn = mdl.posterior_offset(dc, inv[cal], selection_correction=True, prior=False, tileQ=tq)
            g_mle, p_mle = mdl.posterior_offset(dc, inv[cal], selection_correction=False, prior=False)
            for nm, gg, pp in [("sel (prior)", g_sel, p_sel), ("naive (prior)", g_nai, p_nai)]:
                iv = mdl.total_interval_posterior(de, de.w.values, gg, pp, (0.05, 0.5, 0.95)) / obs_e
                cover[nm].append(iv[0] <= 1 <= iv[2])
            res["sel (prior)"].append(np.sum(g_sel * p_sel) - hb)
            res["naive (prior)"].append(np.sum(g_nai * p_nai) - hb)
            res["sel (no prior)"].append(g_sn[np.argmax(p_sn)] - hb)
            res["naive/MLE (no prior)"].append(g_mle[np.argmax(p_mle)] - hb)
        print("held-out true offset %.1f, %d reps of 5 positive tiles:" % (hb, nreps))
        for k, v in res.items():
            v = np.array(v)
            print("   %-22s mean bias %+.3f  median abs err %.3f  RMSE %.3f  share |err|<0.5 = %.2f"
                  % (k, v.mean(), np.median(np.abs(v)), np.sqrt((v ** 2).mean()), np.mean(np.abs(v) < 0.5)))
        print("   selection-corrected (prior) |bias| < naive (prior) |bias|:",
              abs(np.mean(res["sel (prior)"])) < abs(np.mean(res["naive (prior)"])))
        print("   90%% posterior interval of eval total covers observed: sel %.2f, naive %.2f"
              % (np.mean(cover["sel (prior)"]), np.mean(cover["naive (prior)"])))

    # 5. ablation without learned offsets runs and keeps offsets at zero
    m0 = HiPhyBoost(T=1, rounds=50, learn_offsets=False, nthread=2).fit(tr)
    assert all(v == 0.0 for v in m0.offsets.values())
    print("no-offset ablation fitted, tau=%.1f" % m0.tau)
    print("SELFTEST OK")


if __name__ == "__main__":
    _selftest()


# ---------------------------------------------------------------- tile-level overdispersion
# Landslides cluster in space beyond what the covariates explain. Treating every 30 m cell of a
# mapped tile as an independent Bernoulli trial then makes the offset posterior far too narrow.
# We add a tile random effect u_t ~ N(0, sigma_u^2):  logit p_i = m_i + b + u_{t(i)}, and
# integrate it out per tile with Gauss-Hermite quadrature, so that the tile becomes the unit of
# information.
SHIFT = np.linspace(-30, 30, 2401)


def tile_shift_loglik(m, y, w, tile_ids, shifts=SHIFT, chunk=100):
    """Per-tile log-likelihood L_t(s) = sum_{i in t} w_i [y_i (m_i+s) - log(1+exp(m_i+s))] and
    per-tile log P(no landslide) A_t(s) = sum_{i in t} w_i log(1-sigma(m_i+s)) on a shift grid."""
    tiles, tinv = np.unique(np.asarray(tile_ids), return_inverse=True)
    tinv = tinv.reshape(-1)
    n = len(m)
    S = sparse.csr_matrix((np.ones(n), (tinv, np.arange(n))), shape=(len(tiles), n))
    m = np.asarray(m, float); y = np.asarray(y, float); w = np.asarray(w, float)
    L = np.empty((len(tiles), len(shifts))); A = np.empty((len(tiles), len(shifts)))
    wy = S @ (w * y); wym = S @ (w * y * m)
    for s0 in range(0, len(shifts), chunk):
        sg = shifts[s0:s0 + chunk]
        l1m = -np.logaddexp(0, m[:, None] + sg[None, :])
        a = S @ (w[:, None] * l1m)
        A[:, s0:s0 + chunk] = a
        L[:, s0:s0 + chunk] = a + wym[:, None] + wy[:, None] * sg[None, :]
    return tiles, L, A


def _marginal_over_u(F, grid, sigma_u, n_gh=None, shifts=SHIFT, log_space=True):
    """For each row f_t(s) on the (uniform) shift grid, return log E_u[exp f_t(b+u)] (log_space)
    or E_u[f_t(b+u)] for every b in grid, u ~ N(0, sigma_u^2). The expectation is computed as an
    exact discrete Gaussian convolution on the fine shift grid; tile log-likelihoods are far too
    peaked for low-order quadrature."""
    from scipy.ndimage import gaussian_filter1d
    F = np.atleast_2d(np.asarray(F, float))
    if sigma_u > 0:
        sd = sigma_u / (shifts[1] - shifts[0])
        if log_space:
            mx = F.max(axis=1, keepdims=True)
            E = gaussian_filter1d(np.exp(F - mx), sd, axis=1, mode="constant", truncate=6.0)
            F = np.log(np.maximum(E, 1e-300)) + mx
        else:
            F = gaussian_filter1d(F, sd, axis=1, mode="nearest", truncate=6.0)
    return np.vstack([np.interp(grid, shifts, f) for f in F])


def sigma_profile(m, y, w, tid, be, sig_grid=np.arange(0.0, 3.01, 0.25)):
    """Marginal log-likelihood of one event's tiles as a function of sigma_u (offset fixed at be)."""
    _, L, _ = tile_shift_loglik(m, y, w, tid)
    return np.array([_marginal_over_u(L, np.array([be]), sgm).sum() for sgm in sig_grid])


def estimate_sigma_u(events_data, tile=3000.0, sig_grid=np.arange(0.0, 3.01, 0.25)):
    """Maximum marginal likelihood estimate of the tile effect sd from events with complete
    inventories. events_data: list of (m, y, w, tile_ids, b_e) with b_e the event offset."""
    tot = np.zeros(len(sig_grid))
    for (m, y, w, tid, be) in events_data:
        tiles, L, _ = tile_shift_loglik(m, y, w, tid)
        for k, sgm in enumerate(sig_grid):
            tot[k] += _marginal_over_u(L, np.array([be]), sgm).sum()
    return float(sig_grid[np.argmax(tot)]), tot


def posterior_offset_od(m_cal, y_cal, w_cal, tile_cal, chosen, tileA_all, mu, tau, sigma_u,
                        grid=np.linspace(-20, 20, 1601), selection_correction=True, prior=True):
    """Selection-aware posterior of b with tile-level overdispersion.
    tileA_all = (tiles_all, P0_all) with P0_all = tile_p0_matrix(...) over the whole target domain."""
    from scipy.special import gammaln
    _, Lc, _ = tile_shift_loglik(m_cal, y_cal, w_cal, tile_cal)
    lp = _marginal_over_u(Lc, grid, sigma_u).sum(0)
    if selection_correction:
        tiles_all, P0_all = tileA_all          # P0_all[t, g] = E_u[P(no landslide in t | b_g + u)]
        mask = ~np.isin(tiles_all, np.unique(chosen))
        P0 = P0_all[mask]
        k = len(np.unique(chosen))
        n = k + np.sum(1 - P0, axis=0)
        lp = lp - (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))
    if prior:
        lp = lp - 0.5 * ((grid - mu) / max(tau, 1e-6)) ** 2
    lp -= lp.max(); p = np.exp(lp)
    return grid, p / p.sum()


def predict_od(m, grid, post, sigma_u, n_gh=21):
    """Predictive probability for unmapped cells: E_{b|C} E_u sigma(m + b + u)."""
    x, wts = hermegauss(n_gh); wts = wts / wts.sum()
    keep = post > 1e-10 * post.max(); g, pw = grid[keep], post[keep] / post[keep].sum()
    shifts = (g[:, None] + sigma_u * x[None, :]).ravel(); ww = (pw[:, None] * wts[None, :]).ravel()
    order = np.argsort(shifts); shifts, ww = shifts[order], ww[order]
    # compress the mixing distribution to <= 40 support points (equal-probability bins)
    if len(shifts) > 40:
        c = np.cumsum(ww) / ww.sum(); idx = np.minimum((c * 40).astype(int), 39)
        s2 = np.bincount(idx, weights=shifts * ww, minlength=40); w2 = np.bincount(idx, weights=ww, minlength=40)
        ok = w2 > 0; shifts, ww = s2[ok] / w2[ok], w2[ok]
    out = np.empty(len(m))
    for s0 in range(0, len(m), 50000):
        out[s0:s0 + 50000] = expit(m[s0:s0 + 50000, None] + shifts[None, :]) @ ww
    return out


def tile_p0_matrix(A_all, grid, sigma_u):
    """E_u[exp(A_t(b+u))]: probability that tile t holds no landslide, for every grid value b."""
    return _marginal_over_u(np.exp(np.minimum(A_all, 0)), grid, sigma_u, log_space=False)
