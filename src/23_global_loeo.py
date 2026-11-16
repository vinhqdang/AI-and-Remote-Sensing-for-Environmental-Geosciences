"""Global leave-one-event-out fits of HiPhyBoost and PC-GBM (GPU if available).

For each target event the two models are trained on all other events; saved per target:
  out/<target>_scores.parquet : HiPhyBoost score m (log-odds without offset) and PC-GBM probability
  out/<target>_offsets.csv    : HiPhyBoost offsets b_e of the training events (and mu, tau)
Also fits HiPhyBoost on all events (target "ALL") to obtain the full set of offsets.
Usage: python 23_global_loeo.py <data_dir> <out_dir> <device: cuda|cpu> <target ...|ALL>
data_dir must contain the per-event parquet files (all events of the global set).
"""
import os, sys, glob, importlib, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
X = importlib.import_module("04_experiments")
H = importlib.import_module("hiphyboost")
NEG_MAX = 150_000


def load_global(data_dir):
    dfs = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.parquet"))):
        d = pd.read_parquet(f)
        e = os.path.basename(f)[:-8]
        d["event"] = e
        neg = d[d.y == 0]
        if len(neg) > NEG_MAX:                      # harmonise sampling of the original nine events
            keep = neg.sample(NEG_MAX, random_state=0)
            keep = keep.assign(w=keep.w * len(neg) / NEG_MAX)
            d = pd.concat([d[d.y == 1], keep])
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df["log_pga"] = np.log10(np.clip(df.pga, 1e-3, None))
    df["log_pgv"] = np.log10(np.clip(df.pgv, 1e-2, None))
    df["log_newmark"] = np.log10(df.newmark_cm + 0.1)
    return df.dropna(subset=X.FEATS).reset_index(drop=True)


def main():
    data_dir, out_dir, device = sys.argv[1], sys.argv[2], sys.argv[3]
    targets = sys.argv[4:]
    os.makedirs(out_dir, exist_ok=True)
    if device == "cuda":
        X.XGB_PARAMS.update(device="cuda", tree_method="hist"); X.XGB_PARAMS.pop("nthread", None)
    df = load_global(data_dir)
    events = sorted(df.event.unique())
    print("events", len(events), "rows", len(df), flush=True)
    if targets == ["ALL"]:
        targets = ["ALL"]
    for t in targets:
        if os.path.exists(os.path.join(out_dir, f"{t}_offsets.csv")):
            continue
        t0 = time.time()
        tr = df if t == "ALL" else df[df.event != t]
        mdl = H.HiPhyBoost(T=3, rounds=X.N_ROUNDS, rounds_early=150).fit(tr)
        if t != "ALL":
            ts = df[df.event == t]
            pc = X.fit_xgb(tr, X.event_balanced_weights(tr), mono=True)(ts)
            pd.DataFrame({"m": mdl.score(ts), "p_pcgbm": pc, "y": ts.y.values, "w": ts.w.values,
                          "x": ts.x.values, "y_utm": ts.y_utm.values}).to_parquet(os.path.join(out_dir, f"{t}_scores.parquet"))
        pd.DataFrame({"train_event": list(mdl.offsets), "b": list(mdl.offsets.values()), "mu": mdl.mu, "tau": mdl.tau,
                      "alpha": mdl.alpha, "beta": mdl.beta}).to_csv(os.path.join(out_dir, f"{t}_offsets.csv"), index=False)
        print(t, "done in %.0f s, tau=%.2f" % (time.time() - t0, mdl.tau), flush=True)


if __name__ == "__main__":
    main()
