"""Spatial block bootstrap of AUC differences: HiPhyBoost (LOEO scores) vs transferred PC-GBM and GBM."""
import os, sys, importlib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
X = importlib.import_module("04_experiments")
R = X.RES_DIR
rng = np.random.default_rng(11)
df = X.load()
sc = pd.read_parquet(os.path.join(R, "hiphy_loeo_scores.parquet"))
pc = pd.read_parquet(os.path.join(R, "loeo_preds_lr_pcgbm.parquet"))
rows = []
for e in df.event.unique():
    ts = df[df.event == e].reset_index(drop=True)
    a = sc[sc.event == e].m.values
    b = pc[(pc.event == e) & (pc.model == "PC-GBM")].p.values
    assert len(a) == len(b) == len(ts)
    bl = X.blocks(ts, 5000); ub, inv = np.unique(bl, return_inverse=True)
    idx = [np.flatnonzero(inv == i) for i in range(len(ub))]
    y, w = ts.y.values, ts.w.values
    d = []
    for _ in range(200):
        s = np.concatenate([idx[i] for i in rng.integers(0, len(ub), len(ub))])
        if y[s].sum() == 0: continue
        d.append(roc_auc_score(y[s], a[s], sample_weight=w[s]) - roc_auc_score(y[s], b[s], sample_weight=w[s]))
    d = np.array(d)
    rows.append(dict(event=e, auc_hiphy=roc_auc_score(y, a, sample_weight=w), auc_pcgbm=roc_auc_score(y, b, sample_weight=w),
                     diff=np.median(d), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(R, "e4_auc_diff_hiphy_vs_pcgbm.csv"), index=False)
