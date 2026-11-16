"""Summary table of few-shot updating (E5-E7): median over events of per-event medians of
|log2 E/O| and AUC, mean over events of 90% interval coverage, median interval width factor."""
import os, numpy as np, pandas as pd
R = os.path.join(os.path.dirname(__file__), "..", "results"); T = os.path.join(os.path.dirname(__file__), "..", "manuscript", "tables")
r = pd.concat([pd.read_csv(os.path.join(R, f)) for f in ["e5_hiphy_fewshot.csv", "e6_fewshot_overdispersed.csv", "e7_fewshot_quasi.csv"]], ignore_index=True)
r["ale"] = np.abs(np.log2(r.e_over_o)); r["wid"] = r.eo_q95 / r.eo_q05
g = r.groupby(["method", "k", "event"]).agg(ale=("ale", "median"), auc=("auc", "median"), cov=("covered", "mean"), wid=("wid", "median")).reset_index()
A = g.groupby(["method", "k"]).agg(ale=("ale", "median"), auc=("auc", "median"), cov=("cov", "mean"), wid=("wid", "median")).reset_index()
A.to_csv(os.path.join(R, "fewshot_summary.csv"), index=False)
M = [("PC-GBM + intercept recal.", "PC-GBM, intercept recalibration"), ("PC-GBM + Platt recal.", "PC-GBM, Platt recalibration"),
     ("HiPhyBoost + MLE offset", "HiPhyBoost, ML offset"), ("HiPhyBoost + Bayes (naive)", "HiPhyBoost, cell-level Bayes"),
     ("HiPhyBoost + Bayes (selection-corrected)", "\\quad + selection term"),
     ("HiPhyBoost + Bayes + overdispersion (no selection term)", "HiPhyBoost, tile random effect"),
     ("HiPhyBoost (full): selection-aware + overdispersion", "\\quad + selection term"),
     ("HiPhyBoost + quasi-Bayes (no selection term)", "HiPhyBoost, quasi-Bayes"),
     ("HiPhyBoost (full): selection-aware quasi-Bayes", "\\quad + selection term (full HiPhyBoost)")]
ks = [1, 5, 10, 40]
L = []
for key, lab in M:
    d = A[A.method == key].set_index("k")
    f = lambda col, k, fmt: (fmt % d.loc[k, col]) if (k in d.index and pd.notna(d.loc[k, col])) else "--"
    L.append(lab + " & " + " & ".join(f("ale", k, "%.2f") for k in ks) + " & " + " & ".join(f("cov", k, "%.2f") for k in ks) + " & " + " & ".join(f("wid", k, "%.0f" if (k in d.index and pd.notna(d.loc[k, "wid"]) and d.loc[k, "wid"] >= 10) else "%.1f") for k in [1, 10]) + "\\\\")
s = ("\\begin{table*}[t]\\centering\\small\n\\caption{Few-shot updating from $k$ rapidly mapped $3\\times3$~km tiles (Experiment~2; identical tile draws for all methods, 30 repetitions per event and $k$). "
     "$|\\log_2 E/O|$: absolute log$_2$ ratio of predicted to observed landslide cells in the unmapped area (median over events of per-event medians; 1 = a factor of two). "
     "Coverage: share of draws in which the 90\\% predictive interval contained the observed total (mean over events; nominal 0.90). Width: ratio of the upper to the lower interval bound (median).}\\label{tab:fewshot}\n"
     "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lrrrrrrrrrr}\\toprule\n & \\multicolumn{4}{c}{$|\\log_2 E/O|$} & \\multicolumn{4}{c}{90\\% coverage} & \\multicolumn{2}{c}{Width}\\\\\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-11}\n"
     "Method & $k$=1 & 5 & 10 & 40 & 1 & 5 & 10 & 40 & $k$=1 & 10\\\\\\midrule\n" + "\n".join(L) + "\n\\bottomrule\\end{tabular}}\\end{table*}\n")
open(os.path.join(T, "tab_fewshot.tex"), "w").write(s)
print(A[A.method.str.contains("quasi|PC-GBM \\+ int")].round(3).to_string())

# Supplementary Table S3: per-event results of full HiPhyBoost updating
import sys; sys.path.insert(0, os.path.dirname(__file__))
from events import EVENTS
e7 = pd.read_csv(os.path.join(R, "e7_fewshot_quasi.csv")); e7 = e7[e7.method.str.contains("full")].copy()
e7["ale"] = np.abs(np.log2(e7.e_over_o))
pe = e7.groupby(["event", "k"]).agg(ale=("ale", "median"), cov=("covered", "mean")).reset_index()
rows = []
for ev in EVENTS:
    d = pe[pe.event == ev].set_index("k")
    lab = EVENTS[ev]["label"].replace("ō", "\\={o}").replace("é", "\\'{e}")
    cells = [("%.2f" % d.loc[k, "ale"]) if k in d.index else "--" for k in [1, 5, 10, 40]] + \
            [("%.2f" % d.loc[k, "cov"]) if k in d.index else "--" for k in [1, 5, 10, 40]]
    rows.append(lab + " & " + " & ".join(cells) + "\\\\")
s3 = ("\\begin{table}[ht]\\centering\\small\n\\caption{Per-event results of full HiPhyBoost updating (quasi-posterior with selection term): median $|\\log_2 E/O|$ over 30 tile draws and coverage of the 90\\% interval of the landslide total in the unmapped area. Niigata-Chuetsu has too few tiles for $k=40$.}\\label{tab:perevent}\n"
      "\\begin{tabular}{lrrrrrrrr}\\toprule\n & \\multicolumn{4}{c}{$|\\log_2 E/O|$} & \\multicolumn{4}{c}{90\\% coverage}\\\\\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
      "Event & $k$=1 & 5 & 10 & 40 & $k$=1 & 5 & 10 & 40\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
open(os.path.join(T, "tab_perevent.tex"), "w").write(s3)
