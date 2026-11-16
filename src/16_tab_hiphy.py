"""LaTeX table of HiPhyBoost LOEO transfer (Table tab_hiphy)."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from events import EVENTS
R = os.path.join(os.path.dirname(__file__), "..", "results"); T = os.path.join(os.path.dirname(__file__), "..", "manuscript", "tables")
e = pd.read_csv(os.path.join(R, "e4_hiphy_loeo.csv")); e1 = pd.read_csv(os.path.join(R, "e1_loeo_metrics.csv"))
bd = pd.read_csv(os.path.join(R, "e4_auc_diff_hiphy_vs_pcgbm.csv")).set_index("event")
lab = lambda k: EVENTS[k]["label"].replace("ō", "\\={o}").replace("é", "\\'{e}")
L = []
for k in EVENTS:
    h = e[(e.event == k) & (e.method == "HiPhyBoost (plug-in)")].iloc[0]; a = e[(e.event == k) & (e.method.str.startswith("Ablation"))].iloc[0]
    z = e[(e.event == k) & (e.method == "HiPhyBoost (zero-shot Bayes)")].iloc[0]
    pc = e1[(e1.event == k) & (e1.model == "PC-GBM") & (e1.setting == "LOEO transfer")].iloc[0]
    aucs = [pc.auc, a.auc, h.auc]; best = max(aucs)
    cell = ["\\textbf{%.3f}" % v if v == best else "%.3f" % v for v in aucs]
    b = bd.loc[k]; cell[2] += "$^{*}$" if (b.lo > 0 or b.hi < 0) else ""
    L.append(f"{lab(k)} & " + " & ".join(cell) + f" & {pc.ap_lift:.1f} & {h.ap_lift:.1f} & {pc.e_over_o:.2f} & {h.e_over_o:.2f} & [{z.eo_q05:.3f}, {z.eo_q95:.1f}] & {'yes' if z.covered else 'no'}\\\\")
hh = e[e.method == "HiPhyBoost (plug-in)"]; aa = e[e.method.str.startswith("Ablation")]; pp = e1[(e1.model == "PC-GBM") & (e1.setting == "LOEO transfer")]
L.append(f"\\midrule Median & {pp.auc.median():.3f} & {aa.auc.median():.3f} & {hh.auc.median():.3f} & {pp.ap_lift.median():.1f} & {hh.ap_lift.median():.1f} & {pp.e_over_o.median():.2f} & {hh.e_over_o.median():.2f} & & 7/9\\\\")
s = ("\\begin{table*}[t]\\centering\\small\n\\caption{Leave-one-event-out transfer of HiPhyBoost compared with PC-GBM. AUC of the transferred PC-GBM, of HiPhyBoost without event offsets (ablation) and of the full HiPhyBoost (best of the three in bold; $^{*}$: 95\\% spatial block-bootstrap interval of the difference from PC-GBM excludes zero); AP lift; ratio of predicted to observed landslide cells of the point forecasts ($E/O$, ideal 1); HiPhyBoost 90\\% zero-shot predictive interval of $E/O$ and whether it contains 1. Across the nine events the AUC difference from PC-GBM is not significant (Wilcoxon signed-rank $p=0.16$).}\\label{tab:hiphy}\n"
     "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lrrrrrrrlc}\\toprule\n & \\multicolumn{3}{c}{AUC} & \\multicolumn{2}{c}{AP lift} & \\multicolumn{2}{c}{$E/O$} & \\multicolumn{2}{c}{Zero-shot 90\\% interval}\\\\\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n"
     "Event & PC-GBM & No offsets & HiPhyBoost & PC-GBM & HiPhyBoost & PC-GBM & HiPhyBoost & $E/O$ interval & covers\\\\\\midrule\n" + "\n".join(L) + "\n\\bottomrule\\end{tabular}}\\end{table*}\n")
open(os.path.join(T, "tab_hiphy.tex"), "w").write(s)
