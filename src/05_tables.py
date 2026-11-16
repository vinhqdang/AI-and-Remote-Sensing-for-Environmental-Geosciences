"""Write LaTeX tables from processed data and experiment results."""
import os, sys, json, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from events import EVENTS
ROOT = os.path.join(os.path.dirname(__file__), ".."); P = os.path.join(ROOT, "data", "processed")
R = os.path.join(ROOT, "results"); T = os.path.join(ROOT, "manuscript", "tables"); os.makedirs(T, exist_ok=True)
SETTING = {"kaikoura2016": "Multi-fault oblique-reverse/strike-slip, upper plate", "northridge1994": "Blind thrust, Transverse Ranges",
           "niigata2004": "Reverse, fold-and-thrust belt", "iwate2008": "Reverse, volcanic arc", "aisen2007": "Strike-slip swarm (Liquiñe-Ofqui)",
           "denali2002": "Strike-slip, Alaska Range", "cinchona2009": "Strike-slip/oblique, volcanic arc", "porgera2018": "Thrust, Papuan fold belt",
           "palu2018": "Supershear strike-slip"}
ORDER = list(EVENTS)

def t1():
    rows = []
    for k in ORDER:
        i = json.load(open(os.path.join(P, f"{k}_info.json")))
        ev = EVENTS[k]
        rows.append(f"{ev['label'].replace('ō', '\\={o}').replace('é', "\\'{e}")} & {ev['mw']:.1f} & {i['area_km2']:,.0f} & {i['n_points_in_domain']:,} & "
                    f"{i['n_pos_cells']:,} & {100*i['prevalence']:.3f} & {i['pga_max']:.2f} & {i['pga_mean_pos']:.2f} \\\\")
    s = ("\\begin{table}[t]\\centering\\small\n\\caption{The nine Pacific Rim earthquakes, their study domains and landslide statistics on the 30~m grid. "
         "Prevalence is the percentage of domain cells that contain a landslide point. PGA values are from the USGS ShakeMap used as model input.}\\label{tab:events}\n"
         "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lrrrrrrr}\\toprule\nEvent & M\\textsubscript{w} & Domain (km$^2$) & Landslides & Positive cells & Prevalence (\\%) & PGA$_{\\max}$ (g) & mean PGA at landslides (g)\\\\\\midrule\n"
         + "\n".join(rows).replace(",", "\\,") + "\n\\bottomrule\\end{tabular}}\\end{table}\n")
    open(os.path.join(T, "tab_events.tex"), "w").write(s)

def lab(k):
    return EVENTS[k]["label"].replace("ō", "\\={o}").replace("é", "\\'{e}")


MODELS = ["Physics (Jibson 2000)", "Physics-logit", "Global LR", "GBM", "PC-GBM"]


def t2():
    """Main-text summary: per-event AUC and E/O for each model."""
    r = pd.read_csv(os.path.join(R, "e1_loeo_metrics.csv"))
    tr = r[r.setting == "LOEO transfer"]; loc = r[r.setting == "Local spatial CV"].set_index("event")
    hdr = " & ".join(["Physics", "Phys.-logit", "Global LR", "GBM", "PC-GBM"])
    lines = []
    for k in ORDER:
        d = tr[tr.event == k].set_index("model")
        aucs = [d.loc[m, "auc"] for m in MODELS]; best = max(aucs)
        cells = [("\\textbf{%.3f}" % a) if a == best else "%.3f" % a for a in aucs]
        lines.append(f"{lab(k)} & " + " & ".join(cells) + f" & {loc.loc[k, 'auc']:.3f} & {d.loc['PC-GBM', 'e_over_o']:.2f} & {d.loc['PC-GBM', 'cal_slope']:.2f}\\\\")
    med = [tr[tr.model == m].auc.median() for m in MODELS]
    mn = [tr[tr.model == m].auc.min() for m in MODELS]
    lines.append("\\midrule Median & " + " & ".join("%.3f" % x for x in med) + f" & {loc.auc.median():.3f} & {tr[tr.model=='PC-GBM'].e_over_o.median():.2f} & {tr[tr.model=='PC-GBM'].cal_slope.median():.2f}\\\\")
    lines.append("Minimum & " + " & ".join("%.3f" % x for x in mn) + f" & {loc.auc.min():.3f} & & \\\\")
    s = ("\\begin{table}[t]\\centering\\small\n\\caption{Leave-one-event-out transfer. ROC AUC on each held-out event for the five models trained on the other eight events "
         "(best transferred model in bold), the AUC of a local PC-GBM evaluated by 5~km spatial-block cross-validation within the event, and calibration of the transferred PC-GBM: "
         "the expected-to-observed ratio of landslide cells ($E/O$, ideal 1) and the calibration slope (ideal 1).}\\label{tab:loeo}\n"
         "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lrrrrrrrr}\\toprule\n & \\multicolumn{5}{c}{Transferred (LOEO) AUC} & Local AUC & \\multicolumn{2}{c}{PC-GBM calibration}\\\\\n"
         "\\cmidrule(lr){2-6}\\cmidrule(lr){8-9}\nEvent & " + hdr + " & PC-GBM & $E/O$ & slope\\\\\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\\end{tabular}}\\end{table}\n")
    open(os.path.join(T, "tab_loeo.tex"), "w").write(s)


def t_full():
    """Supplementary: all metrics."""
    r = pd.read_csv(os.path.join(R, "e1_loeo_metrics.csv"))
    lines = []
    for k in ORDER:
        for (st, m), d in r[r.event == k].groupby(["setting", "model"], sort=False):
            d = d.iloc[0]
            nm = m if st == "LOEO transfer" else "PC-GBM (local CV)"
            lines.append(f"{lab(k)} & {nm} & {d.auc:.3f} & {d.ap_lift:.2f} & {d.bss:.3f} & {d.e_over_o:.3g} & {d.cal_slope:.2f} & {d.cal_int:.2f}\\\\")
        lines.append("\\midrule")
    s = ("\\begin{longtable}{llrrrrrr}\n\\caption{All performance metrics of Experiment~1. AP lift: average precision divided by prevalence; BSS: Brier skill score relative to a forecast that knows the true prevalence; "
         "$E/O$: expected over observed landslide cells; slope and intercept: logistic calibration of outcomes on predicted log-odds.}\\label{tab:full}\\\\\n\\toprule Event & Model & AUC & AP lift & BSS & $E/O$ & Cal. slope & Cal. int.\\\\\\midrule\\endhead\n"
         + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{longtable}\n")
    open(os.path.join(T, "tab_full.tex"), "w").write(s)


if __name__ == "__main__":
    t1()
    if os.path.exists(os.path.join(R, "e1_loeo_metrics.csv")):
        t2(); t_full()
