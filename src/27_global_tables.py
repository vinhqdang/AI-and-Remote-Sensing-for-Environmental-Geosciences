"""Supplementary table of the 38-event global set."""
import os, json, glob, sys
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from events import EVENTS
ROOT = os.path.join(os.path.dirname(__file__), ".."); T = os.path.join(ROOT, "manuscript", "tables")
cov = pd.read_csv(os.path.join(ROOT, "data", "global", "event_covariates.csv")).set_index("event")
a = pd.read_csv(os.path.join(ROOT, "results", "global", "global_auc.csv")).set_index("event")
off = pd.read_csv(os.path.join(ROOT, "results", "global_loeo", "ALL_offsets.csv")).set_index("train_event").b
sel = pd.read_csv(os.path.join(ROOT, "data", "global", "selected_events.csv")).set_index("key")
rows = []
for e in cov.sort_values("year").index:
    if e in EVENTS:
        info = json.load(open(os.path.join(ROOT, "data", "processed", f"{e}_info.json"))); lab = EVENTS[e]["label"]
    else:
        info = json.load(open(os.path.join(ROOT, "data", "processed_global", f"{e}_info.json"))); lab = sel.loc[e, "label"].split(" (")[0]
    lab = lab.replace("ō", "\\={o}").replace("é", "\\'{e}").replace("Milin Earthquake", "Milin")
    c = cov.loc[e]
    src = "GNS" if e == "kaikoura2016" else (sel.loc[e, "sb_id"] if e in sel.index else {k: v for k, v in [("northridge1994", "58595917e4b03639a6025f30"), ("niigata2004", "5874bd79e4b0a829a320bbbe"), ("iwate2008", "5b59ec98e4b0610d7f4da850"), ("aisen2007", "586fe5a7e4b01a71ba0c5f06"), ("denali2002", "5874a336e4b0a829a320bb15"), ("cinchona2009", "61ead71bd34e8b818ad9f44b"), ("porgera2018", "61f04680d34e8b818adc32a8"), ("palu2018", "61f196a1d34e8b818add4fdf")]}[e])
    rows.append(f"{lab} & {c.mag:.1f} & {info['n_pos_cells']:,} & {info['area_km2']:,.0f} & {'P' if c.point_inventory else 'A'} & {int(c.shakemap_stations) if pd.notna(c.shakemap_stations) else '--'} & {c.ppt_ratio:.2f} & {a.loc[e,'auc_pcgbm']:.3f} & {a.loc[e,'auc_hiphy']:.3f} & {off[e]:+.2f}".replace(",", "\\,") + f" & \\texttt{{\\scriptsize {src}}}\\\\")
s = ("\\begin{longtable}{lrrrcrrrrrl}\n\\caption{The 38 earthquakes of the global extension (ordered by year). Landslide cells and domain area on the 30~m grid; inventory type A = polygons (areas), P = points; ShakeMap stations; antecedent precipitation relative to its 1981--2010 climatology; leave-one-event-out AUC of PC-GBM and HiPhyBoost; HiPhyBoost event offset from the fit to all 38 events; source: ScienceBase item identifier of the inventory in the USGS repository (Schmitt et al. 2017; https://www.sciencebase.gov/catalog/item/<identifier>) or GNS Science (Kaik\\={o}ura).}\\label{tab:global}\\\\\n"
     "\\toprule Event & M & Landslide cells & km$^2$ & Inv. & Stations & Rain ratio & AUC PC-GBM & AUC HiPhy & $b_e$ & Source\\\\\\midrule\\endhead\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{longtable}\n")
open(os.path.join(T, "tab_global.tex"), "w").write(s)

# Supplementary Table S5: zero-shot priors for the offset (from 24_offset_regression.py outputs)
import json
R = os.path.join(ROOT, "results", "global")
s = pd.read_csv(os.path.join(R, "offset_regression_summary.csv")).set_index("model"); pl = json.load(open(os.path.join(R, "offset_regression_placebo.json")))
covd = pd.read_csv(os.path.join(R, "offset_regression.csv")).groupby("model").covered.sum()
order = ["M0 unconditional", "F1 shaking (forecast-time)", "F2 climate (forecast-time)", "F3 all forecast-time", "P1 inventory (post-event)", "P2 inventory + domain (post-event)", "P3 all"]
lab = {"M0 unconditional": "None (unconditional prior)", "F1 shaking (forecast-time)": "F1: Mw, depth, mechanism, ShakeMap stations", "F2 climate (forecast-time)": "F2: PDSI, rain ratio, mean annual rain",
       "F3 all forecast-time": "F3: F1 + F2", "P1 inventory (post-event)": "P1: inventory type, mapping extent, year", "P2 inventory + domain (post-event)": "P2: P1 + domain area", "P3 all": "P3: F3 + P2"}
L = []
for m in order:
    r = s.loc[m]; p = pl.get(m, {}); m0 = m.startswith("M0")
    L.append(f"{lab[m]} & {'' if m0 else ('forecast' if m.startswith('F') else 'post-event')} & {r.lpd_mean:.2f} & {'--' if m0 else f'{r.lpd_gain_vs_M0:+.2f}'} & {'--' if m0 else f'{r.p_lpd_gain:.3f}'} & {'--' if m0 else f'{p[chr(112)+chr(95)+chr(112)+chr(101)+chr(114)+chr(109)]:.3f}'} & {int(covd[m])}/38 & {r.width_median:.0f} & {2 ** r.abs_log2_eo_median:.1f}\\\\")
t = ("\\begin{table}[ht]\\centering\\small\n\\caption{Zero-shot priors for the event offset in the 38-event leave-one-event-out analysis. LPD: mean log predictive density of the held-out offset (nats); gain relative to the unconditional prior with paired Wilcoxon $p$ and permutation $p$ (200 random permutations of the covariate vectors among events); coverage of the 90\\% interval for the landslide total; median interval width (ratio of bounds) and median error factor of the median forecast.}\\label{tab:offreg}\n"
     "\\resizebox{\\textwidth}{!}{\\begin{tabular}{llrrrrrrr}\\toprule\nCovariates & Available & LPD & Gain & $p$ & $p_{\\mathrm{perm}}$ & Coverage & Width & Error\\\\\\midrule\n" + "\n".join(L) + "\n\\bottomrule\\end{tabular}}\\end{table}\n")
open(os.path.join(T, "tab_offreg.tex"), "w").write(t)
