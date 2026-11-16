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
    rows.append(f"{lab} & {c.mag:.1f} & {info['n_pos_cells']:,} & {info['area_km2']:,.0f} & {'P' if c.point_inventory else 'A'} & {int(c.shakemap_stations) if pd.notna(c.shakemap_stations) else '--'} & {c.ppt_ratio:.2f} & {a.loc[e,'auc_pcgbm']:.3f} & {a.loc[e,'auc_hiphy']:.3f} & {off[e]:+.2f}\\\\".replace(",", "\\,"))
s = ("\\begin{longtable}{lrrrcrrrrr}\n\\caption{The 38 earthquakes of the global extension (ordered by year). Landslide cells and domain area on the 30~m grid; inventory type A = polygons (areas), P = points; ShakeMap stations; antecedent precipitation relative to its 1981--2010 climatology; leave-one-event-out AUC of PC-GBM and HiPhyBoost; HiPhyBoost event offset from the fit to all 38 events.}\\label{tab:global}\\\\\n"
     "\\toprule Event & M & Landslide cells & km$^2$ & Inv. & Stations & Rain ratio & AUC PC-GBM & AUC HiPhy & $b_e$\\\\\\midrule\\endhead\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{longtable}\n")
open(os.path.join(T, "tab_global.tex"), "w").write(s)
