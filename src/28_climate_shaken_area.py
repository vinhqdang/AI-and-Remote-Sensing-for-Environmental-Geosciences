"""Antecedent-wetness covariates averaged over the strongly shaken area (ShakeMap PGA >= 0.2 g;
0.1 g if fewer than 20 grid nodes exceed 0.2 g) instead of the area of the mapped landslides, so
that the climate covariates are available at forecast time and do not use outcome information.
TerraClimate is sampled at up to 300 randomly chosen shaken ShakeMap nodes (nearest 4 km cell);
ocean cells are ignored.
Usage: python 28_climate_shaken_area.py <part> <event ...>  -> data/global/climsh_part_<part>.csv
       python 28_climate_shaken_area.py merge             -> adds *_sh columns to event_covariates.csv
"""
import os, sys, re, glob, importlib
import numpy as np, pandas as pd, xarray as xr
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.join(os.path.dirname(__file__), "..")
G = os.path.join(ROOT, "data", "global")


def shakemap_path(e):
    p = os.path.join(ROOT, "data", "raw", "shakemaps", f"{e}_grid.xml")
    return p if os.path.exists(p) else os.path.join(ROOT, "data", "raw_global", e, "grid.xml")


def shaken_nodes(e, n=300, seed=0):
    txt = open(shakemap_path(e)).read()
    names = [nm for _, nm in sorted(re.findall(r'<grid_field index="(\d+)" name="(\w+)"', txt), key=lambda x: int(x[0]))]
    arr = np.array(txt.split("<grid_data>")[1].split("</grid_data>")[0].split(), dtype=float).reshape(-1, len(names))
    df = pd.DataFrame(arr, columns=names)
    for thr in (20.0, 10.0, 5.0):                       # %g
        s = df[df.PGA >= thr]
        if len(s) >= 20:
            break
    s = s.sample(min(n, len(s)), random_state=seed)
    return s.LAT.values, s.LON.values, thr / 100, len(df[df.PGA >= thr])


def main(part, events):
    m = importlib.import_module("22_event_covariates")
    tc = m.open_tc()
    cov = pd.read_csv(os.path.join(G, "event_covariates.csv")).set_index("event")
    out = []
    for e in events:
        la, lo, thr, nshaken = shaken_nodes(e)
        P = tc[["ppt", "soil", "pdsi"]].sel(lat=xr.DataArray(la, dims="pt"), lon=xr.DataArray(lo, dims="pt"), method="nearest")
        t = pd.Timestamp(cov.loc[e, "date"]); m0 = pd.Timestamp(t.year, t.month, 1); mp = m0 - pd.DateOffset(months=1)
        frac = (t.day - 1) / t.days_in_month
        def val(var, ts): return float(np.nanmean(P[var].sel(time=ts).values))
        yrs = range(1981, 2011)
        cmp_ = P[["ppt", "soil"]].sel(time=[pd.Timestamp(y, mp.month, 1) for y in yrs]).load()
        cm0 = P["ppt"].sel(time=[pd.Timestamp(y, m0.month, 1) for y in yrs]).load()
        ppt_ante = val("ppt", mp) + frac * val("ppt", m0)
        ppt_clim = float(np.nanmean(cmp_.ppt.values)) + frac * float(np.nanmean(cm0.values))
        mapv = float(np.nanmean(P["ppt"].sel(time=slice("1991-01-01", "2010-12-01")).load().values)) * 12
        out.append(dict(event=e, shaken_threshold_g=thr, n_shaken_nodes=nshaken, ppt_ante_mm_sh=ppt_ante,
                        ppt_ratio_sh=ppt_ante / max(ppt_clim, 1.0), soil_anom_mm_sh=val("soil", mp) - float(np.nanmean(cmp_.soil.values)),
                        pdsi_prev_sh=val("pdsi", mp), map_mm_sh=mapv))
        print(out[-1], flush=True)
    pd.DataFrame(out).to_csv(os.path.join(G, f"climsh_part_{part}.csv"), index=False)


def merge():
    sh = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(G, "climsh_part_*.csv")))])
    cov = pd.read_csv(os.path.join(G, "event_covariates.csv"))
    cov = cov.drop(columns=[c for c in cov.columns if c.endswith("_sh") or c in ("shaken_threshold_g", "n_shaken_nodes")])
    cov = cov.merge(sh, on="event", how="left")
    cov.to_csv(os.path.join(G, "event_covariates.csv"), index=False)
    print(cov[["event", "ppt_ratio", "ppt_ratio_sh", "pdsi_prev", "pdsi_prev_sh", "map_mm", "map_mm_sh"]].round(2).to_string())
    print("corr landslide-area vs shaken-area:", {v: round(float(np.corrcoef(cov[v], cov[v + "_sh"])[0, 1]), 2) for v in ["ppt_ratio", "pdsi_prev", "map_mm"]})


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge()
    else:
        main(sys.argv[1], sys.argv[2:])
