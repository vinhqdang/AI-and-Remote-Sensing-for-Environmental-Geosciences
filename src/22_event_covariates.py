"""Event-level covariates for the global event set.

Climate (TerraClimate, monthly 4 km, Planetary Computer), averaged over the bounding box of the
landslide cells of each event:
  ppt_ante   precipitation of the previous month plus the elapsed fraction of the event month (mm)
  ppt_ratio  ppt_ante relative to its 1981-2010 climatology for the same calendar months
  soil_anom  soil moisture of the previous month minus its 1981-2010 mean for that month (mm)
  pdsi_prev  Palmer drought severity index of the previous month (positive = wet)
  map_mm     mean annual precipitation 1981-2010 (mm)
Earthquake / ShakeMap: Mw, depth, mechanism (reverse / strike-slip / normal), ShakeMap stations.
Inventory: point catalogue vs polygons, published mapping extent, log number of landslides,
  5th percentile polygon area (proxy for minimum mapped landslide size, i.e. mapping resolution),
  year (imagery era).
Output: data/global/event_covariates.csv
"""
import os, sys, json, glob, importlib
import numpy as np, pandas as pd, geopandas as gpd
from pyproj import Transformer
sys.path.insert(0, os.path.dirname(__file__))
from events import EVENTS
ROOT = os.path.join(os.path.dirname(__file__), "..")
P1 = os.path.join(ROOT, "data", "processed"); P2 = os.path.join(ROOT, "data", "processed_global")
CAT = pd.read_csv(os.path.join(ROOT, "data", "global", "catalogue.csv"))
SEL = pd.read_csv(os.path.join(ROOT, "data", "global", "selected_events.csv"))


def open_tc():
    import pystac_client, planetary_computer, xarray as xr
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
    a = cat.get_collection("terraclimate").assets["zarr-abfs"]
    kw = dict(a.extra_fields["xarray:open_kwargs"]); kw.pop("engine", None)
    return xr.open_zarr(a.href, **kw)


def events_table():
    rows = []
    for k, ev in EVENTS.items():
        info = json.load(open(os.path.join(P1, f"{k}_info.json")))
        c = CAT[CAT.comcat_id == ev["comcat"]]
        c = c.iloc[0] if len(c) else None
        rows.append(dict(event=k, comcat_id=ev["comcat"], parquet=os.path.join(P1, f"{k}.parquet"), crs=info["crs"],
                         inv_dir=os.path.join(ROOT, "data", "raw", "inventories", k), layer=ev["facet"],
                         geometry_type="Point" if k in ("kaikoura2016", "palu2018") else "Polygon",
                         has_mapping_extent=ev["extent"] is not None, n_landslides=info["n_landslides_inventory"],
                         stations=(c.shakemap_n_stations if c is not None else np.nan),
                         mechanism=(c.mechanism if c is not None else ("reverse" if k == "kaikoura2016" else np.nan)),
                         mag=ev["mw"], set="original"))
    for f in sorted(glob.glob(os.path.join(P2, "*_info.json"))):
        info = json.load(open(f)); k = info["event"]; s = SEL[SEL.key == k].iloc[0]
        c = CAT[CAT.sb_id == s.sb_id].iloc[0]
        rows.append(dict(event=k, comcat_id=s.comcat_id, parquet=os.path.join(P2, f"{k}.parquet"), crs=info["crs"],
                         inv_dir=os.path.join(ROOT, "data", "raw_global", k), layer=s.layer, geometry_type=s.geometry_type,
                         has_mapping_extent=bool(s.has_mapping_extent), n_landslides=info["n_landslides_inventory"],
                         stations=c.shakemap_n_stations, mechanism=c.mechanism, mag=s.mag, set="global"))
    return pd.DataFrame(rows)


def main(only=None, out_name="event_covariates.csv"):
    import requests
    T = events_table()
    if only:
        T = T[T.event.isin(only)]
    tc = open_tc()
    out = []
    for _, r in T.iterrows():
        d = pd.read_parquet(r.parquet, columns=["x", "y_utm", "y"])
        pos = d[d.y == 1]
        tr = Transformer.from_crs(r.crs, "EPSG:4326", always_xy=True)
        lo, la = tr.transform(pos.x.values, pos.y_utm.values)
        lo0, lo1, la0, la1 = np.percentile(lo, 2), np.percentile(lo, 98), np.percentile(la, 2), np.percentile(la, 98)
        pad = 0.05
        q = requests.get(f"https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={r.comcat_id}&format=geojson", timeout=60).json()
        t = pd.Timestamp(q["properties"]["time"], unit="ms"); depth = q["geometry"]["coordinates"][2]
        box = tc.sel(lat=slice(la1 + pad, la0 - pad), lon=slice(lo0 - pad, lo1 + pad))
        if box.sizes["lat"] == 0 or box.sizes["lon"] == 0:
            box = tc.sel(lat=(la0 + la1) / 2, lon=(lo0 + lo1) / 2, method="nearest").expand_dims(["lat", "lon"])
        m0 = pd.Timestamp(t.year, t.month, 1); mp = m0 - pd.DateOffset(months=1)
        frac = (t.day - 1 + t.hour / 24) / t.days_in_month
        if box.sizes["lat"] * box.sizes["lon"] > 400:          # thin large boxes for speed
            st = int(np.ceil(np.sqrt(box.sizes["lat"] * box.sizes["lon"] / 400)))
            box = box.isel(lat=slice(None, None, st), lon=slice(None, None, st))
        def val(var, ts): return float(box[var].sel(time=ts).mean().values)
        ppt_ante = val("ppt", mp) + frac * val("ppt", m0)
        yrs = range(1981, 2011)
        clim_mp = box[["ppt", "soil"]].sel(time=[pd.Timestamp(y, mp.month, 1) for y in yrs]).mean(["lat", "lon"]).load()
        clim_m0 = box["ppt"].sel(time=[pd.Timestamp(y, m0.month, 1) for y in yrs]).mean(["lat", "lon"]).load()
        ppt_clim = float(clim_mp.ppt.mean()) + frac * float(clim_m0.mean())
        soil_anom = val("soil", mp) - float(clim_mp.soil.mean())
        map_mm = float(box["ppt"].sel(time=slice("1991-01-01", "2010-12-01")).mean(["lat", "lon"]).load().sum()) / 20
        # inventory: minimum mapped size proxy
        p5 = np.nan
        if r.geometry_type == "Polygon":
            shp = os.path.join(r.inv_dir, r.layer + ".shp")
            if os.path.exists(shp):
                g = gpd.read_file(shp); g = (g.set_crs(4326) if g.crs is None else g).to_crs(r.crs)
                a = g.geometry.area.values; a = a[a > 0]
                if len(a): p5 = float(np.percentile(a, 5))
        out.append(dict(event=r.event, set=r.set, date=str(t.date()), year=t.year, mag=r.mag, depth_km=depth, mechanism=r.mechanism,
                        shakemap_stations=r.stations, point_inventory=r.geometry_type == "Point", mapping_extent=r.has_mapping_extent,
                        log_n_landslides=np.log10(r.n_landslides), poly_area_p5_m2=p5, ppt_ante_mm=ppt_ante,
                        ppt_ratio=ppt_ante / max(ppt_clim, 1.0), soil_anom_mm=soil_anom, pdsi_prev=val("pdsi", mp), map_mm=map_mm,
                        lon=(lo0 + lo1) / 2, lat=(la0 + la1) / 2))
        print(out[-1], flush=True)
    pd.DataFrame(out).to_csv(os.path.join(ROOT, "data", "global", out_name), index=False)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[2:], out_name=f"cov_part_{sys.argv[1]}.csv")
    else:
        main()
