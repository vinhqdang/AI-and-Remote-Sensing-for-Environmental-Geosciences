"""Build 30 m samples for the global event set (same features as 03_build_features.py).

Events come from data/global/catalogue.csv (USGS ScienceBase repository). One inventory per event
is used (the one with most features; the nine original events keep their original inventory and
are not rebuilt). Large domains are processed in 90 km blocks with a 1.2 km halo so that window
statistics are identical to whole-domain processing. Negatives are sampled at a fixed rate per
event (target 150 000), with inverse-probability weights.

Usage: python 21_build_global.py list            -> writes data/global/selected_events.csv
       python 21_build_global.py build <key...>  -> builds data/processed_global/<key>.parquet
"""
import os, sys, json, importlib, time
import numpy as np, pandas as pd, geopandas as gpd, requests
from shapely.geometry import MultiPoint
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
sys.path.insert(0, os.path.dirname(__file__))
B = importlib.import_module("03_build_features")
from events import EVENTS

ROOT = os.path.join(os.path.dirname(__file__), "..")
CAT = os.path.join(ROOT, "data", "global", "catalogue.csv")
RAW = os.path.join(ROOT, "data", "raw_global"); os.makedirs(RAW, exist_ok=True)
OUT = os.path.join(ROOT, "data", "processed_global"); os.makedirs(OUT, exist_ok=True)
RES = 30.0; BLOCK = 3000; HALO = 40; NEG_TARGET = 150_000; MAX_AREA_KM2 = 80_000
ORIG = {v["comcat"]: k for k, v in EVENTS.items()}
ORIG["us1000778i"] = "kaikoura2016"


def slug(label, date):
    s = "".join(ch for ch in str(label).lower() if ch.isalnum())[:18]
    return f"{s}{str(date)[:4]}" if not s.endswith(str(date)[:4]) else s


def select():
    c = pd.read_csv(CAT)
    c = c[(c.has_landslide_layer == True) & (c.n_features >= 100) & (c.has_shakemap == True) & c.comcat_id.notna()]
    rows = []
    for cid, g in c.groupby("comcat_id"):
        if cid in ORIG:
            continue
        g = g.sort_values(["n_features"], ascending=False)
        r = g.iloc[0]
        rows.append(dict(key=slug(r.event_label.split(" (")[0], r.event_date), comcat_id=cid, sb_id=r.sb_id, layer=r.main_layer,
                         label=r.event_label, mag=r.mag, n_features=int(r.n_features), geometry_type=r.geometry_type,
                         has_mapping_extent=bool(r.has_mapping_extent), pacific_rim=r.pacific_rim))
    s = pd.DataFrame(rows).sort_values("n_features")
    s.to_csv(os.path.join(ROOT, "data", "global", "selected_events.csv"), index=False)
    print(s[["key", "label", "n_features", "geometry_type"]].to_string())
    return s


def get(url, stream=False):
    for k in range(5):
        try:
            r = requests.get(url, timeout=300, stream=stream); r.raise_for_status(); return r
        except Exception as e:
            print("  retry", k, e); time.sleep(3 * (k + 1))
    raise RuntimeError(url)


def download(ev):
    d = os.path.join(RAW, ev.key); os.makedirs(d, exist_ok=True)
    item = get(f"https://www.sciencebase.gov/catalog/item/{ev.sb_id}?format=json").json()
    for facet in item.get("facets", []):
        if facet.get("name") not in (ev.layer, "mapping_extent"):
            continue
        for f in facet.get("files", []):
            p = os.path.join(d, f["name"])
            if not os.path.exists(p):
                with open(p, "wb") as fh:
                    for chunk in get(f["url"], stream=True).iter_content(1 << 20):
                        fh.write(chunk)
    sm = os.path.join(d, "grid.xml")
    if not os.path.exists(sm):
        dd = get(f"https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={ev.comcat_id}&format=geojson").json()
        prods = sorted(dd["properties"]["products"]["shakemap"],
                       key=lambda s: (s["source"] == "atlas", -int(s["properties"].get("version", 0) or 0)))
        open(sm, "wb").write(get(prods[0]["contents"]["download/grid.xml"]["url"]).content)
    return d


def read_shakemap(path):
    import re
    from scipy.interpolate import RegularGridInterpolator
    txt = open(path).read()
    fields = re.findall(r'<grid_field index="(\d+)" name="(\w+)"', txt)
    names = [n for _, n in sorted(fields, key=lambda x: int(x[0]))]
    body = txt.split("<grid_data>")[1].split("</grid_data>")[0]
    arr = np.array(body.split(), dtype=np.float64).reshape(-1, len(names))
    df = pd.DataFrame(arr, columns=names)
    lons = np.sort(df.LON.unique()); lats = np.sort(df.LAT.unique())
    out = {}
    for f in ["PGA", "PGV"]:
        p = df.pivot_table(index="LAT", columns="LON", values=f).reindex(index=lats, columns=lons).values
        out[f] = RegularGridInterpolator((lats, lons), p, bounds_error=False, fill_value=np.nan)
    st = re.search(r'seismic_stations="(\d+)"', txt)
    return out, (int(st.group(1)) if st else -1)


def build(ev):
    d = download(ev)
    g = gpd.read_file(os.path.join(d, ev.layer + ".shp"))
    g = (g.set_crs(4326) if g.crs is None else g).to_crs(4326)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    g["geometry"] = g.geometry.make_valid()
    pts = g.geometry if (g.geom_type == "Point").all() else g.geometry.representative_point()
    pts = gpd.GeoSeries(pts, crs=4326)
    c = pts.union_all().centroid
    crs = B.utm_crs(c.x, c.y)
    ext = os.path.join(d, "mapping_extent.shp")
    if os.path.exists(ext):
        e = gpd.read_file(ext); e = (e.set_crs(4326) if e.crs is None else e)
        dom = e.to_crs(crs).geometry.make_valid().union_all()
    else:
        dom = MultiPoint(g.to_crs(crs).get_coordinates().values).convex_hull.buffer(2000)
    area = dom.area / 1e6
    if area > MAX_AREA_KM2:
        print(f"  {ev.key}: domain {area:.0f} km2 > {MAX_AREA_KM2}, skipped"); return None
    sm, nst = read_shakemap(os.path.join(d, "grid.xml"))
    pts_u = pts.to_crs(crs)
    minx, miny, maxx, maxy = dom.bounds
    minx, miny = np.floor(minx / RES) * RES - 600, np.floor(miny / RES) * RES - 600
    maxx, maxy = np.ceil(maxx / RES) * RES + 600, np.ceil(maxy / RES) * RES + 600
    W, H = int((maxx - minx) / RES), int((maxy - miny) / RES)
    rate = min(1.0, NEG_TARGET / max(area * 1e6 / RES ** 2, 1))
    import zlib; rng = np.random.default_rng(zlib.crc32(ev.key.encode()))
    from pyproj import Transformer
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    parts, n_valid, n_pos = [], 0, 0
    for r0 in range(0, H, BLOCK):
        for c0 in range(0, W, BLOCK):
            r1, c1 = min(H, r0 + BLOCK), min(W, c0 + BLOCK)
            R0, C0, R1, C1 = max(0, r0 - HALO), max(0, c0 - HALO), min(H, r1 + HALO), min(W, c1 + HALO)
            h, w = R1 - R0, C1 - C0
            bx0, by1 = minx + C0 * RES, maxy - R0 * RES
            T = from_origin(bx0, by1, RES, RES)
            inside = rasterize([(dom, 1)], out_shape=(h, w), transform=T, fill=0, dtype="uint8").astype(bool)
            core = np.zeros((h, w), bool); core[r0 - R0:r1 - R0, c0 - C0:c1 - C0] = True
            if not (inside & core).any():
                continue
            from shapely.geometry import box
            b4326 = gpd.GeoSeries([box(bx0, by1 - h * RES, bx0 + w * RES, by1).buffer(1000)], crs=crs).to_crs(4326).total_bounds
            dem = B.load_dem(b4326, crs, T, (h, w))
            valid = inside & core & ~np.isnan(dem) & (dem > 0.5)
            if not valid.any():
                continue
            feats = B.terrain(dem)
            # ground motion
            xv = bx0 + (np.arange(w) + 0.5) * RES
            pga = np.empty((h, w), np.float32); pgv = np.empty((h, w), np.float32)
            for a0 in range(0, h, 500):
                a1 = min(h, a0 + 500); yv = by1 - (np.arange(a0, a1) + 0.5) * RES
                XX, YY = np.meshgrid(xv, yv); lo, la = tr.transform(XX.ravel(), YY.ravel()); q = np.column_stack([la, lo])
                pga[a0:a1] = sm["PGA"](q).reshape(a1 - a0, w) / 100.0; pgv[a0:a1] = sm["PGV"](q).reshape(a1 - a0, w)
            feats["pga"], feats["pgv"] = pga, pgv
            fs, ac, dn = B.newmark(feats["slope"], pga)
            feats["ac"], feats["newmark_cm"] = ac, dn
            lab = np.zeros((h, w), bool)
            rr, cc = rasterio.transform.rowcol(T, pts_u.x.values, pts_u.y.values)
            rr, cc = np.asarray(rr), np.asarray(cc); ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            lab[rr[ok], cc[ok]] = True
            y = lab & valid
            n_valid += int(valid.sum()); n_pos += int(y.sum())
            neg = np.flatnonzero(valid.ravel() & ~y.ravel()); neg = neg[rng.random(len(neg)) < rate]
            pos = np.flatnonzero(y.ravel()); idx = np.concatenate([pos, neg])
            df = pd.DataFrame({f: v.ravel()[idx] for f, v in feats.items() if f in
                               ["slope", "slope_max5", "relief510", "tpi510", "relpos1k", "curv_prof", "curv_plan", "slope_sd3", "pga", "pgv", "ac", "newmark_cm"]})
            df["y"] = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.int8)
            df["w"] = np.r_[np.ones(len(pos)), np.full(len(neg), 1 / rate)].astype(np.float32)
            rr2, cc2 = np.unravel_index(idx, (h, w))
            df["x"] = (bx0 + (cc2 + 0.5) * RES).astype(np.float32); df["y_utm"] = (by1 - (rr2 + 0.5) * RES).astype(np.float32)
            parts.append(df)
            print(f"  {ev.key} block r{r0} c{c0}: valid {valid.sum()} pos {y.sum()}", flush=True)
    df = pd.concat(parts, ignore_index=True).dropna(subset=["slope", "pga", "pgv"])
    df["event"] = ev.key
    df.to_parquet(os.path.join(OUT, f"{ev.key}.parquet"))
    info = dict(event=ev.key, comcat_id=ev.comcat_id, label=ev.label, crs=crs, area_km2=n_valid * RES * RES / 1e6, n_valid_cells=n_valid,
                n_pos_cells=n_pos, n_landslides_inventory=int(len(pts)), prevalence=n_pos / max(n_valid, 1), neg_rate=rate,
                shakemap_n_stations=nst, pga_max=float(np.nanmax(df.pga)), geometry_type=ev.geometry_type)
    json.dump(info, open(os.path.join(OUT, f"{ev.key}_info.json"), "w"), indent=1)
    print("  done", info, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "list":
        select()
    else:
        s = pd.read_csv(os.path.join(ROOT, "data", "global", "selected_events.csv")).set_index("key")
        for k in sys.argv[2:]:
            if os.path.exists(os.path.join(OUT, f"{k}.parquet")):
                continue
            try:
                ev = s.loc[k]; ev.key = k
                build(ev)
            except Exception as ex:
                print("FAILED", k, repr(ex), flush=True)
