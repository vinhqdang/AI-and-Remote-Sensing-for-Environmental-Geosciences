"""Build 30 m analysis grids and per-cell samples for each event.

For each event:
  1. Study domain = published mapping extent if available, otherwise the convex hull
     of the inventory buffered by 2 km (the area actually inspected by the mappers).
  2. Copernicus GLO-30 DEM tiles (AWS open data) are reprojected (bilinear) to a
     30 m UTM grid covering the domain.
  3. Terrain derivatives, ShakeMap ground motion (bilinear from grid.xml), and a
     Newmark rigid sliding-block displacement are computed.
  4. Labels: a cell is positive if it contains at least one landslide point. Polygons
     are reduced to a representative interior point so that all nine inventories
     share the same point-based sampling unit as the Kaikoura source-area centroids.
  5. All positive cells plus a uniform random sample of negative cells are stored with
     inverse-probability weights so that weighted statistics equal full-domain
     statistics.
"""
import os, sys, json, re
import numpy as np, pandas as pd, geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.features import rasterize
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, os.path.dirname(__file__))
from events import EVENTS

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed"); os.makedirs(OUT, exist_ok=True)
RES = 30.0
RNG = np.random.default_rng(42)
NEG_TARGET = 250_000           # negatives sampled per event (natural prevalence restored via weights)
G = 9.81

# Newmark parameters (uniform, deliberately generic; Jibson et al. 2000; Jibson 2007)
PHI = np.deg2rad(32.0)   # effective friction angle
COH = 15e3               # effective cohesion, Pa
GAMMA = 20e3             # unit weight, N/m^3
T = 3.0                  # slope-normal failure thickness, m


def load_inventory(key):
    d = os.path.join(RAW, "inventories", key)
    ev = EVENTS[key]
    if key == "kaikoura2016":
        g = gpd.read_file(os.path.join(d, "kaikoura_sources.gpkg"))
    else:
        g = gpd.read_file(os.path.join(d, ev["facet"] + ".shp"))
    if g.crs is None:
        g = g.set_crs(4326)
    g = g.to_crs(4326)
    if key == "porgera2018" and "coseismic" in g.columns:
        g = g[g["coseismic"].astype(str).str.strip().isin(["1", "1.0", "Y", "y", "yes", "True"])]
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    g["geometry"] = g.geometry.make_valid()
    pts = g.geometry.representative_point() if not (g.geom_type == "Point").all() else g.geometry
    ext = None
    if ev["extent"]:
        ext = gpd.read_file(os.path.join(d, ev["extent"] + ".shp"))
        ext = (ext.set_crs(4326) if ext.crs is None else ext).to_crs(4326)
    return g, gpd.GeoSeries(pts, crs=4326), ext


def utm_crs(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def cop_tile(lat, lon):
    ns = "N" if lat >= 0 else "S"; ew = "E" if lon >= 0 else "W"
    name = f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
    return f"/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


def load_dem(bounds4326, crs, transform, shape):
    dem = np.full(shape, np.nan, np.float32)
    lon0, lat0, lon1, lat1 = bounds4326
    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                       GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2")
    with env:
        for la in range(int(np.floor(lat0)), int(np.floor(lat1)) + 1):
            for lo in range(int(np.floor(lon0)), int(np.floor(lon1)) + 1):
                url = cop_tile(la, lo)
                try:
                    src = rasterio.open(url)
                except Exception:
                    continue          # ocean tile (not produced)
                with src:
                    tmp = np.full(shape, np.nan, np.float32)
                    reproject(rasterio.band(src, 1), tmp, dst_transform=transform, dst_crs=crs,
                              resampling=Resampling.bilinear, dst_nodata=np.nan)
                    m = np.isnan(dem) & ~np.isnan(tmp)
                    dem[m] = tmp[m]
                    print("   tile", la, lo)
    return dem


def read_shakemap(key):
    txt = open(os.path.join(RAW, "shakemaps", f"{key}_grid.xml")).read()
    fields = re.findall(r'<grid_field index="(\d+)" name="(\w+)"', txt)
    names = [n for _, n in sorted(fields, key=lambda x: int(x[0]))]
    spec = dict(re.findall(r'(\w+)="([^"]+)"', re.search(r"<grid_specification([^>]*)>", txt).group(1)))
    body = txt.split("<grid_data>")[1].split("</grid_data>")[0]
    arr = np.array(body.split(), dtype=np.float64).reshape(-1, len(names))
    nlon, nlat = int(spec["nlon"]), int(spec["nlat"])
    df = pd.DataFrame(arr, columns=names)
    lons = np.sort(df.LON.unique()); lats = np.sort(df.LAT.unique())
    assert len(lons) == nlon and len(lats) == nlat
    grids = {}
    for f in ["PGA", "PGV", "MMI", "SVEL"]:
        if f in df:
            p = df.pivot(index="LAT", columns="LON", values=f).loc[lats, lons].values
            grids[f] = RegularGridInterpolator((lats, lons), p, bounds_error=False, fill_value=np.nan)
    return grids


def terrain(dem):
    z = dem.astype(np.float64)
    zf = np.where(np.isnan(z), np.nanmean(z), z)
    dzdy, dzdx = np.gradient(zf, RES)  # rows go north->south; sign irrelevant for magnitudes
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    # Zevenbergen & Thorne (1987) second derivatives
    zxx = (np.roll(zf, -1, 1) - 2 * zf + np.roll(zf, 1, 1)) / RES**2
    zyy = (np.roll(zf, -1, 0) - 2 * zf + np.roll(zf, 1, 0)) / RES**2
    zxy = (np.roll(np.roll(zf, -1, 0), -1, 1) - np.roll(np.roll(zf, -1, 0), 1, 1)
           - np.roll(np.roll(zf, 1, 0), -1, 1) + np.roll(np.roll(zf, 1, 0), 1, 1)) / (4 * RES**2)
    p = dzdx**2 + dzdy**2 + 1e-12
    prof = -(zxx * dzdx**2 + 2 * zxy * dzdx * dzdy + zyy * dzdy**2) / p * 100
    plan = -(zxx * dzdy**2 - 2 * zxy * dzdx * dzdy + zyy * dzdx**2) / p * 100
    prof = np.clip(prof, -20, 20); plan = np.clip(plan, -20, 20)
    zmax17 = ndimage.maximum_filter(zf, 17); zmin17 = ndimage.minimum_filter(zf, 17)
    relief = zmax17 - zmin17                                         # ~510 m window
    tpi = zf - ndimage.uniform_filter(zf, 17)
    zmax33 = ndimage.maximum_filter(zf, 33); zmin33 = ndimage.minimum_filter(zf, 33)
    relpos = (zf - zmin33) / np.maximum(zmax33 - zmin33, 1.0)        # ~1 km window, 0 = valley, 1 = ridge
    slope_max = ndimage.maximum_filter(slope, 5)
    rough = np.sqrt(np.maximum(ndimage.uniform_filter(slope**2, 3) - ndimage.uniform_filter(slope, 3)**2, 0))
    out = dict(slope=slope, slope_max5=slope_max, relief510=relief, tpi510=tpi, relpos1k=relpos,
               curv_prof=prof, curv_plan=plan, slope_sd3=rough)
    nanm = np.isnan(z)
    for k in out:
        out[k] = np.where(nanm, np.nan, out[k]).astype(np.float32)
    return out


def newmark(slope_deg, pga_g):
    """Infinite-slope factor of safety and Jibson (2007) Eq. 6 displacement (cm)."""
    a = np.deg2rad(np.clip(slope_deg, 0.5, 89))
    fs = COH / (GAMMA * T * np.sin(a)) + np.tan(PHI) / np.tan(a)
    ac = np.clip((fs - 1.0) * np.sin(a), 0.0, None)       # in g
    ratio = np.where(pga_g > 0, ac / np.maximum(pga_g, 1e-6), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        logd = 0.215 + np.log10((1 - ratio) ** 2.341 * ratio ** -1.438)
    dn = np.where((ratio < 1) & (ratio > 0), 10 ** logd, np.where(ratio <= 0, 10 ** 2.5, 0.0))
    return fs.astype(np.float32), ac.astype(np.float32), dn.astype(np.float32)


def process(key):
    ev = EVENTS[key]
    print("==", key)
    inv, pts, ext = load_inventory(key)
    c = pts.union_all().centroid
    crs = utm_crs(c.x, c.y)
    pts_u = pts.to_crs(crs)
    if ext is not None:
        dom = ext.to_crs(crs).union_all()
    else:
        from shapely.geometry import MultiPoint
        dom = MultiPoint(inv.to_crs(crs).get_coordinates().values).convex_hull.buffer(2000)
    minx, miny, maxx, maxy = dom.bounds
    minx, miny = np.floor(minx / RES) * RES - 600, np.floor(miny / RES) * RES - 600
    maxx, maxy = np.ceil(maxx / RES) * RES + 600, np.ceil(maxy / RES) * RES + 600
    W, H = int((maxx - minx) / RES), int((maxy - miny) / RES)
    transform = from_origin(minx, maxy, RES, RES)
    b4326 = gpd.GeoSeries([dom.envelope.buffer(1000)], crs=crs).to_crs(4326).total_bounds
    print("  grid", H, W, crs, f"{H*W*RES*RES/1e6:.0f} km2 bbox")
    dem = load_dem(b4326, crs, transform, (H, W))
    inside = rasterize([(dom, 1)], out_shape=(H, W), transform=transform, fill=0, dtype="uint8").astype(bool)
    valid = inside & ~np.isnan(dem) & (dem > 0.5)  # exclude sea / no-data
    feats = terrain(dem)
    feats["elev"] = dem
    # ShakeMap
    sm = read_shakemap(key)
    from pyproj import Transformer
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xv = minx + (np.arange(W) + 0.5) * RES
    for f in ["PGA", "PGV", "MMI", "SVEL"]:
        if f in sm:
            feats[f.lower()] = np.empty((H, W), np.float32)
    for r0 in range(0, H, 500):
        r1 = min(H, r0 + 500)
        yv = maxy - (np.arange(r0, r1) + 0.5) * RES
        XX, YY = np.meshgrid(xv, yv)
        lo, la = tr.transform(XX.ravel(), YY.ravel())
        q = np.column_stack([la, lo])
        for f in ["PGA", "PGV", "MMI", "SVEL"]:
            if f in sm:
                feats[f.lower()][r0:r1] = sm[f](q).reshape(r1 - r0, W)
    feats["pga"] = feats["pga"] / 100.0          # %g -> g
    fs, ac, dn = newmark(feats["slope"], feats["pga"])
    feats["fs_static"], feats["ac"], feats["newmark_cm"] = fs, ac, dn
    # labels
    lab = np.zeros((H, W), np.uint16)
    r, cidx = rasterio.transform.rowcol(transform, pts_u.x.values, pts_u.y.values)
    r, cidx = np.asarray(r), np.asarray(cidx)
    ok = (r >= 0) & (r < H) & (cidx >= 0) & (cidx < W)
    np.add.at(lab, (r[ok], cidx[ok]), 1)
    y = (lab > 0) & valid
    n_valid = int(valid.sum()); n_pos = int(y.sum())
    neg_idx = np.flatnonzero(valid.ravel() & ~y.ravel())
    k = min(NEG_TARGET, len(neg_idx))
    neg_s = RNG.choice(neg_idx, k, replace=False)
    pos_idx = np.flatnonzero(y.ravel())
    idx = np.concatenate([pos_idx, neg_s])
    df = pd.DataFrame({f: v.ravel()[idx] for f, v in feats.items()})
    df["y"] = np.r_[np.ones(len(pos_idx)), np.zeros(k)].astype(np.int8)
    df["w"] = np.r_[np.ones(len(pos_idx)), np.full(k, len(neg_idx) / k)].astype(np.float32)
    rr, cc = np.unravel_index(idx, (H, W))
    df["x"], df["y_utm"] = (minx + (cc + 0.5) * RES).astype(np.float32), (maxy - (rr + 0.5) * RES).astype(np.float32)
    df["event"] = key
    df = df.dropna(subset=["slope", "pga", "pgv"])
    df.to_parquet(os.path.join(OUT, f"{key}.parquet"))
    # keep full-domain rasters (downsampled to save space) for map figures
    np.savez_compressed(os.path.join(OUT, f"{key}_grid.npz"), **{f: feats[f] for f in ["elev", "slope", "pga", "newmark_cm"]},
                        valid=valid, lab=y, transform=np.array(transform)[:6], crs=crs)
    info = dict(event=key, crs=crs, H=H, W=W, n_landslides_inventory=int(len(pts)), n_points_in_domain=int(ok.sum()),
                n_valid_cells=n_valid, area_km2=n_valid * RES * RES / 1e6, n_pos_cells=n_pos,
                prevalence=n_pos / n_valid, pga_max=float(np.nanmax(np.where(valid, feats["pga"], np.nan))),
                pga_mean_pos=float(np.nanmean(feats["pga"][y])))
    print("  ", info)
    json.dump(info, open(os.path.join(OUT, f"{key}_info.json"), "w"), indent=1)


if __name__ == "__main__":
    keys = sys.argv[1:] or list(EVENTS)
    for k in keys:
        if not os.path.exists(os.path.join(OUT, f"{k}.parquet")):
            process(k)
