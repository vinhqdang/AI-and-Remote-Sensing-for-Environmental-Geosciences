"""Publication figures for the Pacific Rim earthquake Newmark / landslide study.

Figure 1 (fig1_events): Pacific-centred world map of the nine event epicentres,
    marker area scaled by moment magnitude Mw.
Figure 2 (fig2_kaikoura_inputs): Kaikoura 2016 inputs - hillshade with slope
    overlay and mapped landslides, ShakeMap PGA, and log10 Newmark displacement.

Outputs (PDF + 300 dpi PNG) are written to <repo>/figures/.
"""
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import requests
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / "data" / "processed"
FIG_DIR = REPO / "figures"
SCRATCH = Path("/tmp/claude-0/ne")
NE_URL = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"
NE_RAW_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
              "master/geojson/ne_110m_land.geojson")

sys.path.insert(0, str(HERE))
from events import EVENTS  # noqa: E402

EPICENTRES = {  # (lon, lat)
    "kaikoura2016": (173.054, -42.737),
    "northridge1994": (-118.537, 34.213),
    "niigata2004": (138.779, 37.226),
    "iwate2008": (140.881, 39.030),
    "aisen2007": (-72.648, -45.243),
    "denali2002": (-147.444, 63.517),
    "cinchona2009": (-84.197, 10.165),
    "porgera2018": (142.754, -6.070),
    "palu2018": (119.846, -0.256),
}

# Label placement: (dx, dy) in points, horizontal alignment
LABEL_POS = {
    "kaikoura2016": ((6, -7), "left"),
    "northridge1994": ((6, 3), "left"),
    "niigata2004": ((-6, 5), "right"),
    "iwate2008": ((6, 4), "left"),
    "aisen2007": ((-6, 0), "right"),
    "denali2002": ((6, 0), "left"),
    "cinchona2009": ((-6, 0), "right"),
    "porgera2018": ((6, -3), "left"),
    "palu2018": ((0, 7), "center"),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})


# ----------------------------------------------------------------------------
# Natural Earth land
# ----------------------------------------------------------------------------
def load_land():
    """Return a GeoDataFrame of Natural Earth 110m land (lon/lat, WGS84)."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    shp = SCRATCH / "extract" / "ne_110m_land.shp"
    geojson = SCRATCH / "ne_110m_land.geojson"
    if not shp.exists() and not geojson.exists():
        try:
            r = requests.get(NE_URL, timeout=120)
            r.raise_for_status()
            zpath = SCRATCH / "ne_110m_land.zip"
            zpath.write_bytes(r.content)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(SCRATCH / "extract")
        except Exception as exc:  # fall back to GeoJSON mirror
            print(f"Natural Earth zip download failed ({exc}); trying GeoJSON mirror")
            r = requests.get(NE_RAW_URL, timeout=120)
            r.raise_for_status()
            geojson.write_bytes(r.content)
    src = shp if shp.exists() else geojson
    gdf = gpd.read_file(src)
    return gdf


def _unwrap_ring(xy):
    """Make a longitude/latitude ring continuous across the antimeridian."""
    lon = np.unwrap(np.deg2rad(xy[:, 0]))  # continuous radians
    lon = np.rad2deg(lon)
    return np.column_stack([lon, xy[:, 1]])


def polygon_paths(geom):
    """Yield matplotlib Paths for a (Multi)Polygon with continuous longitudes."""
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for poly in polys:
        rings = [poly.exterior] + list(poly.interiors)
        verts, codes = [], []
        for ring in rings:
            xy = _unwrap_ring(np.asarray(ring.coords)[:, :2])
            verts.append(xy)
            codes.append(np.concatenate([[MplPath.MOVETO],
                                         np.full(len(xy) - 1, MplPath.LINETO)]))
        yield MplPath(np.vstack(verts), np.concatenate(codes))


def draw_land(ax, land, offsets=(-360, 0, 360)):
    """Draw land at several longitude offsets so the Pacific window is filled."""
    for geom in land.geometry:
        for path in polygon_paths(geom):
            for off in offsets:
                shifted = MplPath(path.vertices + [off, 0], path.codes)
                ax.add_patch(PathPatch(shifted, facecolor="#d9d9d9",
                                       edgecolor="#8c8c8c", lw=0.3, zorder=1))


def lon360_formatter(x, _pos):
    x = x % 360
    if abs(x - 180) < 1e-6:
        return "180°"
    if x < 180:
        return f"{x:.0f}°E"
    return f"{360 - x:.0f}°W"


def lat_formatter(y, _pos):
    if y == 0:
        return "0°"
    return f"{abs(y):.0f}°{'N' if y > 0 else 'S'}"


# ----------------------------------------------------------------------------
# Figure 1: Pacific event map
# ----------------------------------------------------------------------------
def make_fig1(land):
    xmin, xmax, ymin, ymax = 100, 300, -60, 70
    fig, ax = plt.subplots(figsize=(6.69, 4.1))
    ax.set_facecolor("white")
    draw_land(ax, land)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(120, 301, 30))
    ax.set_yticks(np.arange(-60, 61, 30))
    ax.xaxis.set_major_formatter(FuncFormatter(lon360_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
    ax.grid(True, color="#7f7f7f", lw=0.3, alpha=0.5, linestyle="-", zorder=0)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    palette = plt.get_cmap("tab10").colors[:9]
    keys = list(EVENTS.keys())
    info = {}
    for i, key in enumerate(keys):
        with open(DATA / f"{key}_info.json") as fh:
            info[key] = json.load(fh)
        lon, lat = EPICENTRES[key]
        lon360 = lon % 360
        mw = EVENTS[key]["mw"]
        size = 18.0 * 2 ** (mw - 6.0)  # marker area (pt^2) doubles per Mw unit
        ax.scatter(lon360, lat, s=size, color=palette[i], edgecolor="black",
                   linewidth=0.5, alpha=0.9, zorder=5)
        (dx, dy), ha = LABEL_POS[key]
        ax.annotate(EVENTS[key]["label"], (lon360, lat), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center",
                    fontsize=7, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.7))

    # Size legend (Mw)
    handles = []
    for mw in (6.0, 7.0, 8.0):
        ms = np.sqrt(18.0 * 2 ** (mw - 6.0))
        handles.append(Line2D([], [], ls="", marker="o", markersize=ms,
                              markerfacecolor="white", markeredgecolor="black",
                              markeredgewidth=0.5, label=f"$M_w$ {mw:.0f}"))
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.9,
              edgecolor="#bfbfbf", borderpad=0.6, labelspacing=1.0,
              handletextpad=0.8, title="Event size", title_fontsize=7)

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"fig1_events.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Fig1 written: fig1_events.pdf/.png")
    return info


# ----------------------------------------------------------------------------
# Figure 2: Kaikoura inputs
# ----------------------------------------------------------------------------
def hillshade(elev, cell, azimuth_deg=315.0, altitude_deg=45.0):
    """Lambertian hillshade on a north-up grid (row 0 = north)."""
    gy_row, gx_col = np.gradient(elev.astype(np.float64), cell)
    gx = gx_col                # d(z)/d(east)
    gy = -gy_row               # d(z)/d(north): rows increase southward
    az = np.deg2rad(90.0 - azimuth_deg)  # compass -> math angle (E=0, CCW)
    alt = np.deg2rad(altitude_deg)
    light = np.array([np.cos(alt) * np.cos(az), np.cos(alt) * np.sin(az),
                      np.sin(alt)])
    norm = np.sqrt(1.0 + gx ** 2 + gy ** 2)
    shade = (-gx * light[0] - gy * light[1] + light[2]) / norm
    return np.clip(shade, 0.0, 1.0).astype(np.float32)


def cmap_with_bad_transparent(name):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(alpha=0.0)
    return cmap


def make_fig2():
    key = "kaikoura2016"
    grid = np.load(DATA / f"{key}_grid.npz")
    with open(DATA / f"{key}_info.json") as fh:
        info = json.load(fh)

    transform = grid["transform"]
    a, b, c, d, e, f = [float(v) for v in transform]
    crs = str(grid["crs"])  # EPSG:32759 = WGS 84 / UTM zone 59S
    crs_label = "UTM 59S" if crs == "EPSG:32759" else crs
    H, W = grid["elev"].shape
    cell = abs(a)

    valid = grid["valid"]
    elev = grid["elev"]
    slope = grid["slope"]
    pga = grid["pga"]
    newmark = grid["newmark_cm"]
    lab = grid["lab"]

    # Full-resolution hillshade (gradients need native spacing), then mask/stride
    hs = hillshade(elev, cell, 315.0, 45.0)
    hs = np.where(valid, hs, np.nan)
    slope_m = np.where(valid, slope, np.nan)
    pga_m = np.where(valid, pga, np.nan)
    log_nm = np.where(valid, np.log10(newmark + 0.1), np.nan)

    s = 4
    hs_d = hs[::s, ::s]
    slope_d = slope_m[::s, ::s]
    pga_d = pga_m[::s, ::s]
    nm_d = log_nm[::s, ::s]

    # Extent in km (UTM); row 0 is north -> origin='upper' with top = f
    x0 = c / 1000.0
    x1 = (c + W * a) / 1000.0
    ytop = f / 1000.0
    ybot = (f + H * e) / 1000.0
    extent = (x0, x1, ybot, ytop)

    pga_vmax = float(np.nanmax(pga_m))
    nm_valid = log_nm[valid]
    nm_vmin = -1.0  # log10(0.1 cm): floor of the displayed range
    nm_vmax = float(np.nanpercentile(nm_valid, 99.5))
    print(f"Kaikoura: PGA max {pga_vmax:.2f} g; log10 Newmark range "
          f"[{nm_vmin}, {nm_vmax:.2f}] (99.5th pct, extend='max')")

    fig, axes = plt.subplots(1, 3, figsize=(6.69, 2.95))
    ax_a, ax_b, ax_c = axes

    # ---- (a) hillshade + slope overlay + landslides
    im_hs = ax_a.imshow(hs_d, cmap=cmap_with_bad_transparent("gray"),
                        vmin=0, vmax=1, extent=extent, origin="upper",
                        interpolation="bilinear")
    im_sl = ax_a.imshow(slope_d, cmap=cmap_with_bad_transparent("YlOrBr"),
                        vmin=0, vmax=40, extent=extent, origin="upper",
                        alpha=0.3, interpolation="bilinear")
    rows, cols = np.nonzero(lab)
    xs = (c + (cols + 0.5) * a) / 1000.0
    ys = (f + (rows + 0.5) * e) / 1000.0
    ax_a.scatter(xs, ys, s=0.2, c="black", linewidths=0, zorder=3,
                 rasterized=True,
                 label=f"Mapped landslides (n = {info['n_pos_cells']:,})")
    ax_a.legend(loc="lower right", fontsize=6, frameon=True, framealpha=0.9,
                edgecolor="#bfbfbf", markerscale=6, handletextpad=0.3,
                borderpad=0.4)
    cb_a = fig.colorbar(im_sl, ax=ax_a, orientation="horizontal",
                        fraction=0.05, pad=0.18, aspect=25)
    cb_a.set_label("Slope (°)", fontsize=7)
    cb_a.ax.tick_params(labelsize=6)

    # ---- (b) PGA
    im_pga = ax_b.imshow(pga_d, cmap=cmap_with_bad_transparent("viridis"),
                         vmin=0, vmax=pga_vmax, extent=extent, origin="upper",
                         interpolation="bilinear")
    cb_b = fig.colorbar(im_pga, ax=ax_b, orientation="horizontal",
                        fraction=0.05, pad=0.18, aspect=25)
    cb_b.set_label("PGA (g), ShakeMap", fontsize=7)
    cb_b.ax.tick_params(labelsize=6)

    # ---- (c) log10 Newmark displacement
    cmap_c = cmap_with_bad_transparent("magma")
    im_nm = ax_c.imshow(nm_d, cmap=cmap_c, vmin=nm_vmin, vmax=nm_vmax,
                        extent=extent, origin="upper", interpolation="bilinear")
    cb_c = fig.colorbar(im_nm, ax=ax_c, orientation="horizontal",
                        fraction=0.05, pad=0.18, aspect=25, extend="max")
    ticks = [t for t in (-1, 0, 1, 2, 3) if nm_vmin <= t <= nm_vmax]
    cb_c.set_ticks(ticks)
    cb_c.set_ticklabels([f"{10 ** t:g}" for t in ticks])
    cb_c.set_label("Newmark displacement (cm)", fontsize=7)
    cb_c.ax.tick_params(labelsize=6)

    panel_letters = ["(a)", "(b)", "(c)"]
    titles = ["Hillshade, slope, landslides", "ShakeMap PGA", "Newmark displacement"]
    for ax, letter, title in zip(axes, panel_letters, titles):
        ax.set_facecolor("white")
        ax.set_title(title, fontsize=8, pad=3)
        ax.text(0.03, 0.97, letter, transform=ax.transAxes, ha="left",
                va="top", fontsize=9, fontweight="bold", color="white",
                zorder=10, bbox=dict(boxstyle="round,pad=0.15", fc="black",
                                     ec="none", alpha=0.55))
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax.tick_params(length=2.5, pad=2)
        ax.set_xlabel(f"Easting (km, {crs_label})" if ax is ax_b else "Easting (km)")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax_a.set_ylabel("Northing (km)")
    for ax in (ax_b, ax_c):
        ax.set_yticklabels([])

    fig.tight_layout(pad=0.4, w_pad=0.8)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"fig2_kaikoura_inputs.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Fig2 written: fig2_kaikoura_inputs.pdf/.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    land = load_land()
    print(f"Loaded {len(land)} land features")
    info = make_fig1(land)
    for key, rec in info.items():
        print(f"  {key:15s} n_pos={rec['n_pos_cells']:>7,d} "
              f"prevalence={rec['prevalence']:.5f}")
    make_fig2()


if __name__ == "__main__":
    main()
