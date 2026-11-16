"""Catalogue of earthquake-triggered landslide inventories in the USGS ScienceBase
"Open Repository of Earthquake-Triggered Ground-Failure Inventories" (Schmitt et al., 2017).

Every ScienceBase inventory item is linked to its event folder (matched on the
event date and place in the title/body text) and to a USGS ComCat earthquake
(verified by a successful ComCat query). Layer statistics are read from the
shapefile headers with HTTP streaming, so the landslide layers are not downloaded.

Output: data/global/catalogue.csv (one row per ScienceBase inventory item).
API responses are cached under data/global/cache/ so re-runs are cheap.
"""
import datetime as dt
import hashlib
import json
import os
import re
import struct
import sys
import time

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(ROOT, "data", "global")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
OUT_CSV = os.path.join(OUT_DIR, "catalogue.csv")
os.makedirs(CACHE_DIR, exist_ok=True)

INV_PARENT = "583f4114e4b04fc80e3c4a1a"
EVENT_PARENT = "586d824ce4b0f5ce109fc9a6"
INV_FIELDS = "title,facets,summary,body,citation,spatial,webLinks,dates"
EVENT_FIELDS = "title,webLinks,body"
SB_LIST = "https://www.sciencebase.gov/catalog/items?parentId={pid}&format=json&max=300&fields={fields}"
COMCAT = "https://earthquake.usgs.gov/fdsnws/event/1/query?{q}"
LARGE_FILE = 50 * 1024 ** 2

# Inventory items whose text does not name an event date/place clearly enough for
# the automatic matcher. Maps sb_id -> event folder id (filled after review).
OVERRIDES = {}

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "landslide-inventory-catalogue/1.0"

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}
SHP_GEOM = {0: "Null", 1: "Point", 8: "Point", 11: "Point", 18: "Point", 21: "Point", 28: "Point",
            3: "Line", 13: "Line", 23: "Line", 5: "Polygon", 15: "Polygon", 25: "Polygon"}
GENERIC_WORDS = {"earthquake", "sequence", "region", "landslide", "inventory", "induced",
                 "triggered", "earthquakes", "ground", "failure", "failures"}


# ---------------------------------------------------------------- network helpers
def _cache_file(key):
    return os.path.join(CACHE_DIR, hashlib.sha1(key.encode()).hexdigest() + ".json")


def get_json(url, retries=3):
    """GET a JSON document (cached). Returns None for 204/404 (unknown ComCat id)."""
    path = _cache_file(url)
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    last = None
    for k in range(retries):
        try:
            r = SESSION.get(url, timeout=120)
            if r.status_code in (204, 404):
                return None
            r.raise_for_status()
            data = r.json()
            with open(path, "w") as fh:
                json.dump(data, fh)
            return data
        except Exception as exc:  # network hiccup: retry
            last = exc
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"failed after {retries} tries: {url}: {last}")


def get_head(url, nbytes, retries=3):
    """First nbytes of a remote file, streamed so the rest of the file is never downloaded."""
    path = _cache_file(f"head{nbytes}:{url}")
    if os.path.exists(path):
        with open(path) as fh:
            return bytes.fromhex(fh.read())
    last = None
    for k in range(retries):
        try:
            buf = b""
            with SESSION.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=1024):
                    buf += chunk
                    if len(buf) >= nbytes:
                        break
            buf = buf[:nbytes]
            with open(path, "w") as fh:
                fh.write(buf.hex())
            return buf
        except Exception as exc:
            last = exc
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"failed after {retries} tries: {url}: {last}")


# ---------------------------------------------------------------- text helpers
def strip_html(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _date(y, m, d):
    try:
        return dt.date(int(y), int(m), int(d))
    except ValueError:
        return None


def parse_dates(text):
    """Calendar dates written as 2001-01-13, January 13, 2001 or 13 January 2001."""
    out = []
    for m in re.finditer(r"(\d{4})-(\d{2})-(\d{2})", text):
        out.append(_date(m.group(1), m.group(2), m.group(3)))
    for m in re.finditer(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})\b", text):
        mo = MONTHS.get(m.group(1).lower())
        if mo:
            out.append(_date(m.group(3), mo, m.group(2)))
    for m in re.finditer(r"\b(\d{1,2})\s+([A-Za-z]{3,9}),?\s+(\d{4})\b", text):
        mo = MONTHS.get(m.group(2).lower())
        if mo:
            out.append(_date(m.group(3), mo, m.group(1)))
    return [d for d in out if d]


def parse_folder_title(title):
    """'2015-04-25 Nepal M 7.8' -> (date, 'Nepal', 7.8)."""
    m = re.match(r"(\d{4}-\d{2}-\d{2})\s+(.*?)\s*,?\s*M\s*w?\s*([\d.]+)\s*$", title)
    if m:
        return _date(*m.group(1).split("-")), m.group(2).strip(" ,"), float(m.group(3))
    m = re.match(r"(\d{4}-\d{2}-\d{2})\s+(.*)$", title)
    if m:
        return _date(*m.group(1).split("-")), m.group(2).strip(" ,"), None
    return None, title.strip(), None


def mentions_mag(text, mag):
    if mag is None:
        return False
    return any(abs(float(x) - mag) < 0.05 for x in re.findall(r"M\s*w?\s*(\d\.\d)", text))


# ---------------------------------------------------------------- ScienceBase
def sb_children(parent, fields):
    data = get_json(SB_LIST.format(pid=parent, fields=fields))
    return data.get("items", [])


def layer_info(files):
    """Geometry type, feature count and main-layer metadata from shapefile headers."""
    by_ext = {}
    for f in files:
        ext = f["name"].lower().rsplit(".", 1)[-1] if "." in f["name"] else ""
        by_ext.setdefault(ext, f)
    if "shp" not in by_ext:
        return {"has_shp": False, "geometry_type": "", "n_features": None}
    shp = by_ext["shp"]
    head = get_head(shp["url"], 36)
    geom = SHP_GEOM.get(struct.unpack("<i", head[32:36])[0], "Other")
    if "dbf" in by_ext:  # dBase header: record count is a uint32 at byte 4
        n = struct.unpack("<I", get_head(by_ext["dbf"]["url"], 8)[4:8])[0]
    elif "shx" in by_ext and by_ext["shx"].get("size"):  # 100-byte header + 8 bytes per record
        n = (int(by_ext["shx"]["size"]) - 100) // 8
    elif int(shp.get("size") or 0) > LARGE_FILE:
        n = -1
    else:
        n = -1
    return {"has_shp": True, "geometry_type": geom, "n_features": n}


def pick_main_layer(facets):
    """Main landslide layer: non-extent facet with a shapefile; polygons first, then most features."""
    cands = []
    for facet in facets:
        name = facet.get("name", "")
        if "mapping_ext" in name.lower():
            continue
        info = layer_info(facet.get("files", []))
        if not info["has_shp"]:
            continue
        info["name"] = name
        cands.append(info)
    if not cands:
        return None
    cands.sort(key=lambda c: (c["geometry_type"] != "Polygon", -(c["n_features"] or 0)))
    return cands[0]


def total_size(facets):
    return sum(int(f.get("size") or 0) for facet in facets for f in facet.get("files", []))


# ---------------------------------------------------------------- event folders
def comcat_ids(folder):
    ids = []
    for link in folder.get("webLinks", []) or []:
        m = re.search(r"eventpage/([A-Za-z0-9]+)", link.get("uri", ""))
        if m and m.group(1) not in ids:
            ids.append(m.group(1))
    return ids


def match_inventory(inv, folders):
    """Score each event folder against an inventory's title/body; return best folder or None."""
    text = f"{inv['title']} {strip_html(inv.get('body') or inv.get('summary'))}"
    dates = parse_dates(text)
    words = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", text)}
    best, best_score = None, 0
    for f in folders:
        score = 0
        if f["date"] in dates:
            score += 10
        elif any(abs((f["date"] - d).days) <= 1 for d in dates if f["date"]):
            score += 5
        place_words = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", f["place"])} - GENERIC_WORDS
        score += 3 * len(place_words & words) > 0 and 3 or 0
        score += 2 if mentions_mag(text, f["mag"]) else 0
        if score > best_score:
            best, best_score = f, score
    return best if best_score >= 10 else None


def load_folders():
    folders = []
    for item in sb_children(EVENT_PARENT, EVENT_FIELDS):
        date, place, mag = parse_folder_title(item["title"])
        folders.append({"id": item["id"], "title": item["title"], "date": date, "place": place,
                        "mag": mag, "webLinks": item.get("webLinks", []),
                        "comcat_ids": comcat_ids(item)})
    return folders


# ---------------------------------------------------------------- ComCat
def comcat_by_id(eid, folder):
    """Query ComCat by event id; accept only if the origin time is within 1 day of the folder date."""
    data = get_json(COMCAT.format(q=f"eventid={eid}&format=geojson"))
    if not data or "properties" not in data:
        return None
    t = dt.datetime.fromtimestamp(data["properties"]["time"] / 1000, dt.timezone.utc)
    if folder["date"] and abs((t.date() - folder["date"]).days) > 1:
        return None
    return data


def comcat_search(folder):
    """Fallback: search by time window and place name; closest magnitude wins."""
    if not folder["date"]:
        return None
    start = folder["date"] - dt.timedelta(days=1)
    end = folder["date"] + dt.timedelta(days=2)
    mag = folder["mag"] or 5.0
    q = (f"starttime={start}&endtime={end}&minmagnitude={max(4.5, mag - 0.5):.1f}"
         f"&orderby=magnitude&limit=50&format=geojson")
    data = get_json(COMCAT.format(q=q))
    feats = (data or {}).get("features", [])
    if not feats:
        return None
    words = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", folder["place"])} - GENERIC_WORDS
    named = [f for f in feats if any(w in f["properties"].get("place", "").lower() for w in words)]
    pool = named or feats
    return min(pool, key=lambda f: abs((f["properties"].get("mag") or 0) - mag))


def comcat_lookup(folder):
    """Returns (detail geojson or None, method)."""
    for eid in folder["comcat_ids"]:
        data = comcat_by_id(eid, folder)
        if data:
            return data, "eventpage"
    data = comcat_search(folder)
    if data:
        return data, "search"
    return None, "none"


def shakemap_stations(products):
    sm = (products.get("shakemap") or [None])[0]
    if not sm:
        return None
    url = next((v["url"] for k, v in sm.get("contents", {}).items() if k.endswith("grid.xml")), None)
    if not url:
        return None
    head = get_head(url, 4096).decode("utf-8", "replace")
    m = re.search(r'seismic_stations="(\d+)"', head)
    return int(m.group(1)) if m else None


def rake_of(products):
    for key in ("moment-tensor", "focal-mechanism"):
        prods = products.get(key) or []
        if prods:
            rake = prods[0].get("properties", {}).get("nodal-plane-1-rake")
            if rake not in (None, ""):
                return float(rake)
    return None


def mechanism(rake):
    if rake is None:
        return ""
    if -150 < rake < -30:
        return "normal"
    if 30 < rake < 150:
        return "reverse"
    return "strike-slip"


def pacific_rim(lon, lat):
    """Boxes (lon_min, lon_max, lat_min, lat_max) covering Pacific-bordering regions in the task."""
    if lon is None or lat is None or pd.isna(lon) or pd.isna(lat):
        return None
    boxes = [
        (-170, -129, 51, 72),    # Alaska
        (-125, -114, 32, 49.5),  # US West Coast
        (-160.5, -154, 18.5, 22.5),  # Hawaii
        (-118, -86, 14, 33),     # Mexico
        (-93, -77, 7, 18.5),     # Central America
        (-82, -66, -56, 13),     # Andes: Colombia, Ecuador, Peru, Chile
        (119, 123, 21, 26),      # Taiwan
        (116, 127, 4, 21),       # Philippines
        (95, 141, -11, 6),       # Indonesia
        (128, 146, 30, 46),      # Japan
        (140, 160, -12, 0),      # Papua New Guinea
        (166, 179, -48, -34),    # New Zealand
    ]
    return any(a <= lon <= b and c <= lat <= d for a, b, c, d in boxes)


def clean_text(text, n=None):
    text = strip_html(text)
    return text[:n] if n else text


# ---------------------------------------------------------------- main
def main():
    print("Fetching ScienceBase inventory and event folder listings ...")
    inventories = sb_children(INV_PARENT, INV_FIELDS)
    folders = load_folders()
    print(f"  {len(inventories)} inventory items, {len(folders)} event folders")

    # 1) link inventories to event folders
    links = {}
    for inv in inventories:
        f = None
        if inv["id"] in OVERRIDES:
            f = next(x for x in folders if x["id"] == OVERRIDES[inv["id"]])
        else:
            f = match_inventory(inv, folders)
        links[inv["id"]] = f

    # 2) ComCat per linked event folder (each folder queried once)
    comcat = {}
    for f in folders:
        if not any(links[i["id"]] is f for i in inventories):
            continue
        data, method = comcat_lookup(f)
        comcat[f["id"]] = (data, method)
        status = data["id"] if data else "NOT FOUND"
        print(f"  ComCat {f['title']:<40} -> {status} ({method})")

    # 3) one row per inventory
    rows = []
    for inv in inventories:
        f = links[inv["id"]]
        facets = inv.get("facets", [])
        main = pick_main_layer(facets)
        data, method = comcat.get(f["id"], (None, "none")) if f else (None, "none")
        props = data["properties"] if data else {}
        products = props.get("products", {}) if data else {}
        lon = lat = depth = None
        if data:
            lon, lat, depth = data["geometry"]["coordinates"][:3]
        rake = rake_of(products) if data else None
        has_ext = any("mapping_ext" in x.get("name", "").lower() for x in facets)
        body = inv.get("body") or inv.get("summary") or ""
        rows.append({
            "sb_id": inv["id"],
            "title": inv["title"],
            "has_landslide_layer": main is not None,
            "event_folder_id": f["id"] if f else "",
            "event_label": "",
            "event_date": (dt.datetime.fromtimestamp(props["time"] / 1000, dt.timezone.utc).date().isoformat()
                           if data else (f["date"].isoformat() if f and f["date"] else "")),
            "comcat_id": data["id"] if data else "",
            "comcat_method": method if data else "",
            "mag": props.get("mag", f["mag"] if f else None),
            "lat": lat, "lon": lon, "depth_km": depth,
            "n_features": main["n_features"] if main else None,
            "geometry_type": main["geometry_type"] if main else "",
            "main_layer": main["name"] if main else "",
            "has_mapping_extent": has_ext,
            "total_size_bytes": total_size(facets),
            "has_shakemap": bool(products.get("shakemap")) if data else False,
            "shakemap_n_stations": shakemap_stations(products) if data and products.get("shakemap") else None,
            "rake": rake,
            "mechanism": mechanism(rake),
            "pacific_rim": pacific_rim(lon, lat),
            "citation": clean_text(inv.get("citation")),
            "body_excerpt": clean_text(body, 400),
        })

    df = pd.DataFrame(rows)
    # event labels: place name + year; add the ISO date when two events share a name and year
    def label(row):
        f = links.get(row["sb_id"])
        if not f:
            return ""
        place = re.split(r",", f["place"])[0].strip() or f["place"]
        year = f["date"].year if f["date"] else ""
        return f"{place} {year}"
    df["event_label"] = [label(r) for r in rows]
    dup = df["event_label"].duplicated(keep=False) & (df["event_label"] != "")
    for i in df.index[dup]:
        f = links[df.at[i, "sb_id"]]
        df.at[i, "event_label"] = f"{df.at[i, 'event_label']} ({f['date'].isoformat()})"
    df["event_label"] = df["event_label"]

    df["n_features"] = df["n_features"].astype("Int64")
    df["shakemap_n_stations"] = df["shakemap_n_stations"].astype("Int64")
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    df["pacific_rim"] = df["pacific_rim"].astype("boolean")
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(df)} rows)")

    # 4) summary
    lay = df[df["has_landslide_layer"]]
    print("\n=== Summary ===")
    print(f"Inventory items: {len(df)} ({len(lay)} with a landslide shapefile layer, "
          f"{len(df) - len(lay)} reference-only records)")
    print(f"Distinct linked events (folders): {df['event_folder_id'].replace('', pd.NA).dropna().nunique()}")
    print(f"Distinct linked ComCat events: {df['comcat_id'].replace('', pd.NA).dropna().nunique()}")
    print(f"Items with no event link: {(df['event_folder_id'] == '').sum()}")
    print(f"Items with no ComCat link: {(df['comcat_id'] == '').sum()}")
    big = lay[(lay["n_features"] >= 100) & (lay["has_shakemap"] == True)]  # noqa: E712
    print(f"Layers with >=100 landslides and has_shakemap: {len(big)} "
          f"(distinct events: {big['event_folder_id'].nunique()})")
    print("\n" + lay.sort_values("event_date")[
        ["event_label", "mag", "n_features", "geometry_type", "has_mapping_extent", "mechanism"]
    ].to_string(index=False))
    unlinked = df[(df["event_folder_id"] == "") | (df["comcat_id"] == "")]
    if len(unlinked):
        print("\nNot fully linked:")
        print(unlinked[["sb_id", "title", "event_folder_id", "comcat_id"]].to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
