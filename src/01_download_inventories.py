"""Download coseismic landslide inventories.

ScienceBase USGS open repository of earthquake-triggered ground-failure inventories
(Schmitt et al., 2017) for eight events, and the GNS Science Kaikoura 2016 landslide
source-area centroids (Massey et al., inventory V2) from the GNS ArcGIS REST service.
"""
import json, os, sys, time, requests
import geopandas as gpd
from shapely.geometry import Point
sys.path.insert(0, os.path.dirname(__file__))
from events import EVENTS

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "inventories")
os.makedirs(RAW, exist_ok=True)
S = requests.Session()

def get(url, **kw):
    for k in range(5):
        try:
            r = S.get(url, timeout=120, **kw); r.raise_for_status(); return r
        except Exception as e:
            print("retry", k, e); time.sleep(2 * (k + 1))
    raise RuntimeError(url)

def sciencebase(key, ev):
    item = get(f"https://www.sciencebase.gov/catalog/item/{ev['sb']}?format=json").json()
    out = os.path.join(RAW, key); os.makedirs(out, exist_ok=True)
    for facet in item.get("facets", []):
        if facet.get("name") not in (ev["facet"], ev["extent"]):
            continue
        for f in facet.get("files", []):
            p = os.path.join(out, f["name"])
            if not os.path.exists(p):
                open(p, "wb").write(get(f["url"]).content)
    print(key, sorted(os.listdir(out)))

def kaikoura(key):
    out = os.path.join(RAW, key); os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "kaikoura_sources.gpkg")
    if os.path.exists(p):
        return
    base = "https://gis.gns.cri.nz/server/rest/services/EILD/EILD_V2_Map/MapServer/0/query"
    feats, off = [], 0
    while True:
        r = get(base, params=dict(where="1=1", outFields="source_id", outSR=4326, f="json",
                                  resultOffset=off, resultRecordCount=2000, orderByFields="objectid")).json()
        fs = r.get("features", [])
        feats += [(f["attributes"]["source_id"], Point(f["geometry"]["x"], f["geometry"]["y"])) for f in fs]
        off += len(fs)
        if not fs or not r.get("exceededTransferLimit"):
            break
    g = gpd.GeoDataFrame({"source_id": [a for a, _ in feats]}, geometry=[b for _, b in feats], crs=4326)
    g.to_file(p, driver="GPKG"); print(key, len(g))

if __name__ == "__main__":
    for k, ev in EVENTS.items():
        kaikoura(k) if ev["sb"] is None else sciencebase(k, ev)
