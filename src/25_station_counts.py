"""Count seismic stations (instrumental, excluding macroseismic/DYFI) used by the ShakeMap of
each event, from the ShakeMap stationlist (ComCat). Same ShakeMap product preference as the
feature build (latest contributed version over Atlas)."""
import os, sys, json, requests, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.join(os.path.dirname(__file__), "..")
cov = pd.read_csv(os.path.join(ROOT, "data", "global", "event_covariates.csv"))
from events import EVENTS
sel = pd.read_csv(os.path.join(ROOT, "data", "global", "selected_events.csv")).set_index("key")
ids = {k: v["comcat"] for k, v in EVENTS.items()}
ids.update({k: sel.loc[k, "comcat_id"] for k in sel.index})
out = {}
for e in cov.event:
    d = requests.get(f"https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={ids[e]}&format=geojson", timeout=60).json()
    sm = sorted(d["properties"]["products"]["shakemap"], key=lambda s: (s["source"] == "atlas", -int(s["properties"].get("version", 0) or 0)))[0]
    c = sm["contents"]; key = next((k for k in c if k.endswith("stationlist.json")), None)
    n = None
    if key:
        js = requests.get(c[key]["url"], timeout=120).json()
        n = sum(1 for f in js["features"] if str(f["properties"].get("station_type", f["properties"].get("network", ""))).lower() not in ("macroseismic", "dyfi", "intensity", "ciim")
                and f["properties"].get("source", "").upper() not in ("DYFI", "CIIM", "INTENSITY"))
    elif "download/stationlist.txt" in c:
        n = None
    out[e] = n
    print(e, sm["source"], sm["properties"].get("version"), n, flush=True)
cov["shakemap_stations"] = cov.event.map(out).fillna(cov.shakemap_stations)
cov.to_csv(os.path.join(ROOT, "data", "global", "event_covariates.csv"), index=False)
