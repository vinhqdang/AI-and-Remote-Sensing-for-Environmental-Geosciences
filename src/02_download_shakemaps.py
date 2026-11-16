"""Download USGS ShakeMap grid.xml for each event (ComCat). Prefers the latest
contributed 'us'/regional ShakeMap over the ShakeMap Atlas version when both exist."""
import os, sys, json, requests
sys.path.insert(0, os.path.dirname(__file__))
from events import EVENTS
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "shakemaps"); os.makedirs(OUT, exist_ok=True)
meta = {}
for k, ev in EVENTS.items():
    d = requests.get(f"https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={ev['comcat']}&format=geojson", timeout=60).json()
    sm = d["properties"]["products"]["shakemap"]
    sm = sorted(sm, key=lambda s: (s["source"] == "atlas", -int(s["properties"].get("version", 0) or 0)))
    best = sm[0]
    url = best["contents"]["download/grid.xml"]["url"]
    p = os.path.join(OUT, f"{k}_grid.xml")
    if not os.path.exists(p):
        open(p, "wb").write(requests.get(url, timeout=300).content)
    meta[k] = dict(source=best["source"], version=best["properties"].get("version"), url=url, title=d["properties"]["title"])
    print(k, meta[k]["source"], meta[k]["version"], d["properties"]["title"])
json.dump(meta, open(os.path.join(OUT, "shakemap_sources.json"), "w"), indent=1)
