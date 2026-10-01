"""DIAGNOSTIC gauge shortlist per configured location (reads cached BoM station lists; changes nothing).

Writes reports/v2/gauge_shortlist.md and .csv. Two kinds of gauge:
* automated: has a BoM daily file on the public FTP product IDCKWCDEA0 (the pipeline downloads these itself);
  completeness is measured over the dataset period;
* manual / CDO-only: listed in BoM's station list IDCJMC0014 but daily data only via Climate Data Online, which needs a
  manual download. Completeness and quality flags are unknown until the official file is imported.
Ranking: open gauges first, then those whose record covers the forecast archive (ECMWF IFS single runs from
2024-03-14; JMA GSM from 2016), then distance.
"""
import json
from pathlib import Path

import pandas as pd
import tomllib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "v2"
OUT.mkdir(parents=True, exist_ok=True)
locs = tomllib.loads((ROOT / "config.toml").read_text())["locations"]


def slug(n):
    return n.lower().replace(" ", "_")


rows = []
for e in locs:
    s = slug(e["name"])
    rs = json.loads((ROOT / "reports" / s / "run_summary.json").read_text())
    current = rs["station"]["station_id"]
    auto = pd.read_csv(ROOT / "reports" / s / "station_candidates.csv", dtype={"station_id": str})
    for r in auto.itertuples():
        if r.completeness <= 0:
            continue
        rows.append({"location": e["name"], "station_id": r.station_id.zfill(6), "station_name": r.station_name,
                     "distance_km": round(r.distance_km, 1), "operating": f"{r.db_start_date} .. open",
                     "open": True, "start_year": int(str(r.db_start_date)[:4]),
                     "daily_rainfall": "automated BoM daily file (FTP IDCKWCDEA0)",
                     "completeness_2016_2026": f"{r.completeness:.1%}",
                     "quality_info": "no BoM quality flag in this product; pipeline QC applied",
                     "station_number_changed": bool(r.id_ambiguous),
                     "current_gauge": r.station_id.zfill(6) == current})
    allst = pd.read_csv(ROOT / "reports" / s / "nearby_bom_stations_all_IDCJMC0014.csv", dtype={"station_id": str})
    ids_auto = {x.zfill(6) for x in auto.station_id}
    for r in allst[allst.distance_km <= 12].itertuples():
        sid = r.station_id.zfill(6)
        if sid in ids_auto:
            continue
        is_open = str(r.end_year) == ".."
        end_year = None if is_open else int(r.end_year)
        if not is_open and end_year < 2024:
            continue                                    # closed before the ECMWF archive starts: no overlap
        rows.append({"location": e["name"], "station_id": sid, "station_name": r.station_name,
                     "distance_km": round(r.distance_km, 1),
                     "operating": f"{r.start_year} .. {'open' if is_open else r.end_year}", "open": is_open,
                     "start_year": int(r.start_year),
                     "daily_rainfall": "Climate Data Online only (manual download)",
                     "completeness_2016_2026": "unknown until imported",
                     "quality_info": "CDO file carries BoM quality flag (Y = quality controlled)",
                     "station_number_changed": None, "current_gauge": sid == current})
S = pd.DataFrame(rows)
S["covers_ecmwf_archive"] = S.start_year <= 2024
S["covers_jma_archive"] = S.start_year <= 2016
S = S.sort_values(["location", "open", "covers_ecmwf_archive", "distance_km"], ascending=[True, False, False, True])
S.to_csv(OUT / "gauge_shortlist.csv", index=False)

md = ["# Gauge shortlist (v2 planning; nothing has been changed)\n",
      "Sources: BoM IDCKWCDEA0 `stations_db.txt` (automated daily files) and IDCJMC0014 `stations.zip` (all stations), "
      "both cached in `data/raw/` on 2026-09-30. Distances are from the configured suburb coordinates. Manual gauges "
      "within 12 km that were open at some point since 2024 are listed; completeness for those is unknown until the "
      "Climate Data Online file is imported.\n"]
for loc, g in S.groupby("location", sort=False):
    md.append(f"\n## {loc}\n")
    cols = ["station_id", "station_name", "distance_km", "operating", "daily_rainfall", "completeness_2016_2026",
            "covers_ecmwf_archive", "current_gauge"]
    t = g[cols].head(8)
    md.append("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.values]
(OUT / "gauge_shortlist.md").write_text("\n".join(md) + "\n", encoding="utf-8")
pd.set_option("display.width", 250)
for loc, g in S.groupby("location", sort=False):
    print("==", loc)
    print(g[["station_id", "station_name", "distance_km", "operating", "daily_rainfall", "completeness_2016_2026",
             "current_gauge"]].head(7).to_string(index=False))
