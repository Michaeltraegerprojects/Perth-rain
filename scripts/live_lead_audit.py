"""DIAGNOSTIC: what did the Previous Runs API return for FUTURE valid times in the live request, and which
served rows depended on model runs that had not yet been issued at prediction time?

Reads only cached live responses (data/raw/open_meteo/live/) - no network.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perthrain import forecasts  # noqa: E402

live = ROOT / "data" / "raw" / "open_meteo" / "live" / "previous_runs"
for meta_p in sorted(live.glob("*.meta.json")):
    meta = json.loads(meta_p.read_text())
    body = json.loads(meta_p.with_name(meta_p.name.replace(".meta", "")).read_text())
    p = meta["params"]
    h = pd.DataFrame(body["hourly"])
    h["time"] = pd.to_datetime(h.time).dt.tz_localize("UTC")
    cols = [c for c in h.columns if c.startswith("precipitation_previous_day")]
    print(f"\n=== {p['models']} lat={p['latitude']} lon={p['longitude']} {p['start_date']}..{p['end_date']} "
          f"retrieved {meta['retrieved_at_utc']}")
    ret = pd.Timestamp(meta["retrieved_at_utc"])
    for c in cols:
        n = int(c[-1])
        v = h.set_index("time")[c]
        last = v.last_valid_index()
        # inferred contributing run for each hour under the documented/verified rule
        inf = forecasts.inferred_run_time(pd.Series(v.index, index=v.index), p["models"], n)
        future_run = inf > ret
        print(f"  {c}: non-null {int(v.notna().sum())}/{len(v)}, last non-null {last}, "
              f"hours whose inferred run is AFTER retrieval: {int(future_run.sum())}, "
              f"of those non-null: {int((future_run & v.notna()).sum())}")
    # are the offsets identical at future valid times? (evidence of filling from one latest run)
    fut = h[h.time > ret]
    if len(cols) >= 2 and len(fut):
        same12 = np.isclose(fut[cols[0]], fut[cols[1]], equal_nan=True).mean()
        same23 = np.isclose(fut[cols[1]], fut[cols[2]], equal_nan=True).mean() if len(cols) > 2 else np.nan
        nz = fut[fut[cols].fillna(0).abs().sum(axis=1) > 0]
        same_nz = np.isclose(nz[cols[1]], nz[cols[2]], equal_nan=True).mean() if len(nz) and len(cols) > 2 else np.nan
        print(f"  future valid times ({len(fut)} h): share day1==day2 {same12:.2f}, day2==day3 {same23:.2f}; "
              f"on the {len(nz)} rainy future hours day2==day3 {same_nz:.2f}")
