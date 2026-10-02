"""DIAGNOSTIC: provenance status of every SERVED row in the newest prediction record of each location.

For each served row and each model input it used:
  * sr_* inputs come from the Single Runs API with an explicit run= time -> VERIFIED by request (run time is in the URL).
  * pr_* inputs come from the Previous Runs API, whose response does not name the run. Each hour is compared with the
    run the offset rule names (floor_6h(ceil_to_native_step(t)) - 24*N h), fetched from the Single Runs archive.
    All hours equal (|d| <= 0.05 mm) -> VERIFIED; any named run not archived -> UNVERIFIED (kept, labelled);
    any mismatch -> FAILED.
A row is fully verified only if every input is VERIFIED. Writes reports/served_row_verification.csv.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
from perthrain import forecasts  # noqa: E402
from perthrain.calibrate import FEATURE_SETS  # noqa: E402
from perthrain.config import load_configs  # noqa: E402
from perthrain.pipeline import make_fetcher  # noqa: E402

TOL = 0.051
LIVE = ROOT / "data" / "raw" / "open_meteo" / "live" / "previous_runs"
cfgs = load_configs("config.toml")
F = make_fetcher(cfgs[0])
_runs = {}


def single_run(model, lat, lon, run):
    key = (model, round(lat, 4), round(lon, 4), run)
    if key not in _runs:
        p = forecasts.single_run_params(lat, lon, model, run, 6)
        res = F.get_json(forecasts.SINGLE_RUNS_URL, p, "open_meteo/verify_live", label=f"verify {model} {run}",
                       retry_failed=True)   # a run requested before it was published must be asked again
        _runs[key] = (forecasts.parse_single_run(res.body, model, run, res.url, res.params, res.retrieved_at_utc)
                      .set_index("valid_end_utc").precip_mm if res.ok else None)
    return _runs[key]


META_ID = {"jma_gsm": "jma_gsm", "ecmwf_ifs025": "ecmwf_ifs025", "ncep_gfs_global": "ncep_gfs013"}
_meta = {}


def last_run_init(model):
    """Latest run the model's own Open-Meteo metadata reports as available (checked now, after the issue)."""
    if model not in _meta:
        import requests
        try:
            j = requests.get(f"https://api.open-meteo.com/data/{META_ID.get(model, model)}/static/meta.json",
                             headers={"User-Agent": cfgs[0].user_agent}, timeout=30).json()
            _meta[model] = pd.Timestamp(j["last_run_initialisation_time"], unit="s", tz="UTC")
        except Exception:
            _meta[model] = None
    return _meta[model]


def live_response(model, lat, retrieved):
    for mp in LIVE.glob("*.meta.json"):
        m = json.loads(mp.read_text())
        if (m["params"].get("models") == model and abs(float(m["params"]["latitude"]) - lat) < 1e-4
                and m["retrieved_at_utc"][:16] == retrieved[:16]):
            body = json.loads(mp.with_name(mp.name.replace(".meta", "")).read_text())
            h = pd.DataFrame(body["hourly"])
            h.index = pd.to_datetime(h.pop("time")).dt.tz_localize("UTC")
            return m, h
    return None, None


rows = []
for cfg in cfgs:
    rec = sorted((cfg.data_dir / "predictions").glob("forecast_*.csv"))[-1]
    pred = pd.read_csv(rec)
    obs = pd.read_parquet(cfg.data_dir / "clean" / "observations.parquet", columns=["station_latitude", "station_longitude"])
    lat, lon = float(obs.station_latitude.iloc[0]), float(obs.station_longitude.iloc[0])
    for _, s in pred[pred.status == "ok"].iterrows():
        n = int(s.lead_group[-1])
        ws, we = pd.Timestamp(s.window_start_utc), pd.Timestamp(s.window_end_utc)
        hours = pd.date_range(ws + pd.Timedelta(hours=1), we, freq="h")
        for col in FEATURE_SETS[s.feature_set_used]["cols"]:
            r = {"location": cfg.location_name, "record": rec.name, "gauge_day": s.label_date_local, "lead": s.lead_group,
                 "model_used": s.feature_set_used, "input": col, "input_mm": s.get(f"input_{col}_mm")}
            if col.startswith("sr_"):
                r.update(status="VERIFIED", how="Single Runs request with explicit run= time",
                         detail=next((x.strip() for x in str(s.source_detail).split(";") if x.strip().startswith(col)), ""))
            else:
                model = col[3:]
                meta, h = live_response(model, lat, str(s.retrieved_at_utc))
                if h is None:
                    r.update(status="UNVERIFIED", how="live response not found in cache", detail="")
                    rows.append(r)
                    continue
                rule = forecasts.inferred_run_time(pd.Series(hours, index=hours), model, n)
                tested = matched = 0
                missing = set()
                for t in hours:
                    sr = single_run(model, lat, lon, rule[t])
                    if sr is None or t not in sr.index or pd.isna(sr[t]):
                        missing.add(rule[t])
                        continue
                    tested += 1
                    matched += bool(np.isclose(h.loc[t, f"precipitation_previous_day{n}"], sr[t], atol=TOL))
                runs = sorted({f"{x:%d %b %HZ}" for x in rule})
                latest = last_run_init(model) if missing else None
                unpublished = sorted(x for x in missing if latest is not None and x > latest)
                if tested and matched < tested:
                    st, why = "FAILED", ""
                elif unpublished:
                    # the named run is still not published now, after the forecast was made, so the API must have
                    # served values from an older run: the input does not have the trained lead time
                    st = "FAILED"
                    why = (f"; named run {', '.join(f'{x:%d %b %HZ}' for x in unpublished)} was not yet published "
                           f"(model's latest run {latest:%d %b %HZ} when checked)")
                elif missing:
                    st, why = "UNVERIFIED", ""
                else:
                    st, why = "VERIFIED", ""
                r.update(status=st, how="hourly values vs the rule-named Single Runs archive",
                         detail=f"runs {', '.join(runs)}; {matched}/{tested} hours equal"
                                + (f"; not archived: {', '.join(f'{x:%d %b %HZ}' for x in sorted(set(missing) - set(unpublished)))}"
                                   if set(missing) - set(unpublished) else "") + why)
            rows.append(r)

V = pd.DataFrame(rows)
V.to_csv(ROOT / "reports" / "served_row_verification.csv", index=False)
row_status = (V.groupby(["location", "gauge_day", "lead", "model_used"]).status
              .agg(lambda x: "FAILED" if (x == "FAILED").any() else "UNVERIFIED" if (x == "UNVERIFIED").any() else "VERIFIED")
              .rename("row_status").reset_index())
row_status.to_csv(ROOT / "reports" / "served_row_status.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 110)
print(V[["location", "gauge_day", "lead", "input", "status", "detail"]].to_string(index=False))
print()
print(row_status.to_string(index=False))
