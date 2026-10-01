"""Small availability probes for challenger models (Open-Meteo public APIs). Writes
reports/challenge/source_availability.csv and saves every raw response under
data/raw/open_meteo/challenge_probe/<UTC stamp>/ as evidence. Read-only with respect to the champion.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "perthrain-dataset/0.1 (non-commercial forecast verification research)"}
LAT, LON = -31.9192, 115.8728          # 009225 PERTH METRO gauge
PR = "https://previous-runs-api.open-meteo.com/v1/forecast"
SR = "https://single-runs-api.open-meteo.com/v1/forecast"
EN = "https://ensemble-api.open-meteo.com/v1/ensemble"
NOW = pd.Timestamp.now(tz="UTC").floor("min")
EV = ROOT / "data" / "raw" / "open_meteo" / "challenge_probe" / f"{NOW:%Y%m%dT%H%MZ}"
OUT = ROOT / "reports" / "challenge"

DETERMINISTIC = {   # Open-Meteo API id: (metadata id, role)
    "ecmwf_ifs": ("ecmwf_ifs", "existing benchmark (single runs)"),
    "ecmwf_ifs025": ("ecmwf_ifs025", "existing benchmark"),
    "jma_gsm": ("jma_gsm", "existing benchmark"),
    "ncep_gfs_global": ("ncep_gfs025", "existing benchmark"),
    "bom_access_global": ("bom_access_global", "candidate"),
    "icon_global": ("dwd_icon", "candidate"),
    "gem_global": ("cmc_gem_gdps", "candidate"),
    "ecmwf_aifs025_single": ("ecmwf_aifs025_single", "candidate"),
}
ENSEMBLE = {"ecmwf_ifs025": "ECMWF IFS ENS 0.25 (51 members)", "gfs025": "NOAA GEFS 0.25 (31 members)"}
_n = 0


def get(url, params, tag):
    global _n
    _n += 1
    t = pd.Timestamp.now(tz="UTC")
    try:
        r = requests.get(url, params=params, headers=UA, timeout=120)
        status, text = r.status_code, r.text
    except requests.RequestException as e:
        status, text = None, str(e)
    (EV / f"{_n:03d}_{tag}.json").write_text(text, encoding="utf-8")
    (EV / f"{_n:03d}_{tag}.meta.json").write_text(json.dumps(
        {"url": url, "params": params, "status": status, "retrieved_at_utc": f"{t:%Y-%m-%dT%H:%M:%SZ}"}, indent=1))
    try:
        body = json.loads(text) if status == 200 else None
    except ValueError:
        body = None
    return status, body, text[:140]


def nonnull(body, var):
    if not body or "hourly" not in body or var not in body["hourly"]:
        return pd.Series(dtype=float)
    s = pd.Series(body["hourly"][var], index=pd.to_datetime(body["hourly"]["time"]), dtype="float64")
    return s.dropna()


def previous_runs_coverage(model):
    first = last = None
    nvals = 0
    for y in range(2016, NOW.year + 1):
        end = min(pd.Timestamp(f"{y}-12-31"), NOW.tz_localize(None).normalize() - pd.Timedelta(days=1))
        st, b, _ = get(PR, {"latitude": LAT, "longitude": LON, "models": model, "timezone": "GMT",
                            "hourly": "precipitation_previous_day2", "start_date": f"{y}-01-01",
                            "end_date": f"{end:%Y-%m-%d}"}, f"pr_{model}_{y}")
        s = nonnull(b, "precipitation_previous_day2")
        if len(s):
            first = first or s.index.min()
            last = s.index.max()
            nvals += len(s)
    return first, last, nvals


def single_run_ok(model, day):
    run = pd.Timestamp(day).normalize() + pd.Timedelta(hours=12)
    st, b, txt = get(SR, {"latitude": LAT, "longitude": LON, "models": model, "hourly": "precipitation",
                          "run": f"{run:%Y-%m-%dT%H:%M}", "forecast_days": 4, "timezone": "GMT"},
                     f"sr_{model}_{run:%Y%m%dT%H}")
    return st == 200 and len(nonnull(b, "precipitation")) > 0, b, txt


def single_runs_start(model):
    lo, hi = pd.Timestamp("2024-01-01"), (NOW - pd.Timedelta(days=2)).tz_localize(None).normalize()
    ok_hi, _, txt = single_run_ok(model, hi)
    if not ok_hi:
        return None, txt
    if single_run_ok(model, lo)[0]:
        return lo, "available at 2024-01-01 (earlier not probed)"
    while (hi - lo).days > 1:                 # first available 12Z run, assuming no later gaps (spot-checked)
        mid = lo + (hi - lo) / 2
        mid = mid.normalize()
        if single_run_ok(model, mid)[0]:
            hi = mid
        else:
            lo = mid
    return hi, ""


def main():
    EV.mkdir(parents=True, exist_ok=False)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, (meta_id, role) in DETERMINISTIC.items():
        st, meta, _ = get(f"https://api.open-meteo.com/data/{meta_id}/static/meta.json", {}, f"meta_{meta_id}")
        meta = meta or {}
        first, last, n = previous_runs_coverage(model)
        sr_start, sr_note = single_runs_start(model)
        _, b, _ = single_run_ok(model, (NOW - pd.Timedelta(days=2)).tz_localize(None))
        ts = lambda k: pd.Timestamp(meta[k], unit="s", tz="UTC") if meta.get(k) else pd.NaT  # noqa: E731
        rows.append({
            "model_id": model, "metadata_id": meta_id, "role": role,
            "grid_lat": (b or {}).get("latitude"), "grid_lon": (b or {}).get("longitude"),
            "rain_variable": "precipitation (mm, preceding hour; previous_dayN = value predicted 24*N h earlier)",
            "native_step_h": (meta.get("temporal_resolution_seconds") or 0) / 3600 or None,
            "last_run_init_utc": ts("last_run_initialisation_time"),
            "last_run_available_utc": ts("last_run_availability_time"),
            "availability_delay_h": round((ts("last_run_availability_time") - ts("last_run_initialisation_time"))
                                          / pd.Timedelta(hours=1), 1) if meta else None,
            "previous_runs_first_utc": first, "previous_runs_last_utc": last, "previous_runs_hours": n,
            "single_runs_first_12z": sr_start, "single_runs_note": sr_note})
    for model, label in ENSEMBLE.items():
        st, b, txt = get(EN, {"latitude": LAT, "longitude": LON, "models": model, "hourly": "precipitation",
                              "forecast_days": 4, "timezone": "GMT"}, f"ens_{model}")
        members = [k for k in (b or {}).get("hourly", {}) if k.startswith("precipitation")]
        st2, b2, txt2 = get(EN, {"latitude": LAT, "longitude": LON, "models": model, "hourly": "precipitation",
                                 "start_date": "2026-06-01", "end_date": "2026-06-02", "timezone": "GMT"},
                            f"ens_{model}_past")
        st3, b3, txt3 = get(SR, {"latitude": LAT, "longitude": LON, "models": f"{model}_ensemble" if "ecmwf" in model
                                 else "ncep_gefs025", "hourly": "precipitation", "run": "2026-09-29T12:00",
                                 "forecast_days": 4, "timezone": "GMT"}, f"ens_sr_{model}")
        rows.append({"model_id": f"ensemble:{model}", "metadata_id": "", "role": "ensemble candidate",
                     "grid_lat": (b or {}).get("latitude"), "grid_lon": (b or {}).get("longitude"),
                     "rain_variable": f"{len(members)} member columns (hourly mm)" if members else txt,
                     "single_runs_note": f"ensemble API past window: HTTP {st2} "
                                         f"({len(nonnull(b2, 'precipitation'))} control hours); single-runs API: "
                                         f"HTTP {st3} {'' if st3 == 200 else txt3}",
                     "previous_runs_hours": 0})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "source_availability.csv", index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 90)
    print(df.drop(columns=["rain_variable"]).to_string(index=False))
    print("evidence:", EV.relative_to(ROOT), "requests:", _n)


if __name__ == "__main__":
    sys.exit(main())
