"""DIAGNOSTIC feasibility check: can forecasts be issued from explicitly identified, already-available Single Runs?

Writes reports/v2/live_run_feasibility.md and .csv. Raw responses are saved (append-only, one folder per check) under
data/raw/open_meteo/feasibility/<UTC stamp>/ as evidence. Nothing in the frozen release is read for writing.

1. Model metadata (Open-Meteo /data/<model>/static/meta.json): last run initialisation and availability times.
2. Single Runs requests for every 6-hourly run of the last ~36 h: which exist, their horizon.
3. For each upcoming 9am-9am window: complete (24 non-null hours)? total, actual lead hours.
4. Which existing calibrators share the run's training semantics (only ECMWF IFS 12 UTC single runs at the trained
   lead offsets do).
"""
import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perthrain import observations  # noqa: E402
from perthrain.forecasts import SINGLE_RUNS_URL, single_run_params  # noqa: E402

UA = {"User-Agent": "perthrain-dataset/0.1 (non-commercial forecast verification research)"}
TZ = "Australia/Perth"
LAT, LON = -31.9192, 115.8728            # 009225 PERTH METRO (availability is model-wide; one point suffices)
now = pd.Timestamp.now(tz="UTC").floor("min")
stamp = f"{now:%Y%m%dT%H%MZ}"
EV = ROOT / "data" / "raw" / "open_meteo" / "feasibility" / stamp
EV.mkdir(parents=True, exist_ok=False)
OUT = ROOT / "reports" / "v2"
OUT.mkdir(parents=True, exist_ok=True)


def get(url, params=None, name=None):
    t = pd.Timestamp.now(tz="UTC")
    r = requests.get(url, params=params, headers=UA, timeout=60)
    rec = {"url": r.url, "status": r.status_code, "retrieved_at_utc": f"{t:%Y-%m-%dT%H:%M:%SZ}"}
    (EV / f"{name}.json").write_text(r.text, encoding="utf-8")
    (EV / f"{name}.meta.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return r, rec


def ts(epoch):
    return pd.Timestamp(epoch, unit="s", tz="UTC") if epoch else pd.NaT


# ---- 1. metadata
meta_rows = []
for m in ["ecmwf_ifs", "ecmwf_ifs025", "jma_gsm", "ncep_gfs025", "ncep_gfs013"]:
    r, rec = get(f"https://api.open-meteo.com/data/{m}/static/meta.json", name=f"meta_{m}")
    j = r.json() if r.ok else {}
    meta_rows.append({"model": m, "http": r.status_code,
                      "last_run_init_utc": ts(j.get("last_run_initialisation_time")),
                      "last_run_available_utc": ts(j.get("last_run_availability_time")),
                      "data_end_utc": ts(j.get("data_end_time")),
                      "update_interval_h": (j.get("update_interval_seconds") or 0) / 3600,
                      "retrieved_at_utc": rec["retrieved_at_utc"]})
M = pd.DataFrame(meta_rows)
M["latency_h"] = (M.last_run_available_utc - M.last_run_init_utc) / pd.Timedelta(hours=1)

# ---- 2/3. single runs
labels = [d for d in pd.date_range((now.tz_convert(TZ)).date(), periods=9, freq="D").date]
wins = []
for d in labels:
    s, e = observations.obs_window(d, TZ)
    if s > now:
        wins.append((d, s, e))
runs = pd.date_range(now.floor("6h") - pd.Timedelta(hours=36), now.floor("6h"), freq="6h")
run_rows, win_rows = [], []
for model in ["ecmwf_ifs", "ecmwf_ifs025", "jma_gsm", "ncep_gfs_global"]:
    for run in runs:
        p = single_run_params(LAT, LON, model, run, 8)
        r, rec = get(SINGLE_RUNS_URL, p, name=f"sr_{model}_{run:%Y%m%dT%H}")
        row = {"model": model, "run_init_utc": run, "http": r.status_code, "retrieved_at_utc": rec["retrieved_at_utc"],
               "request": rec["url"]}
        if not r.ok:
            row["note"] = r.text[:120]
            run_rows.append(row)
            continue
        h = r.json()["hourly"]
        s = pd.Series(h["precipitation"], index=pd.to_datetime(h["time"]).tz_localize("UTC"), dtype="float64")
        nn = s.dropna()
        row.update(non_null_hours=len(nn), first_valid_utc=nn.index.min(), last_valid_utc=nn.index.max(),
                   horizon_h=(nn.index.max() - run) / pd.Timedelta(hours=1) if len(nn) else None)
        run_rows.append(row)
        for d, ws, we in wins:
            hrs = pd.date_range(ws + pd.Timedelta(hours=1), we, freq="h")
            vals = s.reindex(hrs)
            complete = bool(vals.notna().all())
            lead_s = (ws - run) / pd.Timedelta(hours=1)
            # training semantics of the existing sr calibrators: 12 UTC run, window starts 13 + 24 (N-1) h after it
            n = (lead_s - 13) / 24 + 1
            group = f"day{int(n)}" if (run.hour == 12 and model == "ecmwf_ifs" and n == int(n) and n >= 1) else ""
            compat = ("existing ecmwf_ifs_sr calibrator" if group in ("day1", "day2", "day3")
                      else "no calibrator yet (same semantics; trainable from archive)" if group
                      else "none: no calibrator trained on this model/run hour/lead")
            win_rows.append({"model": model, "run_init_utc": run, "gauge_day": d,
                             "window_local": f"{ws.tz_convert(TZ):%a %d %b %H:%M} -> {we.tz_convert(TZ):%a %d %b %H:%M}",
                             "complete_24h": complete, "total_mm": round(float(vals.sum()), 2) if complete else None,
                             "lead_h_start": lead_s, "lead_h_end": (we - run) / pd.Timedelta(hours=1),
                             "published_by_now": None, "lead_group_if_trained": group, "compatible_calibrator": compat})
R = pd.DataFrame(run_rows)
W = pd.DataFrame(win_rows)
# a run counts as available only if it answered now; metadata availability time is the evidence of when
R.to_csv(OUT / "live_run_feasibility_runs.csv", index=False)
W.to_csv(OUT / "live_run_feasibility_windows.csv", index=False)
M.to_csv(OUT / "live_run_feasibility_meta.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 80)
print("NOW", now)
print(M.to_string(index=False))
print(R.drop(columns=["request"]).to_string(index=False))
ok = W[W.complete_24h]
print(ok[ok.model == "ecmwf_ifs"].to_string(index=False))
print(ok.groupby(["model", "run_init_utc"]).gauge_day.agg(["min", "max", "count"]).to_string())
print("evidence:", EV)
