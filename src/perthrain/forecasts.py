"""Open-Meteo historical forecast downloads and parsing to tidy hourly records.

Two sources, never mixed and never stitched:

* Previous Runs API (``previous_runs``): ``precipitation_previous_dayN`` = value for
  valid hour t "predicted 24*N hours before valid time". No run timestamp is
  returned. We verified (see reports/probe_verification.json) that for 6-hourly
  global models the contributing run is ``floor_6h(t) - 24*N h``; this is stored
  as an *inferred* issue time range, never as a source-provided issue time.
* Single Runs API (``single_runs``): complete output of one run selected by its UTC
  initialisation time (``run=``). Issue time is explicit.

Hourly ``precipitation`` is the accumulation over the hour ENDING at the timestamp
(verified: the initialisation hour of every single run is null).
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from urllib.parse import urlencode

import pandas as pd

from .http import Fetcher

log = logging.getLogger(__name__)

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"

# Native output interval of precipitation (hours), from Open-Meteo's model table
# ("Temporal Resolution"). Hourly values of a 3- or 6-hourly model are a uniform
# disaggregation of the native step (verified by probe: every rainy native block has
# identical hourly values).
NATIVE_STEP_HOURS = {"ecmwf_ifs": 1, "ecmwf_ifs025": 3, "bom_access_global": 1,
                     "ncep_gfs_global": 1, "jma_gsm": 6}
RUN_CADENCE_HOURS = 6


def inferred_run_time(valid_end: pd.Series, model: str, lead_day: int) -> pd.Series:
    """Latest model run that can have produced ``precipitation_previous_day{lead_day}``.

    Verified against the Single Runs API for ncep_gfs_global (1 h), ecmwf_ifs025 (3 h)
    and jma_gsm (6 h), days 1 and 2: 100 % of comparable hours match

        run = floor_6h( ceil_to_native_step(valid_end) ) - 24 * lead_day

    i.e. the offset is applied to the END of the native accumulation step that the
    hourly value was disaggregated from, not to the hourly label itself.
    """
    step = NATIVE_STEP_HOURS.get(model, 1)
    floor = valid_end.dt.floor(f"{step}h")
    block_end = floor.where(floor == valid_end, floor + pd.Timedelta(hours=step))
    return block_end.dt.floor(f"{RUN_CADENCE_HOURS}h") - pd.Timedelta(hours=24 * lead_day)


def full_url(url: str, params: dict) -> str:
    return f"{url}?{urlencode(params)}"


def quarter_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks, cur = [], start
    while cur <= end:
        q_end_month = ((cur.month - 1) // 3 + 1) * 3
        nxt = date(cur.year + (q_end_month == 12), (q_end_month % 12) + 1, 1)
        chunks.append((cur, min(end, nxt - timedelta(days=1))))
        cur = nxt
    return chunks


def previous_runs_params(lat, lon, model, start, end, lead_days) -> dict:
    return {"latitude": lat, "longitude": lon, "models": model,
            "hourly": ",".join(f"precipitation_previous_day{n}" for n in lead_days),
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "timezone": "GMT", "precipitation_unit": "mm"}


def single_run_params(lat, lon, model, run: pd.Timestamp, forecast_days: int) -> dict:
    return {"latitude": lat, "longitude": lon, "models": model, "hourly": "precipitation",
            "run": run.strftime("%Y-%m-%dT%H:%M"), "forecast_days": forecast_days,
            "timezone": "GMT", "precipitation_unit": "mm"}


def parse_previous_runs(body: dict, model: str, lead_days, url: str, params: dict,
                        retrieved_at: str) -> pd.DataFrame:
    units = body.get("hourly_units", {})
    h = body["hourly"]
    frames = []
    for n in lead_days:
        var = f"precipitation_previous_day{n}"
        if units.get(var) != "mm":
            raise ValueError(f"{model} {var}: unexpected unit {units.get(var)!r}")
        valid_end = pd.to_datetime(h["time"]).tz_localize("UTC")
        issue = inferred_run_time(pd.Series(valid_end), model, n)
        frames.append(pd.DataFrame({
            "source_api": "previous_runs", "model": model, "lead_day": n,
            "source_lead_offset": var.replace("precipitation_", ""),
            "valid_end_utc": valid_end, "precip_mm": pd.to_numeric(pd.Series(h[var]), errors="coerce"),
            "issue_time_inferred_utc": issue, "forecast_issue_time_utc": pd.NaT,
            "grid_latitude": body.get("latitude"), "grid_longitude": body.get("longitude"),
            "grid_elevation": body.get("elevation"), "unit": units.get(var),
            "source_url": full_url(url, params), "retrieved_at_utc": retrieved_at}))
    return pd.concat(frames, ignore_index=True)


def parse_single_run(body: dict, model: str, run: pd.Timestamp, url: str, params: dict,
                     retrieved_at: str) -> pd.DataFrame:
    unit = body.get("hourly_units", {}).get("precipitation")
    if unit != "mm":
        raise ValueError(f"{model} run {run}: unexpected unit {unit!r}")
    h = body["hourly"]
    valid_end = pd.to_datetime(h["time"]).tz_localize("UTC")
    return pd.DataFrame({
        "source_api": "single_runs", "model": model, "valid_end_utc": valid_end,
        "precip_mm": pd.to_numeric(pd.Series(h["precipitation"]), errors="coerce"),
        "forecast_issue_time_utc": run, "grid_latitude": body.get("latitude"),
        "grid_longitude": body.get("longitude"), "grid_elevation": body.get("elevation"),
        "unit": unit, "source_url": full_url(url, params), "retrieved_at_utc": retrieved_at})


def download_previous_runs(fetcher: Fetcher, lat, lon, models, start: date, end: date, lead_days,
                           concurrency: int = 2):
    jobs = [(m, a, b) for m in models for a, b in quarter_chunks(start, end)]

    def run(job):
        m, a, b = job
        p = previous_runs_params(lat, lon, m, a, b, lead_days)
        return m, fetcher.get_json(PREVIOUS_RUNS_URL, p, f"open_meteo/previous_runs/{m}",
                                   label=f"previous_runs {m} {a}..{b}")

    frames, failures = [], []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for m, res in ex.map(run, jobs):
            if res.ok:
                frames.append(parse_previous_runs(res.body, m, lead_days, res.url, res.params,
                                                  res.retrieved_at_utc))
            else:
                failures.append({"stage": "previous_runs", "item": m, "url": full_url(res.url, res.params),
                                 "problem": res.error})
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return df, failures


def download_single_runs(fetcher: Fetcher, lat, lon, model, runs: list[pd.Timestamp],
                         forecast_days: int, concurrency: int = 2, progress_every: int = 100):
    def run(r):
        p = single_run_params(lat, lon, model, r, forecast_days)
        return r, fetcher.get_json(SINGLE_RUNS_URL, p, f"open_meteo/single_runs/{model}",
                                   label=f"single_runs {model} {r:%Y-%m-%dT%H}")

    frames, failures = [], []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for i, (r, res) in enumerate(ex.map(run, runs), 1):
            if res.ok:
                frames.append(parse_single_run(res.body, model, r, res.url, res.params, res.retrieved_at_utc))
            else:
                reason = res.error
                if res.status == 400 and res.cache_path.exists():
                    try:
                        reason = json.loads(res.cache_path.read_text()).get("reason", reason)
                    except Exception:
                        pass
                failures.append({"stage": "single_runs", "item": f"{model} {r:%Y-%m-%dT%H:%M}",
                                 "url": full_url(res.url, res.params), "problem": reason})
            if i % progress_every == 0:
                log.info("single_runs %s: %d/%d runs processed", model, i, len(runs))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return df, failures
