"""Challenger forecast sources (Open-Meteo public APIs) and window aggregation.

Everything here mirrors the champion's own rules (perthrain.build / perthrain.forecasts) without modifying them:
same 9am-9am windows, same "all 24 hours or nothing" totals, same previous_dayN offset semantics and the same
as-of rule (latest contributing run + latency <= cutoff).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from perthrain.build import _hourly_stats, assign_windows, single_run_issue_time
from perthrain.forecasts import PREVIOUS_RUNS_URL, SINGLE_RUNS_URL, full_url
from perthrain.http import Fetcher

ROOT = Path(__file__).resolve().parents[1]
UA = "perthrain-dataset/0.1 (non-commercial forecast verification research)"

# Native precipitation step (h) from Open-Meteo model metadata (temporal_resolution_seconds), probed 2026-10-01.
NATIVE_STEP = {"icon_global": 1, "ecmwf_aifs025_single": 6, "ecmwf_ifs": 1, "ecmwf_ifs025": 3, "jma_gsm": 6,
               "ncep_gfs_global": 1}
CHALLENGERS = {
    "icon_global": {"label": "DWD ICON Global", "metadata_id": "dwd_icon"},
    "ecmwf_aifs025_single": {"label": "ECMWF AIFS (AI model)", "metadata_id": "ecmwf_aifs025_single"},
}
LATENCY_H = 6.0          # same conservative publication latency as the champion's rules
_FETCHER: Fetcher | None = None


def fetcher() -> Fetcher:
    global _FETCHER
    if _FETCHER is None:
        _FETCHER = Fetcher(ROOT / "data" / "raw", rate_per_second=2.0, user_agent=UA)
    return _FETCHER


def rule_run_time(valid_end, step: int, lead_day: int):
    """Run that produced previous_day{N} at this valid hour: floor_6h(ceil_to_native_step(t)) - 24 N h."""
    t = pd.Timestamp(valid_end)
    floor = t.floor(f"{step}h")
    block_end = floor if floor == t else floor + pd.Timedelta(hours=step)
    return block_end.floor("6h") - pd.Timedelta(hours=24 * lead_day)


def rule_run_series(valid_end: pd.Series, step: int, lead_day: int) -> pd.Series:
    floor = valid_end.dt.floor(f"{step}h")
    block_end = floor.where(floor == valid_end, floor + pd.Timedelta(hours=step))
    return block_end.dt.floor("6h") - pd.Timedelta(hours=24 * lead_day)


def fetch_previous_runs(model, lat, lon, start: pd.Timestamp, end: pd.Timestamp, leads, refresh=False,
                        subdir="open_meteo/challenge/previous_runs") -> pd.DataFrame:
    """Hourly previous_dayN values (quarterly chunks, cached). Nulls are kept as NaN (missing, never zero)."""
    frames = []
    cur = pd.Timestamp(start).normalize()
    while cur <= end:
        q_end = min(pd.Timestamp(end), (cur + pd.offsets.QuarterEnd(0)).normalize())
        params = {"latitude": lat, "longitude": lon, "models": model, "timezone": "GMT",
                  "hourly": ",".join(f"precipitation_previous_day{n}" for n in leads),
                  "start_date": f"{cur:%Y-%m-%d}", "end_date": f"{q_end:%Y-%m-%d}", "precipitation_unit": "mm"}
        res = fetcher().get_json(PREVIOUS_RUNS_URL, params, f"{subdir}/{model}", label=f"challenge pr {model}",
                                 refresh=refresh)
        if res.ok:
            b = res.body
            t = pd.to_datetime(b["hourly"]["time"]).tz_localize("UTC")
            for n in leads:
                v = f"precipitation_previous_day{n}"
                if b.get("hourly_units", {}).get(v) != "mm":
                    raise ValueError(f"{model} {v}: unit {b.get('hourly_units', {}).get(v)!r}")
                frames.append(pd.DataFrame({
                    "model": model, "lead_day": n, "valid_end_utc": t,
                    "precip_mm": pd.to_numeric(pd.Series(b["hourly"][v]), errors="coerce"),
                    "grid_latitude": b.get("latitude"), "grid_longitude": b.get("longitude"),
                    "source_url": full_url(PREVIOUS_RUNS_URL, params), "retrieved_at_utc": res.retrieved_at_utc}))
        cur = q_end + pd.Timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=["model", "lead_day", "valid_end_utc", "precip_mm"])
    out = pd.concat(frames, ignore_index=True)
    out["issue_time_inferred_utc"] = np.nan
    for n in leads:
        m = out.lead_day == n
        out.loc[m, "issue_time_inferred_utc"] = rule_run_series(out.loc[m, "valid_end_utc"], NATIVE_STEP.get(model, 1), n)
    out["issue_time_inferred_utc"] = pd.to_datetime(out.issue_time_inferred_utc, utc=True)
    return out


def fetch_single_run(model, lat, lon, run: pd.Timestamp, days=5, refresh=False,
                     subdir="open_meteo/challenge/single_runs") -> pd.DataFrame | None:
    params = {"latitude": lat, "longitude": lon, "models": model, "hourly": "precipitation",
              "run": pd.Timestamp(run).strftime("%Y-%m-%dT%H:%M"), "forecast_days": days, "timezone": "GMT",
              "precipitation_unit": "mm"}
    res = fetcher().get_json(SINGLE_RUNS_URL, params, f"{subdir}/{model}", label=f"challenge sr {model} {run}",
                             refresh=refresh)
    if not res.ok:
        return None
    b = res.body
    if b.get("hourly_units", {}).get("precipitation") != "mm":
        raise ValueError(f"{model} run {run}: unit {b.get('hourly_units', {}).get('precipitation')!r}")
    return pd.DataFrame({"model": model, "valid_end_utc": pd.to_datetime(b["hourly"]["time"]).tz_localize("UTC"),
                         "precip_mm": pd.to_numeric(pd.Series(b["hourly"]["precipitation"]), errors="coerce"),
                         "forecast_issue_time_utc": pd.Timestamp(run).tz_convert("UTC") if pd.Timestamp(run).tzinfo
                         else pd.Timestamp(run, tz="UTC"),
                         "grid_latitude": b.get("latitude"), "grid_longitude": b.get("longitude"),
                         "source_url": full_url(SINGLE_RUNS_URL, params), "retrieved_at_utc": res.retrieved_at_utc})


def window_totals_previous_runs(hourly: pd.DataFrame, windows: pd.DataFrame, cutoff_rule="window_start",
                                as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per window x lead: 24-hour total (NaN unless all 24 hours present and none negative), the latest
    contributing run and whether it was published (run + latency) before the cutoff (window start, or as_of)."""
    h = hourly.copy()
    h["w"] = assign_windows(h.valid_end_utc, windows)
    h = h[h.w >= 0]
    h["source_url"] = h.get("source_url", "")
    stats = _hourly_stats(h, ["model", "lead_day", "w"])
    iss = h.groupby(["model", "lead_day", "w"]).issue_time_inferred_utc.max().rename("latest_run_utc").reset_index()
    out = stats.merge(iss, on=["model", "lead_day", "w"], how="left").join(windows, on="w")
    return _finish(out, cutoff_rule, as_of)


def window_totals_single_run(sr: pd.DataFrame, windows: pd.DataFrame, lead_day: int) -> pd.DataFrame:
    h = sr.copy()
    h["w"] = assign_windows(h.valid_end_utc, windows)
    h = h[h.w >= 0]
    stats = _hourly_stats(h, ["model", "w"])
    stats["lead_day"] = lead_day
    stats["latest_run_utc"] = sr.forecast_issue_time_utc.iloc[0]
    return _finish(stats.join(windows, on="w"), "window_start", None)


def _finish(out, cutoff_rule, as_of):
    expected = ((out.window_end_utc - out.window_start_utc) / pd.Timedelta(hours=1)).astype(int)
    out["complete"] = (out.available_hour_count == expected) & (out.hour_rows == expected)
    out["total_mm"] = np.where(out.complete & (out.negative_hours == 0), out.precip_sum.round(3), np.nan)
    cutoff = out.window_start_utc if cutoff_rule == "window_start" else pd.Timestamp(as_of)
    out["published_before_cutoff"] = (out.latest_run_utc + pd.Timedelta(hours=LATENCY_H)) <= cutoff
    out["lead_hours_to_window_start"] = (out.window_start_utc - out.latest_run_utc) / pd.Timedelta(hours=1)
    return out


def sr_issue_time(window_start: pd.Series, lead_day: int) -> pd.Series:
    """The champion's single-run rule: latest 12 UTC run published (6 h) by window_start - 24 (N-1) h."""
    return single_run_issue_time(window_start, lead_day, 12, LATENCY_H)
