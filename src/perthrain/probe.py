"""Small overlapping test window: confirm units, timestamp semantics and window alignment
before the full download. Writes reports/probe_verification.json."""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import build, forecasts, observations, stations
from .config import Config
from .pipeline import fetch_station_lists, ftp_month_url, make_fetcher

log = logging.getLogger("perthrain.probe")


def native_step_check(hourly: pd.DataFrame) -> dict:
    """Share of rainy 3-h blocks (00-03Z, 03-06Z, ...) whose three hourly values are identical.

    ~1.0 means hourly values are a uniform split of 3-hourly model output."""
    out = {}
    for (m, n), g in hourly.groupby(["model", "lead_day"]):
        s = g.set_index("valid_end_utc").precip_mm.sort_index()
        block = ((s.index - pd.Timedelta(hours=1)).floor("3h"))
        b = s.groupby(block).agg(["min", "max", "sum", "count"])
        b = b[(b["count"] == 3) & (b["sum"] > 0.3)]
        out[f"{m}/day{n}"] = {"rainy_blocks": int(len(b)),
                              "share_identical": float((b["min"] == b["max"]).mean()) if len(b) else None}
    return out


def run_probe(cfg: Config, start: date = date(2026, 6, 1), end: date = date(2026, 6, 14)) -> dict:
    f = make_fetcher(cfg)
    result: dict = {"window": [str(start), str(end)]}
    db, _ = fetch_station_lists(cfg, f)
    near = stations.nearby(db, cfg.latitude, cfg.longitude, cfg.search_radius_km)
    st = near[near.station_id == cfg.station_id].iloc[0] if cfg.station_id else near.iloc[0]
    st = st.to_dict()
    result["probe_station"] = {k: str(v) for k, v in st.items()}

    # observations
    state = stations.STATE_FOLDERS[st["state"]]
    folder = stations.ftp_folder_name(st["station_name"])
    monthly = []
    for ym in sorted({start.strftime("%Y%m"), end.strftime("%Y%m")}):
        url = ftp_month_url(state, folder, ym)
        res = f.get_file(url, cfg.raw_dir / "bom_ftp" / state / folder / f"{folder}-{ym}.csv", label="probe obs")
        monthly.append((url, res.body if res.ok else None, res.retrieved_at_utc, ym))
        if res.ok:
            _, meta = observations.parse_idckwcdea0(res.body)
            result["observation_header_check"] = {k: str(v) for k, v in meta.items()}
    obs, _ = observations.build_ftp_observations(monthly, st, cfg.timezone, start, end, cfg.suspicious_daily_mm)
    result["observation_window_example"] = {
        "label_date_local": str(obs.label_date_local.iloc[0]),
        "window_start_local": str(obs.window_start_local.iloc[0]), "window_end_local": str(obs.window_end_local.iloc[0]),
        "window_start_utc": str(obs.window_start_utc.iloc[0]), "window_end_utc": str(obs.window_end_utc.iloc[0])}

    # previous runs
    lat, lon = st["latitude"], st["longitude"]
    pr, fails = forecasts.download_previous_runs(f, lat, lon, cfg.previous_runs_models, start - timedelta(days=1),
                                                 end, cfg.lead_days, 1)
    result["previous_runs_failures"] = fails
    result["previous_runs_units"] = sorted(pr.unit.unique().tolist())
    result["native_step_check"] = native_step_check(pr)

    # single-run semantics + previous_day1 run mapping on the wettest day
    wet = obs.dropna(subset=["observed_precip_mm"]).sort_values("observed_precip_mm").iloc[-1]
    valid_day = pd.Timestamp(wet.window_start_utc).floor("D")
    result["mapping_test_valid_day_utc"] = str(valid_day.date())
    mapping = {}
    for model in ["ncep_gfs_global", "ecmwf_ifs025", "jma_gsm"]:
        runs = [valid_day - pd.Timedelta(days=1) + pd.Timedelta(hours=h) for h in (0, 6, 12, 18, 24)]
        runs_data = {}
        first_hour_null = []
        for r in runs:
            p = forecasts.single_run_params(lat, lon, model, r, 3)
            res = f.get_json(forecasts.SINGLE_RUNS_URL, p, f"open_meteo/single_runs/{model}", label=f"probe {model} {r}")
            if res.ok:
                d = forecasts.parse_single_run(res.body, model, r, res.url, res.params, res.retrieved_at_utc)
                runs_data[r] = d.set_index("valid_end_utc").precip_mm
                first_hour_null.append(bool(pd.isna(d.precip_mm.iloc[0])) and d.valid_end_utc.iloc[0] == r)
        prd = pr[(pr.model == model) & (pr.lead_day == 1)].set_index("valid_end_utc")
        hours = pd.date_range(valid_day + pd.Timedelta(hours=1), valid_day + pd.Timedelta(hours=24), freq="h")
        match, tested, wet_hours = 0, 0, 0
        for t in hours:
            run = forecasts.inferred_run_time(pd.Series([t]), model, 1).iloc[0]
            if run in runs_data and t in runs_data[run].index and t in prd.index \
                    and pd.notna(runs_data[run].loc[t]) and pd.notna(prd.loc[t, "precip_mm"]):
                a, b = prd.loc[t, "precip_mm"], runs_data[run].loc[t]
                tested += 1
                wet_hours += int(b > 0)
                match += int(np.isclose(a, b, atol=0.051))
        mapping[model] = {"hours_tested": tested, "hours_with_rain_in_run": wet_hours,
                          "previous_day1_equals_inferred_run_floor6h_ceil_step_minus_24h": match,
                          "single_run_first_timestamp_is_init_and_null": first_hour_null}
    result["previous_day1_mapping"] = mapping

    # example window sums for the test window
    windows = build.window_bins(obs)
    fw = build.finalize_forecast_windows(
        build.aggregate_previous_runs(pr, windows, cfg.lead_days, cfg.publication_latency_hours))
    ex = fw[fw.lead_day == 2].pivot_table(index="w", columns="model", values="forecast_precip_mm")
    ex = windows.join(ex).merge(obs[["window_start_utc", "observed_precip_mm"]], on="window_start_utc")
    result["example_day2_window_totals"] = json.loads(ex.drop(columns=["window_start_utc", "window_end_utc"])
                                                      .to_json(orient="records", date_format="iso"))
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "probe_verification.json").write_text(json.dumps(result, indent=1, default=str))
    return result
