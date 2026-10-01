"""End-to-end orchestration: stations -> observations -> forecasts -> join -> reports."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from . import baseline, build, forecasts, observations, stations
from .config import Config
from .http import Fetcher, utcnow_iso

log = logging.getLogger("perthrain")


# --------------------------------------------------------------------------- utils
def setup_logging(cfg: Config) -> None:
    Path("logs").mkdir(exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, force=True, handlers=[
        logging.StreamHandler(), logging.FileHandler("logs/pipeline.log", encoding="utf-8")])


def make_fetcher(cfg: Config) -> Fetcher:
    return Fetcher(cfg.raw_dir, rate_per_second=cfg.rate_per_second, connect_timeout=cfg.connect_timeout_s,
                   read_timeout=cfg.read_timeout_s, max_retries=cfg.max_retries, user_agent=cfg.user_agent)


def write_both(df: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    out = df.copy()
    for c in out.columns:  # parquet needs homogeneous object columns
        if out[c].dtype == object:
            sample = out[c].dropna()
            if len(sample) and not all(isinstance(v, type(sample.iloc[0])) for v in sample):
                out[c] = out[c].astype("string")
    out.to_parquet(stem.with_suffix(".parquet"), index=False)
    log.info("wrote %s (.csv/.parquet, %d rows)", stem, len(df))


def month_list(first: date, last: date) -> list[str]:
    return [p.strftime("%Y%m") for p in pd.period_range(first, last, freq="M")]


def ftp_month_url(state_folder: str, folder: str, ym: str) -> str:
    return f"{stations.FTP_BASE}/{state_folder}/{folder}/{folder}-{ym}.csv"


def today_local(cfg: Config) -> date:
    return datetime.now(ZoneInfo(cfg.timezone)).date()


def resolve_period(cfg: Config) -> tuple[date, date]:
    """Provisional FORECAST label-date range; the end is trimmed to the last reported day."""
    end = today_local(cfg) if cfg.end == "auto" else date.fromisoformat(cfg.end)
    end = min(end, today_local(cfg))  # never future verification dates
    if cfg.start == "auto":
        start = max(date.fromisoformat(cfg.archive_floor),
                    end.replace(year=end.year - cfg.max_years) + timedelta(days=1))
    else:
        start = date.fromisoformat(cfg.start)
    return start, end


def resolve_obs_period(cfg: Config, forecast_first: date, last: date) -> tuple[date, date]:
    """Observations are downloaded back to observations_start, which is normally earlier
    than any forecast archive, so the clean observation series outlives the paired data."""
    if cfg.start != "auto":
        return forecast_first, last
    return min(forecast_first, date.fromisoformat(cfg.observations_start)), last


# ------------------------------------------------------------------------ stations
def fetch_station_lists(cfg: Config, f: Fetcher):
    db = f.get_file(stations.STATIONS_DB_URL, cfg.raw_dir / "bom_ftp" / "stations_db.txt",
                    refresh=True, label="IDCKWCDEA0 stations_db")
    allz = f.get_file(stations.ALL_STATIONS_URL, cfg.raw_dir / "bom_ftp" / "stations_IDCJMC0014.zip",
                      refresh=True, label="IDCJMC0014 stations.zip")
    if not db.ok:
        raise SystemExit(f"cannot download station list: {db.error}")
    db_df = stations.parse_stations_db(db.body.decode("latin-1"))
    all_df = stations.parse_all_stations_zip(allz.body) if allz.ok else pd.DataFrame()
    return db_df, all_df


def download_station_months(cfg: Config, f: Fetcher, st: dict, months: list[str], folder_names: set[str]):
    state = stations.STATE_FOLDERS.get(st["state"], st["state"].lower())
    folder = stations.ftp_folder_name(st["station_name"])
    if folder not in folder_names:
        return None
    recent = set(months[-2:])  # current/previous month files are still being updated
    out = []
    for ym in months:
        url = ftp_month_url(state, folder, ym)
        res = f.get_file(url, cfg.raw_dir / "bom_ftp" / state / folder / f"{folder}-{ym}.csv",
                         refresh=ym in recent, label=f"obs {folder} {ym}")
        out.append((url, res.body if res.ok else None, res.retrieved_at_utc, ym))
    return out


def select_station(cfg: Config, f: Fetcher, first: date, last: date, score_years: int = 3):
    """Pick the primary gauge.

    Completeness is scored over the most recent ``score_years`` of the period only, so
    selecting a station does not require downloading every monthly file of every
    candidate; the chosen station is then downloaded over the full period.
    """
    db_df, all_df = fetch_station_lists(cfg, f)
    near = stations.nearby(db_df, cfg.latitude, cfg.longitude, cfg.search_radius_km)
    near_all = stations.nearby(all_df, cfg.latitude, cfg.longitude, cfg.search_radius_km) \
        if not all_df.empty else pd.DataFrame()
    score_first = max(first, last.replace(year=last.year - score_years) + timedelta(days=1))
    months = month_list(score_first, last)
    rows, monthly_cache = [], {}
    folder_lists: dict[str, set[str]] = {}
    for _, st in near.iterrows():
        state = stations.STATE_FOLDERS.get(st.state, st.state.lower())
        if state not in folder_lists:
            folder_lists[state] = set(f.list_ftp_dir(f"{stations.FTP_BASE}/{state}/"))
        monthly = download_station_months(cfg, f, st.to_dict(), months, folder_lists[state])
        rec = st.to_dict()
        rec["ftp_folder"] = stations.ftp_folder_name(st.station_name)
        rec["downloadable"] = monthly is not None
        if monthly is None:
            rec.update(completeness=0.0, id_ambiguous=False)
        else:
            obs, _ = observations.build_ftp_observations(monthly, rec, cfg.timezone, score_first, last,
                                                         cfg.suspicious_daily_mm)
            rec["completeness"] = float(obs.observed_precip_mm.notna().mean())
            monthly_cache[st.station_id] = monthly
            # A station number that started inside the period, or another station with the
            # same name family (e.g. PERTH AIRPORT 009291 replacing 009021), means the
            # folder may hold records from different station numbers.
            rec["id_ambiguous"] = bool(pd.Timestamp(st.db_start_date) > pd.Timestamp(first))
        rows.append(rec)
    cand = pd.DataFrame(rows)
    chosen, reason = stations.choose_station(cand, score_first, cfg.min_completeness, cfg.station_id)
    reason += f" (completeness scored over {score_first} .. {last})"
    cand["selected"] = cand.station_id == chosen.station_id
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cand.to_csv(cfg.reports_dir / "station_candidates.csv", index=False)
    if not near_all.empty:
        near_all[["station_id", "station_name", "start_year", "end_year", "open", "latitude", "longitude",
                  "distance_km"]].to_csv(cfg.reports_dir / "nearby_bom_stations_all_IDCJMC0014.csv", index=False)
    info = chosen.to_dict()
    info["selection_reason"] = reason
    # Gauges that are closer than the chosen one but have no automated daily file.
    # They are reachable only through Climate Data Online (manual download, --cdo-csv).
    closer = pd.DataFrame()
    if not near_all.empty:
        auto_ids = set(cand.station_id[cand.downloadable])
        closer = near_all[(near_all.distance_km < chosen.distance_km)
                          & (~near_all.station_id.isin(auto_ids))].copy()
        closer = closer[["station_id", "station_name", "distance_km", "start_year", "end_year", "open",
                         "latitude", "longitude"]].sort_values("distance_km")
        closer.to_csv(cfg.reports_dir / "closer_cdo_only_gauges.csv", index=False)
    info["closer_cdo_only_gauges"] = closer.head(8).to_dict("records")
    if not closer.empty:
        op = closer[closer.open]
        log.warning("%s: %d BoM gauges are closer than the selected station but need manual CDO "
                    "download (nearest open: %s)", cfg.location_name, len(closer),
                    f"{op.iloc[0].station_id} {op.iloc[0].station_name} at {op.iloc[0].distance_km:.1f} km"
                    if not op.empty else "none open")
    log.info("selected station %s %s (%.2f km): %s", chosen.station_id, chosen.station_name,
             chosen.distance_km, reason)
    return info, monthly_cache.get(chosen.station_id), cand


# ---------------------------------------------------------------------- run steps
def run_all(cfg: Config, *, skip_single_runs: bool = False) -> dict:
    f = make_fetcher(cfg)
    first, last = resolve_period(cfg)
    log.info("provisional period %s .. %s", first, last)

    obs_first, _ = resolve_obs_period(cfg, first, last)
    st, _, cand = select_station(cfg, f, first, last)
    folders = set(f.list_ftp_dir(f"{stations.FTP_BASE}/{stations.STATE_FOLDERS[st['state']]}/"))
    # re-download over the full observation period (station selection only scored the
    # forecast period, which is shorter)
    monthly = download_station_months(cfg, f, st, month_list(obs_first, last), folders)
    log.info("observations %s .. %s; forecasts from %s", obs_first, last, first)

    obs, problems = observations.build_ftp_observations(monthly, st, cfg.timezone, obs_first, last,
                                                        cfg.suspicious_daily_mm)
    # trim trailing days that the source has not reported yet (incomplete period)
    reported = obs[obs.obs_status != "missing_row"]
    last_reported = reported.label_date_local.max()
    obs = obs[obs.label_date_local <= last_reported].reset_index(drop=True)
    last = last_reported
    for p in cfg.cdo_csv_paths:
        cdo = observations.load_cdo_csv(p, st["station_id"], cfg.timezone)
        obs = observations.merge_cdo(obs, cdo, cfg.timezone)
    write_both(obs, cfg.clean_dir / "observations")
    last_complete_end = obs.window_end_utc.max()

    obs_windows = build.window_bins(obs)
    windows = obs_windows[obs_windows.label_date_local >= first].reset_index(drop=True)
    lat, lon = st["latitude"], st["longitude"]  # forecasts extracted at the gauge location

    pr_hourly, pr_fail = forecasts.download_previous_runs(
        f, lat, lon, cfg.previous_runs_models, first - timedelta(days=1), last, cfg.lead_days, cfg.concurrency)
    sr_frames, sr_fail = [], []
    if not skip_single_runs:
        for model, start in cfg.single_runs.items():
            run_start = max(pd.Timestamp(start), windows.window_start_utc.min().tz_localize(None).floor("D")
                            - pd.Timedelta(days=max(cfg.lead_days) + 1))
            run_end = windows.window_start_utc.max().tz_localize(None).floor("D")
            runs = [pd.Timestamp(d).tz_localize("UTC") + pd.Timedelta(hours=cfg.single_run_hour_utc)
                    for d in pd.date_range(run_start, run_end, freq="D")]
            needed = set()
            for n in cfg.lead_days:
                needed |= set(build.single_run_issue_time(windows.window_start_utc, n, cfg.single_run_hour_utc,
                                                          cfg.publication_latency_hours))
            runs = sorted(r for r in runs if r in needed)
            log.info("single_runs %s: %d runs %s .. %s", model, len(runs), runs[0], runs[-1])
            df, fl = forecasts.download_single_runs(f, lat, lon, model, runs, forecast_days=5,
                                                    concurrency=cfg.concurrency)
            sr_frames.append(df)
            sr_fail += fl
    sr_hourly = pd.concat([d for d in sr_frames if not d.empty], ignore_index=True) if sr_frames else pd.DataFrame()

    # hourly QC + clean forecast records
    pr_hourly, pr_conf = build.dedupe_hourly(pr_hourly, ["model", "lead_day", "valid_end_utc"])
    if not sr_hourly.empty:
        sr_hourly, sr_conf = build.dedupe_hourly(sr_hourly, ["model", "forecast_issue_time_utc", "valid_end_utc"])
    else:
        sr_conf = pd.DataFrame()
    hourly_all = pd.concat([pr_hourly, sr_hourly], ignore_index=True)
    hourly_all["negative_value_flag"] = hourly_all.precip_mm < 0
    # ~1M+ rows over a decade: Parquet only, and the repeated request URL is dropped
    # (it is preserved per request in data/raw/manifest.jsonl and per window in
    # forecast_windows / paired_long).
    hourly_slim = hourly_all.drop(columns=["source_url"])
    hourly_slim.to_parquet(cfg.clean_dir / "forecasts_hourly.parquet", index=False)
    log.info("wrote %s (.parquet only, %d rows)", cfg.clean_dir / "forecasts_hourly", len(hourly_slim))

    fw_pr = build.aggregate_previous_runs(pr_hourly, windows, cfg.lead_days, cfg.publication_latency_hours)
    fw_sr = build.aggregate_single_runs(sr_hourly, windows, cfg.lead_days, cfg.single_run_hour_utc,
                                        cfg.publication_latency_hours, list(cfg.single_runs)) \
        if not skip_single_runs else pd.DataFrame()
    fw = build.finalize_forecast_windows(pd.concat([fw_pr, fw_sr], ignore_index=True))
    fw_out = fw.drop(columns=["w", "precip_sum", "hour_rows"])
    write_both(fw_out, cfg.clean_dir / "forecast_windows")

    long = build.join_long(fw, obs, cfg, last_complete_end)
    write_both(long, cfg.joined_dir / "paired_long")
    wide = build.build_wide(long)
    write_both(wide, cfg.joined_dir / "training_wide")

    # --------------------------------------------------------------- diagnostics
    rep = cfg.reports_dir
    failures = pd.DataFrame(pr_fail + sr_fail + problems)
    failures.to_csv(rep / "download_failures.csv", index=False)
    excl = long[~long.training_eligible]
    excl.to_csv(rep / "exclusions.csv", index=False)
    excl_summary = (excl.assign(reason=excl.exclusion_reason.str.split(";")).explode("reason")
                    .groupby(["source_api", "model", "lead_group", "reason"]).size().rename("rows").reset_index())
    excl_summary.to_csv(rep / "exclusion_summary.csv", index=False)
    conflicts = pd.concat([pr_conf, sr_conf], ignore_index=True)
    conflicts.to_csv(rep / "forecast_conflicting_duplicates.csv", index=False)
    obs_diag = obs[obs.obs_status != "ok"]
    obs_diag.to_csv(rep / "observation_issues.csv", index=False)
    miss = (long.groupby(["station_id", "source_api", "model", "lead_group"])
            .agg(rows=("window_start_utc", "size"),
                 forecast_missing_pct=("forecast_precip_mm", lambda s: round(100 * s.isna().mean(), 2)),
                 observation_missing_pct=("observed_precip_mm", lambda s: round(100 * s.isna().mean(), 2)),
                 training_eligible_rows=("training_eligible", "sum"),
                 first_available=("label_date_local", lambda s: s[long.loc[s.index, "forecast_precip_mm"].notna()].min()),
                 last_available=("label_date_local", lambda s: s[long.loc[s.index, "forecast_precip_mm"].notna()].max()))
            .reset_index())
    miss.to_csv(rep / "missingness.csv", index=False)

    # ------------------------------------------------------------------ baseline
    th = cfg.rain_thresholds_mm
    per = baseline.per_model(long, th)
    pr_models_ok = [f"pr_{m}" for m in cfg.previous_runs_models
                    if long[(long.source_api == "previous_runs") & (long.model == m)].training_eligible.any()]
    comps = [per]
    if len(pr_models_ok) >= 2:
        comps.append(baseline.complete_case(wide, pr_models_ok, "complete_case_previous_runs", th))
    sr_ok = [f"sr_{m}" for m in cfg.single_runs
             if long[(long.source_api == "single_runs") & (long.model == m)].training_eligible.any()]
    if len(sr_ok) >= 2:
        comps.append(baseline.complete_case(wide, sr_ok, "complete_case_single_runs", th))
    cross = sr_ok[:1] + pr_models_ok
    if len(cross) >= 2:
        comps.append(baseline.complete_case(wide, cross, "complete_case_cross_source(lead definitions differ)", th))
    for m in cfg.single_runs:
        comps.append(baseline.chronological_scaling(long, "single_runs", m, th))
    metrics = pd.concat([c for c in comps if not c.empty], ignore_index=True)
    metrics.to_csv(rep / "baseline_metrics.csv", index=False)

    summary = {"generated_at_utc": utcnow_iso(), "location": cfg.location_name,
               "observations_first_label_date": str(obs.label_date_local.min()),
               "period_first_label_date": str(first),
               "period_last_label_date": str(last), "station": {k: (v if isinstance(v, (int, float, list, dict)) else str(v))
                            for k, v in st.items()},
               "fetch_stats": f.stats, "rows_long": len(long), "rows_eligible": int(long.training_eligible.sum()),
               "rows_wide": len(wide)}
    (rep / "run_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    write_manifest(cfg, summary)
    from .report import write_report
    write_report(cfg, summary, obs, long, wide, miss, excl_summary, metrics, failures, cand)
    return summary


def write_manifest(cfg: Config, summary: dict) -> None:
    """Source/provenance manifest: one entry per retrieval (from raw/manifest.jsonl) + sources."""
    entries = []
    mf = cfg.raw_dir / "manifest.jsonl"
    if mf.exists():
        for line in mf.read_text(encoding="utf-8").splitlines():
            entries.append(json.loads(line))
    latest = {}
    for e in entries:  # keep the latest retrieval per URL+params
        latest[(e["url"], json.dumps(e.get("params", {}), sort_keys=True))] = e
    manifest = {
        "generated_at_utc": utcnow_iso(),
        "sources": {
            "observations": {"product": observations.FTP_PRODUCT,
                             "base_url": stations.FTP_BASE,
                             "terms": "BoM default copyright terms: http://www.bom.gov.au/other/copyright.shtml",
                             "accumulation": "Rain 0900-0900 local clock time, total to 9am on the labelled date",
                             "quality_flags": "none in this product (unverified real-time AWS data)"},
            "station_lists": [stations.STATIONS_DB_URL, stations.ALL_STATIONS_URL],
            "previous_runs": {"url": forecasts.PREVIOUS_RUNS_URL,
                              "docs": "https://open-meteo.com/en/docs/previous-runs-api"},
            "single_runs": {"url": forecasts.SINGLE_RUNS_URL,
                            "docs": "https://open-meteo.com/en/docs/single-runs-api"},
            "open_meteo_terms": "https://open-meteo.com/en/terms (free tier: non-commercial, <10k calls/day, "
                                "<5k/hour, <600/min; data CC-BY 4.0, attribution required)",
            "target_coordinates": cfg.coordinate_source,
        },
        "run_summary": summary,
        "retrievals": list(latest.values()),
    }
    (cfg.reports_dir / "provenance_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))


def write_combined_summary(cfgs, summaries) -> None:
    """Index across all configured locations."""
    if not summaries:
        return
    root = Path(cfgs[0].reports_dir).parent
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg, s in zip(cfgs, summaries):
        st = s["station"]
        rows.append({"location": cfg.location_name, "target_latitude": cfg.latitude,
                     "target_longitude": cfg.longitude, "station_id": st["station_id"],
                     "station_name": st["station_name"],
                     "distance_km": round(float(st["distance_km"]), 2),
                     "observations_first": s["observations_first_label_date"],
                     "paired_first": s["period_first_label_date"],
                     "paired_last": s["period_last_label_date"],
                     "rows_long": s["rows_long"], "rows_eligible": s["rows_eligible"],
                     "report": str(Path(cfg.reports_dir) / "data_quality_report.md")})
    df = pd.DataFrame(rows)
    df.to_csv(root / "locations_summary.csv", index=False)
    (root / "locations_summary.json").write_text(json.dumps(rows, indent=1, default=str))
    log.info("wrote %s", root / "locations_summary.csv")


def rebuild_report(cfg: Config) -> Path:
    """Regenerate reports/<slug>/data_quality_report.md from the files a previous `run` wrote."""
    from .report import write_report
    rep, clean, joined = Path(cfg.reports_dir), Path(cfg.clean_dir), Path(cfg.joined_dir)
    summary = json.loads((rep / "run_summary.json").read_text())
    obs = pd.read_parquet(clean / "observations.parquet")
    long = pd.read_parquet(joined / "paired_long.parquet")
    wide = pd.read_parquet(joined / "training_wide.parquet")
    miss = pd.read_csv(rep / "missingness.csv")
    excl = pd.read_csv(rep / "exclusion_summary.csv")
    metrics = pd.read_csv(rep / "baseline_metrics.csv")
    fpath = rep / "download_failures.csv"
    failures = pd.read_csv(fpath) if fpath.exists() and fpath.stat().st_size > 5 else pd.DataFrame(
        columns=["stage", "item", "url", "problem"])
    cand = pd.read_csv(rep / "station_candidates.csv", dtype={"station_id": str})
    cand["downloadable"] = cand["downloadable"].astype(bool)
    write_report(cfg, summary, obs, long, wide, miss, excl, metrics, failures, cand)
    return rep / "data_quality_report.md"
