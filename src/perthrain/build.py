"""Window aggregation, forecast/observation join, eligibility rules and wide table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import forecasts
from .forecasts import NATIVE_STEP_HOURS

# Models whose previous_dayN -> run mapping was checked against the Single Runs API
# (reports/probe_verification.json). Others keep the mapping marked unverified.
VERIFIED_RUN_MAPPING = ("ncep_gfs_global", "ecmwf_ifs025", "jma_gsm")

SEASONS = {12: "DJF_summer", 1: "DJF_summer", 2: "DJF_summer", 3: "MAM_autumn", 4: "MAM_autumn",
           5: "MAM_autumn", 6: "JJA_winter", 7: "JJA_winter", 8: "JJA_winter", 9: "SON_spring",
           10: "SON_spring", 11: "SON_spring"}

LONG_COLUMNS = [
    "location_name", "target_latitude", "target_longitude", "station_id", "station_name",
    "station_latitude", "station_longitude", "source_api", "model", "forecast_issue_time_utc",
    "issue_time_status", "issue_time_inferred_min_utc", "issue_time_inferred_max_utc",
    "source_lead_offset", "lead_hours", "lead_hours_to_window_start", "lead_group",
    "label_date_local", "window_start_utc", "window_end_utc", "window_start_local", "window_end_local",
    "forecast_precip_mm", "observed_precip_mm", "expected_hour_count", "available_hour_count",
    "forecast_complete", "alignment_status", "available_before_window_start_est",
    "observation_quality_flag", "obs_status", "training_eligible", "exclusion_reason",
    "grid_latitude", "grid_longitude", "source_url", "retrieved_at_utc"]


def window_bins(obs: pd.DataFrame) -> pd.DataFrame:
    """Contiguous 24-h (single-day) observation windows used as aggregation bins."""
    w = obs.loc[obs.period_days == 1, ["label_date_local", "window_start_utc", "window_end_utc"]]
    return w.drop_duplicates().sort_values("window_start_utc").reset_index(drop=True)


def assign_windows(valid_end: pd.Series, windows: pd.DataFrame) -> pd.Series:
    """Map hourly accumulation labels (end of hour) to the window containing that hour.

    A label t belongs to window (start, end] because the hour it represents is
    [t-1h, t). Returns the positional index into ``windows`` (or -1).
    """
    starts = windows.window_start_utc.to_numpy()
    ends = windows.window_end_utc.to_numpy()
    t = valid_end.to_numpy()
    pos = np.searchsorted(ends, t, side="left")
    ok = (pos < len(ends))
    pos_c = np.where(ok, pos, 0)
    ok &= (t > starts[pos_c]) & (t <= ends[pos_c])
    return pd.Series(np.where(ok, pos, -1), index=valid_end.index)


def dedupe_hourly(df: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop exact duplicate hourly rows; flag (and null out) conflicting duplicates."""
    df = df.drop_duplicates(subset=keys + ["precip_mm"])
    dup = df.duplicated(subset=keys, keep=False)
    conflicts = df[dup].copy()
    if not conflicts.empty:
        df = df[~dup]
        keep = conflicts.drop_duplicates(subset=keys).copy()
        keep["precip_mm"] = np.nan  # conflicting values are not trusted
        df = pd.concat([df, keep], ignore_index=True)
    return df, conflicts


def _hourly_stats(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    g = df.groupby(group_cols, dropna=False)
    out = g.agg(available_hour_count=("precip_mm", "count"), hour_rows=("precip_mm", "size"),
                precip_sum=("precip_mm", "sum"), negative_hours=("precip_mm", lambda s: int((s < 0).sum())),
                grid_latitude=("grid_latitude", "first"), grid_longitude=("grid_longitude", "first"),
                source_url=("source_url", lambda s: " | ".join(sorted(set(s)))),
                retrieved_at_utc=("retrieved_at_utc", "max"))
    return out.reset_index()


def aggregate_previous_runs(hourly: pd.DataFrame, windows: pd.DataFrame, lead_days: list[int],
                            latency_h: float) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame()
    h = hourly.copy()
    h["w"] = assign_windows(h.valid_end_utc, windows)
    h = h[h.w >= 0]
    stats = _hourly_stats(h, ["model", "lead_day", "source_lead_offset", "w"])
    iss = h.groupby(["model", "lead_day", "w"]).issue_time_inferred_utc.agg(["min", "max"]).reset_index()
    stats = stats.merge(iss, on=["model", "lead_day", "w"], how="left")
    # full grid: every window x model x lead, so missing forecasts are explicit
    models = sorted(hourly.model.unique())
    grid = pd.MultiIndex.from_product([models, lead_days, range(len(windows))],
                                      names=["model", "lead_day", "w"]).to_frame(index=False)
    grid["source_lead_offset"] = "previous_day" + grid.lead_day.astype(str)
    out = grid.merge(stats, on=["model", "lead_day", "source_lead_offset", "w"], how="left")
    out = out.join(windows, on="w")
    out["source_api"] = "previous_runs"
    out["forecast_issue_time_utc"] = pd.NaT
    out["issue_time_status"] = np.where(
        out.model.isin(VERIFIED_RUN_MAPPING),
        "not_provided_fixed_offset;inferred_range_verified_by_probe",
        "not_provided_fixed_offset;inferred_range_unverified")
    out["issue_time_inferred_min_utc"] = out["min"]
    out["issue_time_inferred_max_utc"] = out["max"]
    # inferred max issue time is deterministic even when data are missing
    latest = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    for (m, n), idx in out.groupby(["model", "lead_day"]).groups.items():
        latest.loc[idx] = forecasts.inferred_run_time(out.loc[idx, "window_end_utc"], m, n)
    out["issue_time_inferred_max_utc"] = out.issue_time_inferred_max_utc.fillna(latest)
    out["lead_hours"] = (out.window_end_utc - out.issue_time_inferred_max_utc) / pd.Timedelta(hours=1)
    out["lead_hours_to_window_start"] = (out.window_start_utc - out.issue_time_inferred_max_utc) / pd.Timedelta(hours=1)
    out["available_before_window_start_est"] = (
        out.issue_time_inferred_max_utc + pd.Timedelta(hours=latency_h)) <= out.window_start_utc
    return out.drop(columns=["min", "max"])


def single_run_issue_time(window_start: pd.Series, lead_day: int, run_hour: int,
                          latency_h: float) -> pd.Series:
    """Latest run at ``run_hour`` UTC estimated to be published by window_start - 24*(lead_day-1) h."""
    deadline = window_start - pd.Timedelta(hours=24 * (lead_day - 1) + latency_h)
    cand = deadline.dt.floor("D") + pd.Timedelta(hours=run_hour)
    return cand.where(cand <= deadline, cand - pd.Timedelta(days=1))


def aggregate_single_runs(hourly: pd.DataFrame, windows: pd.DataFrame, lead_days: list[int],
                          run_hour: int, latency_h: float, models: list[str]) -> pd.DataFrame:
    frames = []
    for n in lead_days:
        f = windows.copy()
        f["w"] = range(len(windows))
        f["lead_day"] = n
        f["forecast_issue_time_utc"] = single_run_issue_time(f.window_start_utc, n, run_hour, latency_h)
        frames.append(f)
    need = pd.concat(frames, ignore_index=True)
    need = pd.concat([need.assign(model=m) for m in models], ignore_index=True)
    if hourly.empty:
        stats = pd.DataFrame(columns=["model", "forecast_issue_time_utc", "w", "available_hour_count",
                                      "hour_rows", "precip_sum", "negative_hours", "grid_latitude",
                                      "grid_longitude", "source_url", "retrieved_at_utc"])
    else:
        h = hourly.copy()
        h["w"] = assign_windows(h.valid_end_utc, windows)
        h = h[h.w >= 0]
        stats = _hourly_stats(h, ["model", "forecast_issue_time_utc", "w"])
    out = need.merge(stats, on=["model", "forecast_issue_time_utc", "w"], how="left")
    out["source_api"] = "single_runs"
    out["source_lead_offset"] = None
    out["issue_time_status"] = "explicit_run_initialisation_time"
    out["issue_time_inferred_min_utc"] = pd.NaT
    out["issue_time_inferred_max_utc"] = pd.NaT
    out["lead_hours"] = (out.window_end_utc - out.forecast_issue_time_utc) / pd.Timedelta(hours=1)
    out["lead_hours_to_window_start"] = (out.window_start_utc - out.forecast_issue_time_utc) / pd.Timedelta(hours=1)
    out["available_before_window_start_est"] = (
        out.forecast_issue_time_utc + pd.Timedelta(hours=latency_h)) <= out.window_start_utc
    return out


def finalize_forecast_windows(fw: pd.DataFrame) -> pd.DataFrame:
    fw = fw.copy()
    fw["expected_hour_count"] = ((fw.window_end_utc - fw.window_start_utc) / pd.Timedelta(hours=1)).astype(int)
    fw["available_hour_count"] = fw.available_hour_count.fillna(0).astype(int)
    fw["forecast_complete"] = (fw.available_hour_count == fw.expected_hour_count) & \
        (fw.hour_rows.fillna(0) == fw.expected_hour_count)
    # never manufacture totals from partial coverage
    fw["forecast_precip_mm"] = np.where(fw.forecast_complete, fw.precip_sum.round(3), np.nan)
    step = fw.model.map(NATIVE_STEP_HOURS).fillna(1).astype(int)
    start_h = fw.window_start_utc.dt.hour
    end_h = fw.window_end_utc.dt.hour
    aligned = (start_h % step == 0) & (end_h % step == 0)
    # For 3 h / 6 h models the window boundary splits a native accumulation block. The
    # split is well defined (Open-Meteo spreads each block uniformly over its hours -
    # verified: 100 % of rainy blocks have identical hourly values), so the window total
    # is a deterministic linear combination of native blocks, with the sub-block
    # distribution of the two boundary blocks unknown.
    fw["alignment_status"] = np.where(
        aligned, "exact_hourly_boundaries",
        "uniform_disaggregation_boundary_split_" + step.astype(str) + "h")
    fw["lead_group"] = "day" + fw.lead_day.astype(str)
    return fw


def join_long(fw: pd.DataFrame, obs: pd.DataFrame, cfg, last_complete_end_utc: pd.Timestamp) -> pd.DataFrame:
    o = obs[["window_start_utc", "window_end_utc", "station_id", "station_name", "station_latitude",
             "station_longitude", "observed_precip_mm", "observation_quality_flag", "obs_status",
             "period_days"]]
    # join on the EXACT accumulation interval, never on a date string
    long = fw.drop(columns=["label_date_local"]).merge(o, on=["window_start_utc", "window_end_utc"], how="left")
    long = long.merge(obs[["window_start_utc", "window_end_utc", "label_date_local"]],
                      on=["window_start_utc", "window_end_utc"], how="left")
    long["location_name"] = cfg.location_name
    long["target_latitude"] = cfg.latitude
    long["target_longitude"] = cfg.longitude
    long["window_start_local"] = long.window_start_utc.dt.tz_convert(cfg.timezone)
    long["window_end_local"] = long.window_end_utc.dt.tz_convert(cfg.timezone)

    reasons = pd.Series([[] for _ in range(len(long))], index=long.index)

    def add(mask, text):
        for i in long.index[mask.fillna(True) if hasattr(mask, "fillna") else mask]:
            reasons[i].append(text)

    add(long.observed_precip_mm.isna(), "no_valid_observation_for_exact_window")
    add(long.obs_status.isin(["conflicting_duplicate", "negative_value", "multi_day_accumulation",
                              "unparseable_value"]), "observation_rejected_status")
    add(long.available_hour_count == 0, "no_forecast_data_for_window")
    add((long.available_hour_count > 0) & ~long.forecast_complete, "incomplete_forecast_hours")
    add(long.negative_hours.fillna(0) > 0, "negative_forecast_precipitation")
    if getattr(cfg, "alignment_policy", "allow_uniform_disaggregation") == "strict":
        add(long.alignment_status != "exact_hourly_boundaries", "accumulation_window_not_exactly_aligned")
    else:
        # uniform disaggregation is accepted (flagged in alignment_status); anything else is not
        add(~long.alignment_status.str.startswith(("exact_hourly_boundaries",
                                                   "uniform_disaggregation")),
            "accumulation_window_not_exactly_aligned")
    issue_latest = long.forecast_issue_time_utc.fillna(long.issue_time_inferred_max_utc)
    add(issue_latest >= long.window_start_utc, "forecast_issued_after_window_start")
    if cfg.require_publication_before_window:
        add(~long.available_before_window_start_est.astype(bool), "not_published_before_window_start_est")
    add(long.window_end_utc > last_complete_end_utc, "window_not_complete_or_future")
    long["exclusion_reason"] = reasons.map(lambda r: ";".join(r) if r else None)
    long["training_eligible"] = long.exclusion_reason.isna()
    long["observation_quality_flag"] = long.observation_quality_flag.fillna("no_observation")
    long["station_id"] = long.station_id.fillna(obs.station_id.iloc[0])
    long["station_name"] = long.station_name.fillna(obs.station_name.iloc[0])
    long["station_latitude"] = long.station_latitude.fillna(obs.station_latitude.iloc[0])
    long["station_longitude"] = long.station_longitude.fillna(obs.station_longitude.iloc[0])
    return long[LONG_COLUMNS].sort_values(
        ["source_api", "model", "lead_group", "window_start_utc"]).reset_index(drop=True)


def build_wide(long: pd.DataFrame) -> pd.DataFrame:
    """One row per station + target window + lead group; model columns side by side."""
    key = ["station_id", "station_name", "label_date_local", "window_start_utc", "window_end_utc",
           "window_start_local", "window_end_local", "lead_group"]
    long = long.copy()
    long["col"] = np.where(long.source_api == "single_runs", "sr_", "pr_") + long.model
    long["available"] = long.forecast_precip_mm.notna()

    def spread(value_col: str, prefix: str) -> pd.DataFrame:
        """Pivot one value column to one column per model.

        groupby(...).first().unstack() only materialises observed key/model pairs.
        ``pivot_table(dropna=False)`` must not be used here: it forms the cartesian
        product of every index level, which explodes on a multi-year period.
        """
        out = long.groupby(key + ["col"], dropna=False, observed=True)[value_col].first().unstack("col")
        out.columns = [f"{prefix}{c}" + ("_mm" if prefix == "fc_" else "") for c in out.columns]
        return out

    fc = spread("forecast_precip_mm", "fc_")
    av = spread("available", "available_")
    el = spread("training_eligible", "eligible_")
    lh = spread("lead_hours", "lead_hours_")
    obs = long.groupby(key, dropna=False, observed=True).agg(
        observed_precip_mm=("observed_precip_mm", "first"),
        observation_quality_flag=("observation_quality_flag", "first"),
        obs_status=("obs_status", "first"))
    wide = obs.join([fc, av.fillna(False).astype(bool), el.fillna(False).astype(bool), lh]).reset_index()
    wide["month"] = pd.to_datetime(wide.label_date_local).dt.month
    wide["season"] = wide.month.map(SEASONS)
    el_cols = [c for c in wide.columns if c.startswith("eligible_")]
    for fam in ("pr", "sr"):
        cols = [c for c in el_cols if c.startswith(f"eligible_{fam}_")]
        wide[f"n_eligible_{fam}_models"] = wide[cols].sum(axis=1)
        # complete case among models that are ever eligible in this family
        usable = [c for c in cols if wide[c].any()]
        wide[f"complete_case_{fam}"] = wide[usable].all(axis=1) if usable else False
    wide["observed_rain_ge_0_2mm"] = np.where(wide.observed_precip_mm.notna(),
                                              wide.observed_precip_mm >= 0.2, None)
    return wide.sort_values(["lead_group", "window_start_utc"]).reset_index(drop=True)
