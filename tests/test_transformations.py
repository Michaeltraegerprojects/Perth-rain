from datetime import date

import numpy as np
import pandas as pd
import pytest

from perthrain import baseline, build, forecasts, observations
from perthrain.config import Config

TZ = "Australia/Perth"
STATION = {"station_id": "009225", "station_name": "PERTH METRO", "latitude": -31.9192, "longitude": 115.8728}

SAMPLE = """'IDCKWCDE11,,,,,,,,,,
Australian Government Bureau of Meteorology,,,,,,,,,,
South Australia,,,,,,,,,,

Daily Evapotranspiration for PERTH METRO Western Australia for August 2026,,,,,,,,,,
Issued at 22:31 GMT on Tuesday 01 September 2026,,,,,,,,,,
Copyright,,,,,,,,,,
Please note,,,,,,,,,,

,,Evapo-,,Pan,,,Maximum,Minimum,Average,
,,Transpiration,Rain,Evaporation,Maximum,Minimum,Relative,Relative,10m Wind,Solar
Station Name,Date,0000-2400,0900-0900,0900-0900,Temperature,Temperature,Humidity,Humidity,Speed,Radiation
,,(mm),(mm),(mm),(C),(C),(%),(%),(m/sec),(MJ/sq m)
PERTH METRO,01/08/2026,1.9,6.0, ,17.7,10.3,99,45,2.21,12.63
PERTH METRO,02/08/2026,1.8, , ,17.5,4.6,98,45,1.60,14.29
PERTH METRO,03/08/2026,2.3,0.0, ,25.3,6.6,89,26,1.25,14.10
PERTH METRO,05/08/2026,2.3,-1.0, ,25.3,6.6,89,26,1.25,14.10
PERTH METRO,06/08/2026,2.3,150.0, ,25.3,6.6,89,26,1.25,14.10
PERTH METRO,07/08/2026,2.3,1.0, ,25.3,6.6,89,26,1.25,14.10
PERTH METRO,07/08/2026,2.3,2.0, ,25.3,6.6,89,26,1.25,14.10
PERTH METRO,08/08/2026,2.3,3.0, ,25.3,6.6,89,26,1.25,14.10
PERTH METRO,08/08/2026,2.3,3.0, ,25.3,6.6,89,26,1.25,14.10
Totals:,,67.7,97, ,,,,,,
"""


# ------------------------------------------------------------------ observations
def test_obs_window_perth_is_0100_utc_to_0100_utc():
    s, e = observations.obs_window(date(2026, 6, 1), TZ)
    assert s == pd.Timestamp("2026-05-31T01:00Z") and e == pd.Timestamp("2026-06-01T01:00Z")


def test_obs_window_handles_dst_zone():
    # Sydney DST starts 2025-10-05 02:00: the 9am-9am window is 23 hours long
    s, e = observations.obs_window(date(2025, 10, 5), "Australia/Sydney")
    assert (e - s) == pd.Timedelta(hours=23)


def test_parse_validates_rain_column_and_keeps_blank_as_missing():
    df, meta = observations.parse_idckwcdea0(SAMPLE.encode())
    assert meta["rain_column"] == "Rain 0900-0900 (mm)"
    assert meta["file_issued_at_utc"] == pd.Timestamp("2026-09-01T22:31Z")
    assert "Totals:" not in df.station_name_in_file.values
    assert df.loc[df.label_date == date(2026, 8, 2), "raw_value"].item() == ""


def test_parse_rejects_unexpected_rain_column():
    bad = SAMPLE.replace("0900-0900,0900-0900", "0000-2400,0900-0900", 1)
    with pytest.raises(observations.ObservationFormatError):
        observations.parse_idckwcdea0(bad.encode())


def test_build_observations_quality_rules():
    obs, problems = observations.build_ftp_observations(
        [("u", SAMPLE.encode(), "t", "202608")], STATION, TZ, date(2026, 8, 1), date(2026, 8, 8), 100.0)
    s = obs.set_index("label_date_local")
    assert s.loc[date(2026, 8, 1), "observed_precip_mm"] == 6.0
    # blank value -> missing, never zero
    assert pd.isna(s.loc[date(2026, 8, 2), "observed_precip_mm"])
    assert s.loc[date(2026, 8, 2), "obs_status"] == "missing_value"
    assert s.loc[date(2026, 8, 3), "observed_precip_mm"] == 0.0
    assert s.loc[date(2026, 8, 4), "obs_status"] == "missing_row"
    assert pd.isna(s.loc[date(2026, 8, 4), "observed_precip_mm"])
    assert s.loc[date(2026, 8, 5), "obs_status"] == "negative_value"
    assert pd.isna(s.loc[date(2026, 8, 5), "observed_precip_mm"])
    # extreme kept but flagged
    assert s.loc[date(2026, 8, 6), "observed_precip_mm"] == 150.0
    assert s.loc[date(2026, 8, 6), "obs_status"] == "suspicious_extreme_kept"
    # conflicting duplicate flagged and not trusted; exact duplicate collapsed
    assert s.loc[date(2026, 8, 7), "obs_status"] == "conflicting_duplicate"
    assert pd.isna(s.loc[date(2026, 8, 7), "observed_precip_mm"])
    assert s.loc[date(2026, 8, 8), "observed_precip_mm"] == 3.0 and s.loc[date(2026, 8, 8), "obs_status"] == "ok"
    assert len(obs) == 8


def test_cdo_loader_refuses_other_station(tmp_path):
    p = tmp_path / "cdo.csv"
    p.write_text("Product code,Bureau of Meteorology station number,Year,Month,Day,Rainfall amount (millimetres),"
                 "Period over which rainfall was measured (days),Quality\n"
                 "IDCJAC0009,009021,2026,08,01,6.0,1,N\n")
    with pytest.raises(observations.ObservationFormatError):
        observations.load_cdo_csv(str(p), "009225", TZ)


def test_cdo_multiday_accumulation_gets_true_window(tmp_path):
    p = tmp_path / "cdo.csv"
    p.write_text("Product code,Bureau of Meteorology station number,Year,Month,Day,Rainfall amount (millimetres),"
                 "Period over which rainfall was measured (days),Quality\n"
                 "IDCJAC0009,009225,2026,08,01,6.0,1,Y\n"
                 "IDCJAC0009,009225,2026,08,03,4.0,2,N\n")
    cdo = observations.load_cdo_csv(str(p), "009225", TZ)
    obs, _ = observations.build_ftp_observations(
        [("u", SAMPLE.encode(), "t", "202608")], STATION, TZ, date(2026, 8, 1), date(2026, 8, 3), 100.0)
    m = observations.merge_cdo(obs, cdo, TZ).set_index("label_date_local")
    assert m.loc[date(2026, 8, 1), "observation_quality_flag"] == "Y_quality_controlled"
    assert m.loc[date(2026, 8, 3), "obs_status"] == "multi_day_accumulation"
    assert m.loc[date(2026, 8, 3), "window_start_utc"] == pd.Timestamp("2026-08-01T01:00Z")


# --------------------------------------------------------------------- forecasts
def _pr_body(hours, values, unit="mm"):
    times = pd.date_range("2026-06-01T00:00", periods=hours, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    return {"latitude": -31.9, "longitude": 115.9, "elevation": 20,
            "hourly_units": {"time": "iso8601", "precipitation_previous_day1": unit},
            "hourly": {"time": times, "precipitation_previous_day1": values}}


def test_previous_runs_inferred_issue_time_and_unit_check():
    df = forecasts.parse_previous_runs(_pr_body(8, [0.0] * 8), "ncep_gfs_global", [1], "u", {}, "t")
    row = df[df.valid_end_utc == pd.Timestamp("2026-06-01T07:00Z")].iloc[0]
    assert row.issue_time_inferred_utc == pd.Timestamp("2026-05-31T06:00Z")
    assert pd.isna(row.forecast_issue_time_utc)  # never invented
    with pytest.raises(ValueError):
        forecasts.parse_previous_runs(_pr_body(8, [0.0] * 8, unit="inch"), "m", [1], "u", {}, "t")


def _windows(n=3, first="2026-06-01"):
    rows = []
    for d in pd.date_range(first, periods=n, freq="D").date:
        s, e = observations.obs_window(d, TZ)
        rows.append({"label_date_local": d, "window_start_utc": s, "window_end_utc": e})
    return pd.DataFrame(rows)


def test_assign_windows_uses_end_labelled_hours():
    w = _windows(2)
    t = pd.Series(pd.to_datetime(["2026-05-31T01:00Z", "2026-05-31T02:00Z", "2026-06-01T01:00Z",
                                  "2026-06-01T02:00Z"]))
    got = build.assign_windows(t, w).tolist()
    # hour ending at window start belongs to the previous window (-1 here)
    assert got == [-1, 0, 0, 1]


def _hourly(model, lead, start, end, value=0.5, drop=()):
    t = pd.date_range(start, end, freq="h", tz="UTC")
    df = pd.DataFrame({"model": model, "lead_day": lead, "source_lead_offset": f"previous_day{lead}",
                       "valid_end_utc": t, "precip_mm": value, "grid_latitude": 0.0, "grid_longitude": 0.0,
                       "source_url": "u", "retrieved_at_utc": "t"})
    df["issue_time_inferred_utc"] = df.valid_end_utc.dt.floor("6h") - pd.Timedelta(hours=24 * lead)
    df.loc[df.valid_end_utc.isin(pd.to_datetime(list(drop), utc=True)), "precip_mm"] = np.nan
    return df


def test_window_totals_require_complete_hours_and_alignment():
    w = _windows(2)
    h = pd.concat([
        _hourly("ncep_gfs_global", 2, "2026-05-30", "2026-06-03", drop=["2026-06-01T05:00Z"]),
        _hourly("ecmwf_ifs025", 2, "2026-05-30", "2026-06-03")])
    fw = build.finalize_forecast_windows(build.aggregate_previous_runs(h, w, [2], 6))
    g = fw[fw.model == "ncep_gfs_global"].set_index("label_date_local")
    assert g.loc[date(2026, 6, 1), "forecast_precip_mm"] == pytest.approx(12.0)
    assert g.loc[date(2026, 6, 1), "expected_hour_count"] == 24
    assert pd.isna(g.loc[date(2026, 6, 2), "forecast_precip_mm"])  # one hour missing -> no total
    assert g.loc[date(2026, 6, 2), "available_hour_count"] == 23
    e = fw[fw.model == "ecmwf_ifs025"].iloc[0]
    assert e.alignment_status == "uniform_disaggregation_boundary_split_3h"
    # day2 composite: latest run = window_end floor 6h - 48h -> published before window start
    assert bool(g.loc[date(2026, 6, 1), "available_before_window_start_est"])


def test_previous_runs_day1_not_published_before_window():
    w = _windows(1)
    h = _hourly("ncep_gfs_global", 1, "2026-05-30", "2026-06-02")
    fw = build.finalize_forecast_windows(build.aggregate_previous_runs(h, w, [1], 6))
    r = fw.iloc[0]
    assert r.issue_time_inferred_max_utc == pd.Timestamp("2026-05-31T00:00Z")  # 1 h before window start
    assert not r.available_before_window_start_est


def test_single_run_issue_time_perth():
    ws = pd.Series([pd.Timestamp("2026-06-10T01:00Z")])
    assert build.single_run_issue_time(ws, 1, 12, 6).iloc[0] == pd.Timestamp("2026-06-09T12:00Z")
    assert build.single_run_issue_time(ws, 3, 12, 6).iloc[0] == pd.Timestamp("2026-06-07T12:00Z")


def test_join_is_on_exact_window_and_sets_eligibility():
    cfg = Config()
    obs, _ = observations.build_ftp_observations(
        [("u", SAMPLE.encode(), "t", "202608")], STATION, TZ, date(2026, 8, 1), date(2026, 8, 3), 100.0)
    w = build.window_bins(obs)
    h = _hourly("ncep_gfs_global", 1, "2026-07-30", "2026-08-04", value=0.25)
    h2 = _hourly("ncep_gfs_global", 2, "2026-07-30", "2026-08-04", value=0.25)
    fw = build.finalize_forecast_windows(build.aggregate_previous_runs(pd.concat([h, h2]), w, [1, 2], 6))
    long = build.join_long(fw, obs, cfg, obs.window_end_utc.max())
    d1 = long[(long.lead_group == "day1") & (long.label_date_local == date(2026, 8, 1))].iloc[0]
    assert not d1.training_eligible and "not_published_before_window_start_est" in d1.exclusion_reason
    d2 = long[(long.lead_group == "day2") & (long.label_date_local == date(2026, 8, 1))].iloc[0]
    assert d2.training_eligible and d2.forecast_precip_mm == pytest.approx(6.0) and d2.observed_precip_mm == 6.0
    d2m = long[(long.lead_group == "day2") & (long.label_date_local == date(2026, 8, 2))].iloc[0]
    assert not d2m.training_eligible and "no_valid_observation" in d2m.exclusion_reason
    wide = build.build_wide(long)
    assert {"fc_pr_ncep_gfs_global_mm", "available_pr_ncep_gfs_global", "month", "season"} <= set(wide.columns)
    assert len(wide) == len(long)  # one model -> one row per window x lead


def test_dedupe_hourly_flags_conflicts():
    h = _hourly("m", 1, "2026-06-01T00:00", "2026-06-01T02:00")
    dup_same = h.iloc[[0]]
    dup_diff = h.iloc[[1]].assign(precip_mm=9.9)
    out, conflicts = build.dedupe_hourly(pd.concat([h, dup_same, dup_diff]), ["model", "lead_day", "valid_end_utc"])
    assert len(out) == 3 and len(conflicts) == 2
    assert pd.isna(out[out.valid_end_utc == h.valid_end_utc.iloc[1]].precip_mm.item())


# ---------------------------------------------------------------------- baseline
def test_contingency_and_signed_error():
    fc = pd.Series([0.0, 1.0, 3.0, 0.0])
    ob = pd.Series([0.0, 0.0, 2.0, 1.0])
    s = baseline.scores(fc, ob, [0.2])
    assert s["hits_0.2"] == 1 and s["false_alarms_0.2"] == 1 and s["misses_0.2"] == 1
    assert s["correct_negatives_0.2"] == 1
    assert s["mean_error_fc_minus_obs_mm"] == pytest.approx(0.25)
    assert s["MAE_mm"] == pytest.approx(0.75)


# ------------------------------------- verified previous-runs run-inference semantics
@pytest.mark.parametrize("model,step,valid_end,expected", [
    # verified against the Single Runs API (reports/probe_verification.json):
    # run = floor_6h(ceil_to_native_step(valid_end)) - 24*N
    ("ncep_gfs_global", 1, "2026-08-10T07:00Z", "2026-08-09T06:00Z"),
    ("ncep_gfs_global", 1, "2026-08-10T06:00Z", "2026-08-09T06:00Z"),
    ("ecmwf_ifs025", 3, "2026-08-10T04:00Z", "2026-08-09T06:00Z"),
    ("ecmwf_ifs025", 3, "2026-08-10T01:00Z", "2026-08-09T00:00Z"),
    ("jma_gsm", 6, "2026-08-10T01:00Z", "2026-08-09T06:00Z"),
    ("jma_gsm", 6, "2026-08-10T00:00Z", "2026-08-09T00:00Z"),
    ("jma_gsm", 6, "2026-08-10T07:00Z", "2026-08-09T12:00Z"),
])
def test_inferred_run_time_matches_verified_rule(model, step, valid_end, expected):
    assert forecasts.NATIVE_STEP_HOURS[model] == step
    got = forecasts.inferred_run_time(pd.Series([pd.Timestamp(valid_end)]), model, 1)
    assert got.iloc[0] == pd.Timestamp(expected)


def test_inferred_run_time_scales_with_lead_day():
    t = pd.Series([pd.Timestamp("2026-08-10T07:00Z")])
    a = forecasts.inferred_run_time(t, "jma_gsm", 1).iloc[0]
    c = forecasts.inferred_run_time(t, "jma_gsm", 3).iloc[0]
    assert c == a - pd.Timedelta(hours=48)


def test_alignment_policy_controls_eligibility():
    obs, _ = observations.build_ftp_observations(
        [("u", SAMPLE.encode(), "t", "202608")], STATION, TZ, date(2026, 8, 1), date(2026, 8, 3), 100.0)
    w = build.window_bins(obs)
    h = _hourly("jma_gsm", 2, "2026-07-30", "2026-08-04", value=0.25)
    fw = build.finalize_forecast_windows(build.aggregate_previous_runs(h, w, [2], 6))
    assert fw.alignment_status.iloc[0] == "uniform_disaggregation_boundary_split_6h"
    lenient = build.join_long(fw, obs, Config(alignment_policy="allow_uniform_disaggregation"), obs.window_end_utc.max())
    strict = build.join_long(fw, obs, Config(alignment_policy="strict"), obs.window_end_utc.max())
    r_len = lenient[lenient.label_date_local == date(2026, 8, 1)].iloc[0]
    r_str = strict[strict.label_date_local == date(2026, 8, 1)].iloc[0]
    assert r_len.training_eligible
    assert not r_str.training_eligible
    assert "accumulation_window_not_exactly_aligned" in r_str.exclusion_reason


def test_multi_location_config_gives_separate_dirs(tmp_path):
    from perthrain.config import load_configs
    p = tmp_path / "c.toml"
    p.write_text(
        '[[locations]]\nname = "Perth"\nlatitude = -31.95\nlongitude = 115.86\nid = "009225"\n'
        '[[locations]]\nname = "Ocean Reef"\nlatitude = -31.75\nlongitude = 115.73\n'
        '[location]\ntimezone = "Australia/Perth"\n')
    cfgs = load_configs(str(p))
    assert [c.location_name for c in cfgs] == ["Perth", "Ocean Reef"]
    assert cfgs[0].station_id == "009225" and cfgs[1].station_id == ""
    assert cfgs[0].data_dir.name == "perth" and cfgs[1].data_dir.name == "ocean_reef"
    # raw cache is shared, not per location
    assert cfgs[0].raw_dir == cfgs[1].raw_dir
    assert [c.location_name for c in load_configs(str(p), only="ocean reef")] == ["Ocean Reef"]


def test_build_wide_scales_to_many_windows():
    """Regression: pivot_table(dropna=False) built a cartesian product of all index
    levels and blew up on multi-year periods. One row per window x lead group."""
    n_days, leads, models = 400, [1, 2, 3], ["jma_gsm", "ncep_gfs_global"]
    w = _windows(n_days, first="2020-01-01")
    h = pd.concat([_hourly(m, n, "2019-12-29", "2021-03-01", value=0.1)
                   for m in models for n in leads])
    fw = build.finalize_forecast_windows(build.aggregate_previous_runs(h, w, leads, 6))
    obs = pd.DataFrame({
        "label_date_local": w.label_date_local, "window_start_utc": w.window_start_utc,
        "window_end_utc": w.window_end_utc, "station_id": "009225", "station_name": "PERTH METRO",
        "station_latitude": -31.9, "station_longitude": 115.9, "observed_precip_mm": 1.0,
        "observation_quality_flag": "not_provided_unqc_realtime", "obs_status": "ok", "period_days": 1})
    long = build.join_long(fw, obs, Config(), obs.window_end_utc.max())
    wide = build.build_wide(long)
    assert len(wide) == n_days * len(leads)
    assert len(long) == n_days * len(leads) * len(models)
    for m in models:
        assert f"fc_pr_{m}_mm" in wide.columns and f"eligible_pr_{m}" in wide.columns
    assert wide.groupby("lead_group").size().eq(n_days).all()
