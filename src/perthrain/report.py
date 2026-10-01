"""Markdown data-quality report."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def _md(df: pd.DataFrame, floatfmt: int = 3) -> str:
    if df is None or df.empty:
        return "_(none)_\n"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(floatfmt)
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(lines) + "\n"


def closer_gauges_note(st: dict, cfg) -> str:
    closer = st.get("closer_cdo_only_gauges") or []
    if not closer:
        return "_No BoM gauge is closer to the target than the selected station._\n"
    open_ones = [c for c in closer if c.get("open")]
    lines = [f"**{len(closer)} BoM gauge(s) are closer to {cfg.location_name} than the selected station "
             f"but have no automated daily file** — they are only available through Climate Data Online, "
             f"which blocks automated access. Download them manually and pass `--cdo-csv` "
             f"(see README). Full list: `reports/closer_cdo_only_gauges.csv`.\n",
             "| distance | station | name | period | open |", "|---|---|---|---|---|"]
    for c in closer[:8]:
        lines.append(f"| {c['distance_km']:.1f} km | {c['station_id']} | {c['station_name']} | "
                     f"{c['start_year']}–{c['end_year']} | {'yes' if c.get('open') else 'no'} |")
    if open_ones:
        b = open_ones[0]
        lines.append(f"\nNearest **open** alternative: **{b['station_id']} {b['station_name']}** at "
                     f"{b['distance_km']:.1f} km (vs {float(st['distance_km']):.1f} km for the station used).")
    return "\n".join(lines) + "\n"


def window_shift_check(long: pd.DataFrame) -> pd.DataFrame:
    """Correlate observed totals with forecast windows shifted by -1/0/+1 day.

    If the 9am-to-9am 'total to 9am on the label date' convention is right, the
    unshifted pairing should correlate best."""
    rows = []
    for (src, model), g in long[long.lead_group == "day1"].groupby(["source_api", "model"]):
        fc = g.dropna(subset=["forecast_precip_mm"]).set_index("label_date_local").forecast_precip_mm
        ob = g.dropna(subset=["observed_precip_mm"]).drop_duplicates("label_date_local") \
            .set_index("label_date_local").observed_precip_mm
        if len(fc) < 60:
            continue
        rec = {"source_api": src, "model": model}
        for shift in (-1, 0, 1):
            f2 = fc.copy()
            f2.index = pd.to_datetime(f2.index) + pd.Timedelta(days=shift)
            o2 = ob.copy()
            o2.index = pd.to_datetime(o2.index)
            j = pd.concat([f2.rename("f"), o2.rename("o")], axis=1).dropna()
            rec[f"corr_obs_vs_fc_window_shifted_{shift:+d}d"] = round(float(j.f.corr(j.o)), 3)
            rec[f"n_{shift:+d}d"] = len(j)
        rows.append(rec)
    return pd.DataFrame(rows)


def write_report(cfg, summary, obs, long, wide, miss, excl_summary, metrics, failures, cand) -> None:
    st = summary["station"]
    rep = cfg.reports_dir
    probe = {}
    if (rep / "probe_verification.json").exists():
        probe = json.loads((rep / "probe_verification.json").read_text())
    elig = long[long.training_eligible]
    avail = (long.dropna(subset=["forecast_precip_mm"])
             .groupby(["source_api", "model", "lead_group"])
             .agg(first_window=("label_date_local", "min"), last_window=("label_date_local", "max"),
                  windows_with_forecast=("label_date_local", "size"),
                  lead_hours_min=("lead_hours", "min"), lead_hours_max=("lead_hours", "max"),
                  lead_hours_to_window_start=("lead_hours_to_window_start", "median"))
             .reset_index())
    elig_counts = (long.groupby(["source_api", "model", "lead_group"]).training_eligible.sum()
                   .rename("training_eligible_rows").reset_index())
    avail = avail.merge(elig_counts, on=["source_api", "model", "lead_group"], how="outer")
    obs_status = obs.obs_status.value_counts().rename_axis("obs_status").rename("days").reset_index()
    shift = window_shift_check(long)
    cc_cols = [c for c in wide.columns if c.startswith("complete_case_")]
    cc = wide.groupby("lead_group")[cc_cols].sum().reset_index() if cc_cols else pd.DataFrame()

    # Sufficiency verdict
    per = metrics[metrics.comparison == "all_eligible_rows_per_model"] if not metrics.empty else pd.DataFrame()
    cc_rows = metrics[metrics.comparison.str.startswith("complete_case")] if not metrics.empty else pd.DataFrame()

    obs_first, obs_last = obs.label_date_local.min(), obs.label_date_local.max()
    n_obs_ok = int(obs.observed_precip_mm.notna().sum())
    text = f"""# Data-quality report — {cfg.location_name} rainfall forecast verification dataset

Generated: {summary['generated_at_utc']} (pipeline run). All figures below are computed from the files in
`data/`; nothing is estimated or filled.

## 1. Target location and gauge

| item | value |
|---|---|
| Location | {cfg.location_name} ({cfg.latitude}, {cfg.longitude}), time zone {cfg.timezone} |
| Coordinate source | {cfg.coordinate_source} |
| Selected station | **{st['station_id']} {st['station_name']}** ({st['latitude']}, {st['longitude']}) |
| Distance from target | **{float(st['distance_km']):.2f} km** |
| Selection reason | {st['selection_reason']} |
| Station open since (IDCKWCDEA0 stations_db) | {st['db_start_date']} |

{closer_gauges_note(st, cfg)}

Candidates considered (downloadable IDCKWCDEA0 stations within {cfg.search_radius_km} km; completeness =
share of days in the period with a valid 0900–0900 rainfall value):

{_md(cand[['station_id','station_name','distance_km','db_start_date','downloadable','completeness','id_ambiguous','selected']])}
`id_ambiguous=True` means the station number began inside the period (e.g. PERTH AIRPORT 009291 replaced
009021 in March 2026 under the same folder), so the series could mix gauges; such stations are never
auto-selected. The complete list of BoM sites (including manual gauges that are only available through
Climate Data Online) is in `reports/nearby_bom_stations_all_IDCJMC0014.csv`.

## 2. Actual downloaded date range

* Observation label dates: **{obs_first} → {obs_last}** ({len(obs)} days, {n_obs_ok} with a valid value).
  The last day is the latest day already published by BoM; later days are excluded as incomplete.
* Forecast availability per source / model / lead group (only windows with a complete 24-hour forecast):

{_md(avail)}
## 3. Accumulation-window definitions

* **Observation**: BoM daily rainfall labelled date D = total from **09:00 local clock time on D-1 to 09:00 on
  D** (file column header `Rain 0900-0900 (mm)`, verified on every file parsed). For {cfg.timezone} this is
  `window_start_utc = D-1 01:00Z`, `window_end_utc = D 01:00Z` (UTC+8, no daylight saving).
  Every row carries explicit `window_start_*`/`window_end_*` in UTC and local time.
* **Forecast**: Open-Meteo hourly `precipitation` is the accumulation over the hour **ending** at the
  timestamp (verified: the initialisation timestamp of every single run is null). A window
  (start, end] therefore uses the {24} hourly labels start+1h … end. `expected_hour_count` = 24,
  `available_hour_count` = non-null hours; the total is only computed when all 24 are present
  (`forecast_complete`). Partial sums are never reported.
* Joining is done on the exact (`window_start_utc`, `window_end_utc`) pair — never on date strings.
* **Alignment check**: correlation of observed totals with day-1 forecast windows shifted by −1/0/+1 day.
  The unshifted pairing should be the maximum if the window convention is right:

{_md(shift)}
## 4. Lead-time groups and timestamp semantics

| source_api | issue time | lead group definition |
|---|---|---|
| `single_runs` (Single Runs API) | **explicit** run initialisation time (`forecast_issue_time_utc`) | dayN = latest {cfg.single_run_hour_utc:02d} UTC run with `issue + {cfg.publication_latency_hours:g} h ≤ window_start − 24·(N−1) h`. For Perth: day1 = 12 UTC run ~13 h before the 09:00 AWST window start (lead to window end 37 h), day2 = 61 h, day3 = 85 h. |
| `previous_runs` (Previous Runs API) | **not provided** by the source (`issue_time_status = not_provided_fixed_offset…`) | dayN = `precipitation_previous_dayN`, each hourly value predicted 24·N h before its valid time. The contributing run was verified against the Single Runs API for GFS (1 h step), ECMWF 0.25° (3 h) and JMA GSM (6 h) at days 1 and 2 to be `floor_6h( ceil_to_native_step(t) ) − 24·N h` - the offset applies to the END of the native accumulation block, not to the hourly label (100 % of comparable hours matched; see `probe_verification.json` and the unit tests). This *inferred* value is stored in `issue_time_inferred_min/max_utc`; `forecast_issue_time_utc` is left null because the source does not provide it. A 24-h window therefore mixes four consecutive 6-hourly runs. For ACCESS-G the mapping could not be verified (no Single Runs archive) and is labelled `inferred_range_unverified`. |

`lead_hours` = window_end − (latest contributing) issue time; `lead_hours_to_window_start` likewise.
`available_before_window_start_est` assumes a {cfg.publication_latency_hours:g} h publication latency
(Open-Meteo: runs are typically available 4–6 h after initialisation).
With `require_publication_before_window = {cfg.require_publication_before_window}`, previous-runs **day1**
rows are *not* training-eligible: the last run contributing to the window (initialised 1 h before the
window starts) could not have been published before the window began. Day-1 predictors therefore come from
the Single Runs source only.

Probe results (small overlapping test window, `reports/probe_verification.json`):
```
{json.dumps({k: probe.get(k) for k in ['observation_window_example','previous_runs_units','native_step_check','previous_day1_mapping']}, indent=1, default=str)}
```

## 5. Rows, eligibility and missingness

* Long-form rows: **{len(long)}**; training-eligible: **{int(long.training_eligible.sum())}**
  ({100*long.training_eligible.mean():.1f} %).
* Wide rows (station × window × lead group): **{len(wide)}**. Complete-case rows per family:

{_md(cc)}
Observation status by day:

{_md(obs_status)}
Missingness by station / model / lead group (percent of windows in the full period):

{_md(miss)}
Exclusion reasons (a row can have several; full rows in `reports/exclusions.csv`):

{_md(excl_summary)}
Download failures / unavailable runs: **{len(failures)}** (details in `reports/download_failures.csv`).
{_md(failures.groupby(['stage']).size().rename('count').reset_index() if not failures.empty else failures)}
## 6. Observation quality, trace and extremes

* The IDCKWCDEA0 product carries **no quality flags** and is real-time AWS data, so
  `observation_quality_flag = not_provided_unqc_realtime`. BoM Climate Data Online (IDCJAC0009, with
  Y/N quality flags and multi-day accumulation periods) blocks automated access; download it manually and
  pass it with `--cdo-csv` to overlay quality flags for the same station.
* **Trace rainfall** is not encoded in this product (values are reported to 0.2 mm resolution; a trace
  cannot be distinguished from 0.0). `trace_flag` is therefore null. Rain-occurrence metrics use ≥ 0.2 mm
  as "measurable rain".
* Missing values stay null (never converted to zero). Days without a row are `missing_row`.
* Values > {cfg.suspicious_daily_mm:g} mm are kept and flagged `suspicious_extreme_kept`.
* Station metadata changes: station number {st['station_id']} unchanged since {st['db_start_date']}.
  The coordinates are from the current station list; historical site moves are not published in the
  anonymous product.

## 7. Known source limitations

* **Archive floor.** The Previous Runs API refuses `start_date` before **2016-01-01**
  ("Parameter 'start_date' is out of allowed range from 2016-01-01"), so no genuine
  historical forecast is obtainable before then from this source at any lead time. The gauge
  record is much longer (see §2), which is why the clean observation file spans more years
  than the paired dataset.
* **JMA GSM** (`jma_gsm`): the only model with a long archive here - `previous_day1/2` from
  **2016-01-01**, `previous_day3` only from **2019-01-03**. It is a 0.5 deg (~55 km) global
  model with 6-hourly precipitation output, so it is the coarsest model in the set; its hourly
  values are a uniform split of each 6-hour block (see the alignment note below).
* **ECMWF IFS HRES 9 km** (`ecmwf_ifs`): not archived in the Previous Runs API (all offsets null);
  available from the Single Runs API from 2024-03-14. Open-Meteo describes the early archive as
  "IFS Cycle 49R1 hindcasts" — runs before 49R1 became operational (12 Nov 2024) are re-forecasts
  initialised at the historical time, not the forecasts ECMWF disseminated then. From 12 May 2026 06 UTC runs
  use Cycle 50R1.
* **ECMWF IFS 0.25°** (`ecmwf_ifs025`): Previous Runs from 2024-02-03; native 3-hourly precipitation.
* **Uniform disaggregation (ECMWF 0.25° and JMA GSM).** Open-Meteo spreads each native 3-h / 6-h
  total evenly over its hours (probe: 100 % of rainy blocks have identical hourly values). The
  09:00 AWST = 01:00 UTC window boundary does not fall on a 3-h or 6-h step, so each window total
  is `5/6·B1 + B2 + B3 + B4 + 1/6·B5` for a 6-hourly model: a deterministic combination of native
  blocks, but the distribution of rain *within* the two boundary blocks is unknown. These rows carry
  `alignment_status = uniform_disaggregation_boundary_split_{{3,6}}h`. With
  `alignment_policy = "allow_uniform_disaggregation"` (the setting used here:
  **{cfg.alignment_policy}**) they remain training-eligible; with `"strict"` only models whose
  native step divides the window (GFS, ACCESS-G, IFS HRES) are eligible. Filter on
  `alignment_status` if you need the strict subset.
* **BoM ACCESS-G** (`bom_access_global`): Previous Runs archive only from 2024-01-19 to early July 2025
  (no values afterwards at the time of download); not available in the Single Runs API (HTTP 400 for every
  run tested). Its issue-time mapping is unverified.
* **GFS** (`ncep_gfs_global`): Previous Runs from 2024-01-19; Single Runs only from 2026-04-02. A few single
  runs are missing or empty (see failures); those windows are incomplete, not filled.
* **Closer gauges.** BoM sites nearer to the target than the selected station, but without an
  automated daily file, are listed in `reports/closer_cdo_only_gauges.csv`. They need a manual
  Climate Data Online download (`--cdo-csv`), because CDO blocks automated access.
* Previous Runs values are composites of several runs per window (see section 4).
* Forecasts are extracted at the gauge coordinates (Open-Meteo nearest grid cell, `grid_latitude/longitude`
  recorded), not at the target coordinates.
* Open-Meteo free API: non-commercial use, < 10 000 calls/day, CC-BY 4.0 attribution required.

## 8. Baseline checks (training-eligible rows only)

Metrics: MAE, mean error (forecast − observation), RMSE, and contingency scores at thresholds
{cfg.rain_thresholds_mm} mm (event = value ≥ threshold). Full table: `reports/baseline_metrics.csv`.

Per model / lead group (all eligible rows; sample sizes differ between models):

{_md(per[['source_api','model','lead_group','n','first_date','last_date','MAE_mm','mean_error_fc_minus_obs_mm','RMSE_mm','POD_0.2','FAR_0.2','CSI_0.2','CSI_1.0','CSI_5.0']] if not per.empty else per)}
Complete-case comparisons (identical rows for all members) with an equal-weight blend:

{_md(cc_rows[['comparison','lead_group','model','n','first_date','last_date','MAE_mm','mean_error_fc_minus_obs_mm','CSI_0.2','CSI_1.0','CSI_5.0']] if not cc_rows.empty else cc_rows)}
Chronological hold-out (scale factor fitted on the earlier 70 % of dates, evaluated on the later 30 %):

{_md(metrics[metrics.comparison == 'chronological_holdout_scaling'][['model','lead_group','variant','train_first','train_last','test_first','test_last','n_train','n','scale_factor_fitted_on_train','MAE_mm','mean_error_fc_minus_obs_mm']] if not metrics.empty and (metrics.comparison == 'chronological_holdout_scaling').any() else pd.DataFrame())}
{BLEND_NOTE}

## 9. Is the dataset sufficient for a fair comparison?

{sufficiency(long, metrics)}
"""
    (rep / "data_quality_report.md").write_text(text, encoding="utf-8")


BLEND_NOTE = ("Whether the blend beats its members must be read from the complete-case table above on identical "
              "rows; no claim is made beyond those numbers. Differences of a few hundredths of a mm in MAE on "
              "a few hundred days are not evidence of a real difference.")


def sufficiency(long: pd.DataFrame, metrics: pd.DataFrame) -> str:
    e = long[long.training_eligible]
    lines = []
    by = e.groupby(["source_api", "model", "lead_group"]).size()
    for (src, model, lg), n in by.items():
        wet = int((e[(e.source_api == src) & (e.model == model) & (e.lead_group == lg)].observed_precip_mm >= 1).sum())
        lines.append(f"* `{src}/{model}/{lg}`: {n} eligible windows, {wet} with observed rain ≥ 1 mm.")
    if metrics is None or metrics.empty:
        return "No verified pairs: not sufficient."
    cc = metrics[metrics.comparison.str.startswith("complete_case") & (metrics.model != "INSUFFICIENT")]
    verdict = []
    verdict.append("* **Within-source comparisons are fair only on the complete-case rows.** "
                   "Previous-runs ACCESS-G vs GFS overlap only while ACCESS-G was archived "
                   "(Jan 2024 – Jul 2025), day2/day3 only.")
    verdict.append("* **Single-runs ECMWF HRES** has the longest, cleanest record with explicit issue times "
                   "(all three lead groups) and is suitable for calibration experiments on its own.")
    verdict.append("* **Cross-source comparisons** (single-runs vs previous-runs) are indicative only: the lead "
                   "definitions differ (single run vs composite of four runs).")
    verdict.append("* Heavy-rain thresholds (≥ 5 mm) have few events per lead group; treat those scores as "
                   "high-variance.")
    return "\n".join(lines + [""] + verdict)
