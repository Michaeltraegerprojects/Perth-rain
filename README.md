# perthrain — local rainfall forecast-verification dataset (Perth, WA)

Builds a machine-learning-ready dataset that pairs **rainfall forecasts issued before an event** with
**rainfall measured afterwards by a BoM rain gauge**, on exactly matching accumulation windows.
Intended for local calibration: probability of measurable rain, amount, uncertainty, lead-time skill.

Ships with four configured locations — **Perth, Clarkson, Ocean Reef, Fremantle** — each producing its
own dataset under `data/<slug>/` and `reports/<slug>/`.

**How far back the data actually goes.** These are two different limits, and the shorter one binds:

| | earliest | source of the limit |
|---|---|---|
| Gauge observations | **2009-01** (~17 years) | BoM FTP product IDCKWCDEA0 starts then; CDO goes back further (manual) |
| Archived forecasts at genuine lead times | **2016-01-01** (~10.7 years) | Open-Meteo Previous Runs API hard limit: *"Parameter 'start_date' is out of allowed range from 2016-01-01"* |

Only **JMA GSM** reaches 2016 (day-3 lead only from 2019-01-03). GFS, ECMWF 0.25° and ACCESS-G start in
early 2024; ECMWF IFS HRES single runs start 2024-03-14. So the pipeline downloads **observations back to
2009** (the clean observation file is the full ~17 years) and **pairs them with forecasts from 2016**.
There is no accessible archive of genuine 17-year-old local rainfall forecasts — reanalysis such as ERA5
goes back to 1940 but is *not* a forecast and is deliberately not used here.

## Setup (Windows / PowerShell or bash; Python ≥ 3.11)

```bash
python -m venv .venv
```
```bash
.venv/Scripts/python -m pip install -r requirements.txt
```
```bash
.venv/Scripts/python -m pip install -e .
```

> Windows note: if the project lives under a very long path, `pyarrow` can fail with
> "The filename or extension is too long". Create the virtualenv at a short path instead
> (e.g. `python -m venv C:\venvs\perthrain`) and use that interpreter in the commands below.

On macOS/Linux use `.venv/bin/python` instead of `.venv/Scripts/python`.

**Recreating the environment used for the current results.** The active models and predictions in this project
were produced with Python 3.12.11 in the project's own `.venv`. That venv was created by
`uv`, with `requirements.txt` plus an editable install of this package. The exact dependency versions of each run
are stored in `data/<slug>/models/manifest.json` under `meta.versions`. To rebuild it from the project root:

```bash
uv venv .venv --python 3.12
```
```bash
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```
```bash
uv pip install --python .venv/Scripts/python.exe -e .
```

(`python -m venv` and `pip` as above work too.) Then confirm with the unit tests, which need no local data:
`.venv/Scripts/python -m pytest -q -m "not local_data"` (see [Tests](#tests)).

## Run

```bash
cp config.example.toml config.toml
```
```bash
.venv/Scripts/python -m perthrain stations --config config.toml
```
```bash
.venv/Scripts/python -m perthrain probe --config config.toml
```
```bash
.venv/Scripts/python -m perthrain run --config config.toml
```
```bash
.venv/Scripts/python -m pytest -q
```

Every command runs all configured locations in turn; `--location Clarkson` runs just one.

* `stations` — lists nearby BoM stations, their completeness and the auto-selected gauge, per location.
* `probe` — small overlapping test window (default 2026-06-01..14): checks units, the 0900–0900 observation
  window, end-labelled hourly forecast accumulations, the Previous-Runs run mapping and native 3-hourly
  output. Writes `reports/probe_verification.json`. Run this before the full download.
* `run` — full download (cached/resumable), clean, join, QC, baseline metrics and reports. The Single Runs
  download is ~1 request per model run (~1 100 requests at ≤ 1 request/s ≈ 20–30 min on first run;
  seconds on re-runs). `--skip-single-runs` for a fast partial build.

Re-running any command never re-downloads a completed request (responses are cached in `data/raw/`).
Only the two most recent monthly observation files are refreshed, because BoM is still appending to them.

### Gauge distances (the honest picture)

| location | auto-selected gauge (automated) | distance | nearest BoM gauge of any kind |
|---|---|---|---|
| Perth | 009225 PERTH METRO | 3.8 km | same |
| Fremantle | 009215 SWANBOURNE | 11.3 km | 009192 FREMANTLE, 1.8 km (CDO only, open) |
| Ocean Reef | 009225 PERTH METRO | 22.0 km | 009265 HILLARYS BOAT HARBOUR, 7.4 km (CDO only, open) |
| Clarkson | 009178 GINGIN AERO | 27.8 km | 009264 TAMALA PARK (MINDARIE), 2.8 km (CDO only, open) |

The automated FTP product only carries AWS sites that report a full weather record. Rainfall-only gauges —
which are the closest ones for the northern suburbs — exist but are served solely by Climate Data Online,
which blocks automated access. For Clarkson and Ocean Reef the auto-selected gauge is **20–28 km away and
in a different local rainfall regime**; treat those two datasets as provisional until you add the CDO CSV
for a nearby gauge (below). Each run writes `reports/<slug>/closer_cdo_only_gauges.csv`.

### Change the location or station

```bash
.venv/Scripts/python -m perthrain run --config config.toml --name "Fremantle" --lat -32.0569 --lon 115.7439 --coord-source "your source here"
```
```bash
.venv/Scripts/python -m perthrain run --config config.toml --station 009215
```

Other options: `--radius KM`, `--start YYYY-MM-DD`, `--end YYYY-MM-DD`, `--timezone`, `--data-dir`,
`--reports-dir`, `--cdo-csv FILE` (repeatable). Use a separate `--data-dir/--reports-dir` per location.
The pipeline never silently substitutes another station into missing days.

### Adding BoM quality-controlled observations (manual step)

BoM Climate Data Online (product IDCJAC0009, which has Y/N quality flags and multi-day accumulation
periods) **blocks automated access** (HTTP 403 "automated access request"). This pipeline does not try
to get around that. To add those flags:

1. In a browser, open BoM Climate Data Online → *Rainfall* → *Daily* → station number (e.g. 009225).
2. Download "All years of data" (zip) and extract the `IDCJAC0009_<station>_1800_Data.csv`.
3. Run with `--cdo-csv path/to/IDCJAC0009_009225_1800_Data.csv` (or list it in `station.cdo_csv_paths`).

The loader refuses files for a different station. CDO values take precedence, disagreements with
the FTP value are flagged (`ftp_cdo_value_conflict_cdo_used`), and multi-day accumulations get their
real window (they are excluded from 24-hour training rows).

## Sources (verified 2026-09-30)

| data | source | access |
|---|---|---|
| Observed daily rain | BoM anonymous FTP `ftp://ftp.bom.gov.au/anon/gen/clim_data/IDCKWCDEA0/tables/<state>/<station>/<station>-YYYYMM.csv` (column `Rain 0900-0900 (mm)`) | free, BoM copyright terms; FTP is the channel BoM points automated users to |
| Station lists | `…/IDCKWCDEA0/tables/stations_db.txt`; `ftp://ftp.bom.gov.au/anon2/home/ncc/metadata/sitelists/stations.zip` (IDCJMC0014) | free |
| Forecasts at fixed lead offsets | Open-Meteo **Previous Runs API** `https://previous-runs-api.open-meteo.com/v1/forecast`, `precipitation_previous_day{1,2,3}` | free non-commercial, CC-BY 4.0; archive floor 2016-01-01 |
| Forecasts with explicit run time | Open-Meteo **Single Runs API** `https://single-runs-api.open-meteo.com/v1/forecast`, `run=YYYY-MM-DDTHH:MM` | free non-commercial, CC-BY 4.0 |
| Target coordinates | GeoNames 2063523 via Open-Meteo Geocoding API | the Geoscience Australia Gazetteer WFS returned HTTP 400/403 at build time |

Model identifiers used (explicit selections, never "best match") and their measured archive start at Perth:

| model | source | precipitation archive | native step |
|---|---|---|---|
| `jma_gsm` | Previous Runs | **2016-01-01** (day3 from 2019-01-03) | 6 h, 0.5° |
| `ncep_gfs_global` | Previous Runs | 2024-01-19 | 1 h |
| `ecmwf_ifs025` | Previous Runs | 2024-02-03 | 3 h |
| `bom_access_global` | Previous Runs | 2024-01-19 → **2025-07-06 only** | 1 h |
| `ecmwf_ifs` (IFS HRES 9 km) | Single Runs | 2024-03-14 | 1 h |
| `ncep_gfs_global` | Single Runs | 2026-04-02 | 1 h |

Models checked and rejected for this purpose: ACCESS-G has no Single Runs archive (HTTP 400 for every run
tested); UKMO global starts 2024-08-08; GEM, ICON, ARPEGE and CMA GRAPES all start 2024-01-19, i.e. none
extends the record. JMA MSM is Japan-only.
The **Historical Forecast API** was *not* used for predictors: it stitches the first hours of successive
runs into one series, so it has no genuine lead time. No reanalysis (ERA5 etc.) is used anywhere.

Rate limiting: ≤ 1 request/s, 2 concurrent workers, exponential backoff with jitter on timeouts/429/5xx,
`Retry-After` honoured. The free Open-Meteo tier allows < 600/min, < 5 000/h and < 10 000/day.
No credentials are needed; none are stored.

## Time alignment

* Observation dated D = rain from **09:00 local on D-1 to 09:00 local on D** (BoM convention; column
  header `0900-0900` checked on every file). Perth: 01:00 UTC → 01:00 UTC. **The label date is the END of the
  window**: the forecast for "3 Oct" is rain from 09:00 on 2 Oct to 09:00 on 3 Oct, local time. The same labels are
  used in `label_date_local` of every prediction.
* Forecast hourly `precipitation` at timestamp t = rain during (t−1h, t] (verified: the init hour of each
  single run is null). A window uses the 24 hourly labels start+1h … end.
* Joins use the exact `(window_start_utc, window_end_utc)` pair. A total is only produced from all 24
  hours; otherwise `forecast_precip_mm` is null and the row is ineligible.
* Lead groups: see `reports/data_quality_report.md` §4. In short, **single_runs dayN** = the 12 UTC run
  published (with an assumed 6 h latency) at least 24·(N−1) h before the window starts. **previous_runs dayN**
  = Open-Meteo's `previous_dayN`, a composite of four 6-hourly runs initialised 24·N h (to 24·N+5 h) before
  each hour. It has no source issue time, so `forecast_issue_time_utc` is null and an inferred range is stored.
  Previous-runs day1 is not training-eligible because its latest run could not be published before the
  window started (configurable: `require_publication_before_window`).
* The inferred run for previous-runs data is `floor_6h( ceil_to_native_step(valid_end) ) − 24·N h`,
  verified against the Single Runs API for GFS (1 h), ECMWF 0.25° (3 h) and JMA GSM (6 h) at days 1 and 2:
  **100 % of comparable hours matched**. The offset applies to the end of the native accumulation block,
  not to the hourly label.
* For 3 h / 6 h models the 01:00 UTC window boundary splits a native block. Open-Meteo spreads each block
  uniformly over its hours (verified), so the window total is a deterministic combination of native blocks
  with the two boundary blocks' internal distribution unknown. Those rows carry
  `alignment_status = uniform_disaggregation_boundary_split_{3,6}h` and stay training-eligible under the
  default `alignment_policy = "allow_uniform_disaggregation"`; set it to `"strict"` to drop them.

## Outputs

```
data/<slug>/...                one dataset per configured location (perth, clarkson, ocean_reef, fremantle)
reports/locations_summary.csv  index across all locations
data/raw/                      raw responses, cached once and shared by every location (+ .meta.json with URL, params, retrieval time, sha256)
data/raw/manifest.jsonl        every network retrieval; failures.jsonl = failed/unavailable requests
data/clean/observations.{csv,parquet}       one row per gauge day with explicit windows + status
data/clean/forecasts_hourly.{csv,parquet}   tidy hourly forecast records (both sources)
data/clean/forecast_windows.{csv,parquet}   forecasts aggregated to the observation windows
data/joined/paired_long.{csv,parquet}       tidy long-form pairs (schema below)
data/joined/training_wide.{csv,parquet}     one row per station + window + lead group
reports/data_quality_report.md               the data-quality report
reports/provenance_manifest.json             sources, terms, run summary, every retrieval
reports/download_failures.csv, exclusions.csv, exclusion_summary.csv, missingness.csv,
reports/observation_issues.csv, forecast_conflicting_duplicates.csv, station_candidates.csv,
reports/nearby_bom_stations_all_IDCJMC0014.csv, closer_cdo_only_gauges.csv,
reports/baseline_metrics.csv, probe_verification.json
logs/pipeline.log
```

Long-form columns: `location_name, target_latitude, target_longitude, station_id, station_name,
station_latitude, station_longitude, source_api, model, forecast_issue_time_utc, issue_time_status,
issue_time_inferred_min_utc, issue_time_inferred_max_utc, source_lead_offset, lead_hours,
lead_hours_to_window_start, lead_group, label_date_local, window_start_utc, window_end_utc,
window_start_local, window_end_local, forecast_precip_mm, observed_precip_mm, expected_hour_count,
available_hour_count, forecast_complete, alignment_status, available_before_window_start_est,
observation_quality_flag, obs_status, training_eligible, exclusion_reason, grid_latitude, grid_longitude,
source_url, retrieved_at_utc`.

Wide table: `fc_<sr|pr>_<model>_mm` (sr = single runs, pr = previous runs), `available_*`, `eligible_*`,
`lead_hours_*`, `observed_precip_mm`, `observation_quality_flag`, `obs_status`, `month`, `season`
(southern-hemisphere DJF/MAM/JJA/SON), `n_eligible_*_models`, `complete_case_pr`, `complete_case_sr`.
Missing model predictions are **not** filled; use `complete_case_*` for fair comparisons.


## Calibration and predictions of future rain

After `run` has built a location's dataset:

```bash
.venv/Scripts/python -m perthrain calibrate --config config.toml
```
```bash
.venv/Scripts/python -m perthrain predict --config config.toml --horizon-days 4
```
```bash
.venv/Scripts/python -m perthrain scan-enso --config config.toml --stage current
```

Add `--location Clarkson` to do one place.

**`calibrate`** fits, per gauge and lead group (day1/2/3), a two-part model on the paired data:

1. *occurrence*: logistic regression for P(rain >= 0.2 mm);
2. *amount given rain*: quantile regression of log(mm) at 19 levels (5 % ... 95 %);
3. these combine into exceedance probabilities (>= 0.2, 1, 5, 10 mm), the 10/25/50/75/90 % quantiles of the
   total (dry outcomes count as 0), and an expected total.

Features (the explicitly approved baseline set) are the forecast rainfall of ONE model or one composite (mean of
log(1+mm)), plus optionally a seasonal cycle. The candidates are ECMWF IFS (single runs), ECMWF IFS 0.25, JMA GSM, an
ECMWF composite and an all-model composite. **Validation is chronological**: the last 20 % of dates are never used
for fitting or selection. Configurations are chosen on three expanding-window folds with a 3-day gap, the held-out
block is scored once, and the deployed model is refitted on all data only afterwards.

* **Equal folds.** Every configuration is scored on the *same* folds. A fold that any configuration cannot score is
  dropped for all of them, and the folds used are recorded (`cv_folds`). A fold whose training data has too few
  wet or dry days is skipped, not fatal. A candidate whose dates barely overlap the others' is excluded from
  selection and named in the report.
* **Identical held-out rows.** Every compared method (calibrated, both climatologies, each raw model, equal-weight
  blend) is scored on the same station / accumulation window / observed target. This is checked by a SHA-256
  fingerprint stored with each metrics row (`key_fingerprint`, `keys_identical_across_methods`). If the row sets
  ever differ, the comparison is recomputed on their intersection.
* **References.** *Season-aware climatology* uses the same calendar month ±1 from the training dates only; it is
  the reference for every skill score. *Season-blind climatology* is reported alongside.
* **Point forecasts.** The calibrated **median** and **expected total** are scored separately (MAE, signed bias =
  forecast − observed, RMSE).
* **Probability scores.** Brier scores use events defined with `>=`: an observed 0.2 mm counts as rain. Raw model
  forecasts are amounts, so no probability comparison with them exists.
* **Uncertainty.** `paired_differences.csv` gives, per lead, the per-day difference between the calibrated forecast
  and each reference, with a 95 % circular block bootstrap (7-day blocks, 2000 resamples, seed 0). An interval
  that includes zero means the ranking is descriptive, not significant.

Outputs per location:

* `reports/<slug>/calibration_report.md` (verdicts computed from held-out scores), `calibration_metrics.csv`,
  `paired_differences.csv` and `routing_table.csv`;
* `data/<slug>/models/hurdle_day{1,2,3}.joblib` plus `manifest.json` and `selection.json`.

**Replacing models safely.** `calibrate` builds every lead in `data/<slug>/staging_<run_id>/`, writes it, reloads it
and validates it. Only then does it move the previous `models/` to `archive/superseded_models/<slug>/` (never
overwriting) and move the new files in. If any lead fails, the staging directory is deleted and the existing models
are left untouched. The default (no-ENSO) run drops any `clim_*` columns before fitting.

**Artifact provenance.** Every artifact carries a `run_id` (for example `noenso-20260930T145237Z-701d2d9d`),
`created_utc`, `enso_status = "no_enso"`, the ordered `feature_schema` of every fit, a SHA-256 of the package source
(`code_sha256`; the project is not a git repository, so there is no commit id) and the SHA-256, row count and date
range of the clean inputs. It also records the dependency versions and the fallback `routing_order`. `manifest.json`
repeats all of this with each artifact's SHA-256. A new `calibrate` never overwrites artifacts: it first moves the
previous ones to `archive/superseded_models/<slug>/`.

**`predict`** builds live features exactly as in training: the same sources, lead-group definitions, 9am-9am
windows, and completeness and publication rules. It writes `data/<slug>/predictions/forecast_<UTC time>.csv|md` and
a `latest` copy. Every row carries `run_id`, `artifact_sha256`, `enso_status`, the model used and its route:

* `primary`: the cross-validation-selected model;
* `fallback#k`: the k-th model in the routing order, used when the primary's inputs are incomplete;
* `exhausted` / `no_model_artifact`: **nothing** is served. No uncalibrated raw-forecast value is ever substituted.

The NWP totals that went into a served row are included as `input_<model>_mm` columns. They are provenance, not
forecasts.

**As-of timing gate (live inputs).** For valid times in the future, the Previous Runs API returns values even when
the run its offset rule names has not yet been issued. In that case the value comes from a newer run, so its lead
time differs from what the model was trained on. An input is therefore used only if:

* its latest contributing run, plus the 6 h publication latency, is at or before the **prediction time**, not
  merely before the window start;
* the window has all 24 hours;
* no forecast hour is negative;
* its alignment passes the configured `alignment_policy`.

These are the same exclusions as in training. Rows whose inputs fail the gate are reported with
`inputs_blocked_by_timing` and served as `exhausted` unless a fallback model's inputs pass. The rule was audited on
the historical training and held-out rows: 0 of ~40 000 eligible rows fail it (`reports/historical_timing_audit.csv`),
so the held-out metrics stand.

`--frozen-inputs --as-of <UTC time>` replays a past prediction **offline** from the cached live inputs. A cache miss
is reported, never downloaded. `--as-of` without `--frozen-inputs` is rejected.

**Verifying served rows.** `scripts/verify_served_rows.py` (diagnostic) labels each input of each served row:

* `sr_*` inputs come from an explicit `run=` request, so they are **VERIFIED** by the request itself;
* `pr_*` inputs are compared hour by hour with the run the offset rule names, fetched from the Single Runs archive.
  If all hours are equal the input is VERIFIED; if a named run is not archived it is **UNVERIFIED**; any mismatch
  is FAILED.

A row is fully verified only if every input is. Results go to `reports/served_row_verification.csv` and
`served_row_status.csv`.

**Serving loader checks.** An artifact is refused unless it:

* is a regular, unshared file (no extra hard links) whose resolved path is directly inside `data/<slug>/models/` and
  not inside the resolved `<project>/archive` (no `..` traversal, file symlinks or linked directories; junctions are
  detected from the file's reparse-point attribute, so the check also works on Python 3.11);
* is not byte-identical to any file listed in a quarantine manifest or to any `.joblib` anywhere under `archive/`
  (the bytes that are hashed are the bytes that are loaded);
* can be unpickled;
* was trained for this location and gauge (`meta.location`, `meta.gauge_data_dir`);
* has complete, correctly typed metadata;
* contains a real, fitted `HurdleModel` in every fit;
* has, for every fit, a stored ordered feature schema that equals what the code builds from that fit's own
  parameters, with every estimator expecting exactly that many inputs;
* contains only approved baseline features.

The prediction is for the **gauge**. For Clarkson and Ocean Reef the auto-selected gauge is 22-28 km away (see
"Gauge distances"); use `--cdo-csv` with a closer gauge for a better local answer.

### El Nino / Indian Ocean Dipole: not used (unverified)

All active models are **no-ENSO** (`enso_status = no_enso`). A climate-index feature (BoM's traditional Niño3.4 and
IOD files) was tried and then withdrawn. Its definition could not be confirmed from BoM's own documentation, which
blocks automated access. Its historical values may also have been revised after first publication, and the
original join used the gauge date rather than the forecast issue time. See `reports/enso_source_verification.md`,
which lists confirmed facts, documentation claims and hypotheses separately, together with a corrected comparison
against NOAA CPC.

* The prohibition covers the unverified ENSO inputs (Niño3.4, relative Niño3.4, SOI), the IOD SST-anomaly index used
  with them, and any feature derived from them. **JMA GSM rainfall is a legitimate forecast input and is used.**
* Other climate indices (MJO, SAM, ...) are not named by the prohibition, but they are **not approved features**
  either: the guard accepts only the approved baseline set.
* `--enso-experimental` exists only to investigate the feature. Its artifacts and predictions go to
  `models_enso_experimental/` and `predictions_enso_experimental/` and can never be loaded by the default
  predictor.
* No claim is made that El Niño causes local drying or that the index improves these forecasts.

### Archives

* `archive/INVALIDATED_20260930T1437Z_enso_contaminated/`: model artifacts, predictions and reports moved out of
  service on 2026-09-30, **invalidated pending verification** (not proven numerically wrong), with a hash manifest and
  the live inputs they used. See its README.
* `archive/FORENSIC_post_interruption_snapshot_20260930T2235/`: copies taken after an interrupted job. It is a
  **post-interruption forensic snapshot, not a restore point**, and it contains items that were later invalidated.
* `archive/WITHDRAWN_20260930T1522Z_pre_timing_fix_predictions/`: predictions served before the as-of timing gate
  existed, with the cached live inputs they used. `row_status.csv` marks each row:
  * WITHDRAWN_TIMING_FAIL (20): an input came from a run not issued by the prediction time;
  * TIMING_PASS_REISSUED (12);
  * NOT_SERVED (16).
* `archive/superseded_models/<slug>/models_before_<run_id>/`: every earlier model set, moved aside (never
  overwritten) by the next `calibrate`. These files are hash-blocked from serving.
* `archive/superseded_predictions/<slug>/`: prediction records made with superseded models. They are kept rather
  than overwritten.
* `reports/rebuild_before_after.md` compares the withdrawn predictions with the rebuilt models **on identical cached
  inputs**. It measures the software change, not forecast skill.

## Tests

Tests are split by what they need. **Unit tests** run on a fresh clone. **Integration tests** (marker `local_data`)
check the locally built `data/`, `reports/` and `archive/` folders, which are never committed.

Unit tests (Python, then the website's JavaScript; Node 18+):

```bash
.venv/Scripts/python -m pytest -q -m "not local_data"
```
```bash
npm --prefix website test
```

Integration tests, after `run`, `calibrate`, the challenge `historical` step and a website export have built the
local data:

```bash
.venv/Scripts/python -m pytest -q -m local_data
```

Everything (both groups; this is what a release is checked with):

```bash
.venv/Scripts/python -m pytest -q
```

`pytest` collects `tests/`, `challenge/tests/` and `website/tests/`. A regression test copies only the tracked files
to a temporary folder and runs the unit selection there, so a test that quietly needs local data fails.

* `test_transformations.py`: windows (including DST), header and unit validation, missing ≠ zero,
  negative/extreme/duplicate handling, the CDO station guard, inferred Previous-Runs issue times, end-labelled
  hours, the no-partial-totals rule, alignment flags, publication-before-window, exact-window joins and metrics.
* `test_calibration.py`: distribution maths, index lag/no-look-ahead helpers, and future windows.
* `test_enso_guard.py`: what the no-ENSO guard rejects (climate fits, interactions, hidden columns, reordered
  schemas, missing or incompatible metadata, unapproved features, corrupt/foreign/mislabelled artifacts, quarantined
  paths through traversal, symlinks and junctions, byte-identical copies) and allows (JMA rainfall, a project under a
  folder named "archive"). It also checks every **active** artifact and prediction payload.
* `test_routing.py`: all 16 model-availability combinations for each lead, run through the real `predict_windows`
  against **hard-coded** expected routes (a mutation test confirms that reversing the routing order fails). On
  exhaustion every numeric output column must be empty. For each **active** artifact set (locations read from
  `config.toml`; a missing artifact fails rather than skips), the written routing table must match. An **offline**
  replay through the real `compute_predictions` must reproduce the newest served `latest.csv` exactly (`rtol=0`).
* `test_live_timing.py`: the as-of gate. Previous-runs inputs whose named runs were not issued and published by
  the prediction time are unusable and reported. Every served row of every location used only runs issued before
  its prediction time.
* `test_review_regressions.py`: one test per confirmed code-review finding. These cover:
  * equal CV folds, skipping a fold with too few wet days, and candidates whose dates don't overlap;
  * identical held-out keys and the fingerprint;
  * seasonal climatology, the bootstrap, and separate median/expected-total metrics;
  * staged model replacement on failure and on success, and that climate columns are ignored by default;
  * refusing artifacts from another location, a project under a folder named "archive" loading correctly, and every
    metadata field and type;
  * missing or unfitted models, directories posing as artifacts, hard links, and junction/symlinked model folders;
  * BOM-tolerant hashes, blocking of every archived file, and the scan's "active" flag;
  * validating caller-supplied bundles, no-window horizons, and empty inputs;
  * excluding negative forecast hours, records that are never overwritten with UTC names, offline fetching, and
    `input_*` naming.
* `test_noaa_cpc.py`: the diagnostic-only NOAA CPC parser, tested on a committed verbatim excerpt of the CPC file
  (`tests/data/`, with the full file's SHA-256). It covers manually checked rows (glued negative pairs such as
  `20.6-0.1`, positive pairs and a `-0.0` anomaly). Every possible single-character shift of a row must raise, as
  must an impossible date. A static test ensures no package module imports it.

## Diagnostic scripts (not part of the package)

`scripts/` holds audit and report builders. They read package outputs or the public APIs; models are never trained
or served from here.

* `acceptance_report.py` writes `reports/acceptance_report.md`: held-out results per gauge and lead, row-identity
  checks, paired intervals, and which review findings affected this run.
* `verify_served_rows.py`: the served-row provenance check described above.
* `historical_timing_audit.py`: applies the as-of rule to every training and held-out row.
* `live_provenance_audit.py`: documented API semantics vs observed live responses.
* `rebuild_before_after.py`: frozen-input comparisons. These measure software changes only.
* `enso_cpc_compare.py`: the NOAA comparison.

Attribution: weather forecast data by [Open-Meteo.com](https://open-meteo.com/) (CC-BY 4.0), based on
ECMWF, NOAA NCEP and BoM model output; observations © Commonwealth of Australia, Bureau of Meteorology.

## Website

`website/` holds a static site (no framework, no build step) that shows the forecasts and held-out performance from a
reviewed JSON export (`website/data/v1/`). It is separate from training and model loading: its exporter reads only
output tables, never model files. See [website/README.md](website/README.md) for the local preview, tests, the
export-and-approve refresh process and hosting options. Nothing is deployed or scheduled automatically.

## Repository contents

The git repository tracks code, tests, scripts, configuration, documentation and the website with its approved
export. `data/`, `archive/`, `reports/`, `logs/` and model files are excluded by `.gitignore`: they are large,
regenerable or forensic. The data-dependent tests in `tests/` therefore need a locally built dataset and fail
(deliberately, not skip) in a fresh clone. `website/tests` runs anywhere.
