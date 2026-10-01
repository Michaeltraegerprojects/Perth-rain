# Perth rain project: what was built, what went wrong, what was fixed

Status as of 1 Oct 2026.

## What was built

**1. A dataset of forecasts paired with gauge readings.** The Python package `perthrain` pairs archived weather-model
rain forecasts with Bureau of Meteorology rain-gauge readings. Every pair covers the exact same 24-hour window,
9am to 9am, and the label date is the END of that window.

- **Observations.** Daily gauge readings from 2009, taken from BoM's public monthly files. BoM's Climate Data
  Online blocks automated access, so it was not bypassed. Instead there is a loader for CSVs downloaded by hand.
- **Forecasts.** Open-Meteo's archived model runs (Previous Runs and Single Runs APIs) for ECMWF IFS, ECMWF IFS 0.25°,
  JMA GSM and NOAA GFS. JMA goes back to 2016; the ECMWF-based sets start in 2024.
- **Locations and gauges.**

  | Location | Gauge | Distance |
  |---|---|---|
  | Perth | Perth Metro 009225 | 3.8 km |
  | Ocean Reef | Perth Metro 009225 (same gauge) | 22 km |
  | Clarkson | Gingin Aero 009178 | 27.8 km |
  | Fremantle | Swanbourne 009215 | 11.3 km |

- **Pipeline rules.** Downloads are cached and resumable. Forecast and gauge windows are joined on exact UTC
  times, and a window total is only produced when all 24 hours are present. The pipeline also does quality
  checks, writes long and wide tables, and produces data-quality reports.

**2. A calibration model.** For each gauge and lead time (1, 2 and 3 days ahead) the model predicts:

- the chance of rain (≥ 0.2, 1, 5 and 10 mm);
- likely amounts (10th to 90th percentile);
- an expected total.

Testing is strictly chronological. The last 20 % of days are held out, and models are chosen on earlier folds only.
The deployed model can fall back to a backup forecast model when its first-choice inputs are missing.

**3. Live predictions.** These use the same rules as training. A forecast is only issued if every model run it
needs was published before the prediction time. Each row records which models and runs it used, the model version,
and whether its inputs could be checked against the archive.

**4. Safeguards and provenance.**

- Every model file carries a run ID, code and data hashes, and dependency versions.
- Old files are moved to `archive/` and never overwritten.
- The model loader refuses files that:
  - are quarantined or byte-copies of archived ones;
  - are reached through links or junctions;
  - belong to another location;
  - are missing metadata;
  - use features outside the approved list.
- A no-El-Niño guard and scanner make sure no unverified climate index can reach a forecast.

**5. Tests and reports.**

- 375 automated tests.
- Diagnostic audit scripts.
- Per-location calibration reports.
- An acceptance report and a static HTML report
  (`reports/perth_rain_report.html`).
- A README with setup, rerun commands and the rules above.

## Results

- **Against average weather for the time of year:** the calibrated forecast is clearly and significantly better at
  every gauge and lead time, on roughly 180 unseen days (March to September 2026).
- **Against the raw weather models and their blend:** most differences in rain amount are within noise. In 24
  comparisons it was significantly better 3 times and worse none. No general gain over the raw models is claimed.
- **Forecast on 1 Oct:** dry. There is about a 14–16 % chance of measurable rain on Saturday and Sunday, and
  expected totals are under 0.2 mm.

## Issues found and how they were fixed

| Issue | Impact | Fix |
|---|---|---|
| BoM Climate Data Online blocks automated downloads (HTTP 403) | Couldn't use it directly | Not bypassed. Used BoM's public monthly files, plus a loader for hand-downloaded CSVs |
| pyarrow failed on a long Windows path | Install failed | Virtual environment at a short path |
| A table pivot built every possible combination | Memory blow-up | Rewritten, with a regression test |
| Overlapping cached download chunks | Possible duplicate rows | Checked them identical, then removed duplicates |
| El Niño feature added on request; its source definition could not be verified | Unverified input in models | Withdrawn. Models rebuilt without it; a guard and scanner added |
| El Niño models reached live forecasts through a backup (day-3 JMA) route | 2 rows each for Perth and Ocean Reef | Those outputs moved to a quarantine archive ("invalidated pending verification") and all models rebuilt. I first said Clarkson was unaffected; it held an unused El Niño fit, which I corrected |
| Diagnostic NOAA file parser read sea temperature as the anomaly | Wrong comparison numbers (diagnostic only, never in models) | Fixed-width parser, tested on a verbatim excerpt of the real file |
| The guard's word matching missed names like `soi_30d` and `sst34`, and flagged `sensor` | Gaps and false alarms | Letter-boundary matching, tested on both lists |
| **Live forecasts used model runs not yet issued.** The API fills future hours from newer runs, and the code only checked the window start | Wrong lead time | Timing rule added: only runs published before the prediction time count. 20 rows withdrawn, 12 reissued. Training data passed the same rule (0 of 58,768 rows), so test scores stand |
| I inferred "latest-run substitution" from mostly zero-rain values | Unsupported claim | Withdrawn, then re-tested on non-zero hours only |
| Live forecasts kept windows containing negative forecast hours, which training excludes | **Changed 3 served rows** on 30 Sep (chance of rain 7 % → 14 %) | Same exclusion as training |
| Models were compared on different cross-validation days | Unfair model choice (did not occur here) | All candidates now scored on the same folds |
| A fold with too few wet days crashed calibration | Could abort a run (did not occur here) | Fold skipped and logged |
| Old models were moved aside before new ones were finished | A failure could leave a half-replaced set (did not occur here) | Build in a staging folder, validate, then swap |
| Skill measured against a climatology that ignored the season | Shifted skill scores by −1.0 to +0.5 points | Season-aware climatology; both reported |
| Median and expected total were mixed in one error summary | Misleading table | Each now scored separately (MAE, bias, RMSE) |
| Output files had `raw_*` columns that looked like forecasts | Confusing | Renamed `input_*` (provenance only) |
| Junction detection only worked on Python 3.12+ | Guard blind on 3.11 | Detected from file attributes |
| Model files from another location would load | Wrong-gauge risk (did not occur) | Location check on every file |
| Edge cases crashed: no upcoming days, empty fetches, `--as-of` without frozen inputs, local-time file names | Crashes or ignored options | Handled and tested |
| Some tests could pass for the wrong reason (checked names only, compared routing to itself, skipped when data was missing) | Weak assurance | Rewritten with hard-coded expectations and value checks; missing data now fails |

Every confirmed review finding has its own regression test (`tests/test_review_regressions.py`).
None of the fixes changed which model was chosen at any gauge or lead time.

## Known limits

- Forecasts describe the gauge, not the suburb. Clarkson and Ocean Reef gauges are 22–28 km away.
- Perth and Ocean Reef share one gauge, so they are one test sample, not two.
- About 180 held-out days with few heavy-rain events, so heavy-rain probabilities are less certain.
- Some live rows stay **unverified** when a model run they used isn't in the public archive (JMA 30 Sep 06Z).
- Forecasts only appear for days whose required model runs are already published. Rerun `predict` to fill in
  later days.
- El Niño and Indian Ocean Dipole are not used.
- The project is not a git repository, so code versions are tracked by SHA-256 hash.

## Where things are

- `src/perthrain/`: package code.
- `tests/`: 375 tests.
- `scripts/`: diagnostic audits and report builders.
- `data/<location>/`:
  - `models/`: active models;
  - `predictions/`: forecasts;
  - `clean/` and `joined/`: datasets.
- `reports/`:
  - `acceptance_report.md`;
  - `perth_rain_report.html`;
  - per-location calibration reports;
  - audits.
- `archive/`: quarantined, withdrawn and superseded files, each with a manifest or README. Nothing in it is ever
  overwritten.
- `README.md`: setup, rerun commands and method.
