# Forecast Challenge

Compares the frozen champion (the audited calibrated model) with raw weather models and new challenger models, on the
same rain-gauge readings. The champion is never retrained, re-selected or replaced by this code. `challenge/` is
deliberately outside the `perthrain` package, so the audited package and its code hash are unchanged.

## Sources

The probes ran on 2026-10-01 and wrote `reports/challenge/source_availability.csv`; the raw responses are in
`data/raw/open_meteo/challenge_probe/`. All sources are Open-Meteo public APIs (CC BY 4.0, free for non-commercial
use, no key).

| Model | Role | Status | Previous Runs archive | Single runs from |
|---|---|---|---|---|
| ECMWF IFS 9 km, ECMWF IFS 0.25°, JMA GSM, NOAA GFS | existing benchmarks | included | as in the main pipeline | 2024-03-14 (IFS) |
| DWD ICON Global (`icon_global`) | challenger | included | 2024-01-20 onward | 2026-04-02 |
| ECMWF AIFS (`ecmwf_aifs025_single`) | challenger | included | 2025-02-18 onward | 2026-04-02 |
| BoM ACCESS-G (`bom_access_global`) | candidate | **excluded: suspended** | 2024-01-20 to 2025-07-06; last run 2025-06-26 | none |
| Canadian GEM Global (`gem_global`) | candidate | **excluded: provenance unverifiable** | values to today | every request failed; metadata says last run 2026-05-26 |
| ECMWF IFS ensemble, NOAA GEFS | ensemble candidates | **excluded for now** | no archive | no archive; the live run is newer than the champion's fixed-lead inputs |

**Run mapping.** For ICON and AIFS, `precipitation_previous_dayN` values equalled the run named by the rule
`floor_6h(ceil_to_native_step(t)) − 24·N h` on 100% of the rainy hours compared
(`reports/challenge/semantics_verification.csv`).

## Fair-contest rules

- **Location:** forecasts are taken at the gauge coordinates; the returned grid point is recorded.
- **Same gauge reading:** every competitor is scored against the same reading, for the same 9am–9am window, at the
  same lead group, under the same as-of rule. A forecast counts only if its model run was available before the
  cutoff: the window start (historical) or the issue time (prospective).
- **Availability evidence:** each forecast records how its availability was established.
  - **Observed:** the explicit run was fetched (HTTP 200) at issue time, or the model's metadata, read before the
    forecast data, showed the run as published by the issue time. That means either a later run exists, or it is the
    latest run and its availability time is before the issue time.
  - **Estimated:** neither is available, so availability is assumed at initialisation + 6 h. This is a fixed
    estimate; JMA was measured at about 9.6 h.
  - The Challenge tab shows each competitor's run initialisation, information age (hours from initialisation to
    issue) and basis. The same lead group does not mean the same information age.
- **Leads:**
  - Day 1 uses the 12 UTC single run, which is the champion's rule.
  - Days 2–3 use `previous_dayN`, the champion's pr inputs. They also use the 12 UTC single-run rule, labelled
    "12 UTC run", which matches the champion's ECMWF single-run input.
- **Common windows:** leaderboards use only windows where every listed competitor has a forecast. Coverage is
  reported separately.
- **Missing data:** missing forecasts stay missing. They are never zero and never wins.
- **Probabilities:** only the champion issues probabilities. Raw models are deterministic, so their Brier scores are
  "unavailable". For a point forecast the CRPS equals its absolute error.
- **Uncertainty:** paired 7-day block bootstrap. P-values are Holm-adjusted across every comparison in a report.
  When the bootstrap has no variability, p comes from an exact sign test. That happens with identical errors,
  the same difference every day, or fewer than 7 days. Identical errors are a tie (p = 1), never a win.
- **Verdicts:** the thresholds come from `[verdicts]` in `challenge/settings.toml` (currently ≥ 60 common days,
  ≥ 15 wet days and an adjusted p < 0.05). Otherwise the result is "insufficient evidence" or "inconclusive".

## Champion freeze

`challenge/champion_freeze.json` (schema 2) records, by SHA-256:

- every model artifact, `manifest.json` and `selection.json`;
- `config.toml` and every package module on the prediction and calibration import paths;
- the runtime versions (Python, numpy, pandas, scikit-learn, scipy, joblib);
- the per-row held-out reference files.

`verify` compares bytes and versions only and never unpickles an artifact. `refreeze --reason` writes a new record
after a reviewed code, config or runtime change. It works in this order:

- refuses if a recorded per-row reference is missing or changed (references are never recreated);
- re-checks every gauge × lead reproduction;
- refuses if any artifact, manifest or selection changed;
- keeps the old record in `challenge/freeze_history/`.

`freeze` records the references for a new champion; nothing else creates them.

## Tracks

- **Historical:** the champion's original, untouched held-out days. The champion's held-out forecasts are reproduced
  from its frozen configuration. The run stops unless they match on all of the following:
  - the evaluation keys (station, window, reading);
  - every stored aggregate score;
  - every row of the saved reference (`challenge/reference/`, local, hash in the freeze).

  These are exploratory comparisons for the challengers. Nothing is tuned on them.
- **Prospective:** an append-only, hash-chained ledger (`data/challenge/ledger.jsonl`) of every forecast as issued.
  - The schedule is set in `challenge/settings.toml`: 21:00 UTC daily (05:00 AWST).
  - Off-schedule `--force` runs are labelled "unscheduled".
  - Observations are joined later in a separate scored table; the ledger is never rewritten.
  - A checkpoint file stores the record count and head hash, so deleting or truncating the final records is
    detected. `collect` holds an exclusive lock file; a second writer is refused.

## Commands

From the project root, using the project's `.venv`:

```bash
.venv/Scripts/python -m challenge verify
```
```bash
.venv/Scripts/python -m challenge collect
```
```bash
.venv/Scripts/python -m challenge score
```
```bash
.venv/Scripts/python -m challenge export
```
```bash
.venv/Scripts/python website/tools/export_site_data.py
```
```bash
.venv/Scripts/python -m pytest -q challenge/tests
```

Add `-m "not local_data"` to skip the tests that need the locally built champion, data and reference files.

- `collect` must run within 3 hours after 21:00 UTC; nothing schedules it automatically.
- `score` refreshes recent gauge readings from BoM's public FTP files and rescores.
- `export` writes the Challenge-tab data; `website/tools/export_site_data.py` then stages it for review.
- The other subcommands are `freeze`, `refreeze --reason`, `checkpoint`, `probe`, `verify-semantics` and
  `historical`.
