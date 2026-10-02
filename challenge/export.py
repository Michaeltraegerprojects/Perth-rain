"""Dashboard data for the Challenge tab: reports/challenge/challenge_site.json.

The website exporter (website/tools/export_site_data.py) copies it into the reviewed export as challenge.json.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from challenge import champion, ledger, scoring

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports" / "challenge"

# What each source is and why it is in or out. Facts come from the probes in reports/challenge/ (see README).
SOURCE_NOTES = {
    "ecmwf_ifs": ("existing benchmark", "included", "ECMWF IFS 9 km single runs (12 UTC) at the trained lead."),
    "ecmwf_ifs025": ("existing benchmark", "included", "ECMWF IFS 0.25° via Previous Runs."),
    "jma_gsm": ("existing benchmark", "included", "JMA GSM via Previous Runs."),
    "ncep_gfs_global": ("existing benchmark", "included", "NOAA GFS via Previous Runs."),
    "icon_global": ("challenger", "included", "DWD ICON Global. Previous Runs from 2024-01-20, single runs from "
                    "2026-04-02; previous_dayN values matched the rule-named runs on 100% of compared rainy hours."),
    "ecmwf_aifs025_single": ("challenger", "included", "ECMWF AIFS (AI model), 6-hourly output. Previous Runs from "
                             "2025-02-18, single runs from 2026-04-02; 100% of compared rainy hours matched."),
    "bom_access_global": ("candidate", "excluded: suspended", "BoM ACCESS-G on Open-Meteo: last run 2025-06-26, "
                          "Previous Runs end 2025-07-06, no single runs. No current forecasts to compare."),
    "gem_global": ("candidate", "excluded: provenance unverifiable", "Canadian GEM Global: Previous Runs values exist "
                   "to today, but the model metadata reports a last run of 2026-05-26 and every single-run request "
                   "failed, so the run behind each value cannot be verified."),
    "ensemble:ecmwf_ifs025": ("ensemble candidate", "excluded for now: no archive", "ECMWF IFS ensemble: live only "
                              "(no archived runs on the Ensemble or Single Runs APIs), and the live run is newer than "
                              "the champion's fixed-lead inputs, so it would need a separately labelled experiment."),
    "ensemble:gfs025": ("ensemble candidate", "excluded for now: no archive", "NOAA GEFS: same as the ECMWF ensemble."),
}


import re

_RUN = re.compile(r"(\w+): (?:run|runs up to) (\d{4}-\d{2}-\d{2} \d{2})Z \((explicit|inferred)\)")
ESTIMATED = "estimated (initialisation + 6 h)"


def competitor_timing(r: dict) -> dict:
    """Model-run start, information age at issue, and whether availability was OBSERVED (the run answered an
    explicit request, or the model's metadata reported it available) or only ESTIMATED from the fixed 6-hour rule."""
    issued = pd.Timestamp(r["issued_at_utc"])
    comp = r.get("competitor", "")
    ev = r.get("availability_evidence") or ""
    if comp == "champion":
        runs = [(c, pd.Timestamp(t + ":00", tz="UTC"), kind) for c, t, kind in _RUN.findall(r.get("inputs") or "")]
        if not runs:
            return {"run_init_utc": None, "info_age_h": None, "availability_basis": None, "evidence": ""}
        latest = max(t for _, t, _ in runs)
        explicit = all(kind == "explicit" for _, _, kind in runs)
        basis = r.get("availability_basis") or ("observed" if explicit else ESTIMATED)
        ev = r.get("availability_evidence") or ("explicit runs requested" if explicit else
                                                "Previous Runs inputs: run inferred from the offset rule")
    else:
        t = r.get("latest_run_utc") or r.get("named_run_utc")
        if not t or str(t) in ("None", "NaT", "nan"):
            return {"run_init_utc": None, "info_age_h": None, "availability_basis": None, "evidence": ev}
        latest = pd.Timestamp(t)
        latest = latest.tz_localize("UTC") if latest.tzinfo is None else latest.tz_convert("UTC")
        if r.get("availability_basis"):
            basis = r["availability_basis"]
        elif "HTTP 200" in ev or "model metadata" in ev or comp.startswith("sr_"):
            basis = "observed"
        else:
            basis = ESTIMATED
    return {"run_init_utc": f"{latest:%Y-%m-%dT%H:%MZ}",
            "info_age_h": round((issued - latest).total_seconds() / 3600, 1),
            "availability_basis": basis, "evidence": ev}


def _load(name):
    p = REP / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def sources_table() -> list[dict]:
    av = pd.read_csv(REP / "source_availability.csv")
    sem = pd.read_csv(REP / "semantics_verification.csv") if (REP / "semantics_verification.csv").exists() else pd.DataFrame()
    rows = []
    for _, r in av.iterrows():
        role, status, note = SOURCE_NOTES.get(r.model_id, ("", "unknown", ""))
        s = sem[sem.model == r.model_id] if len(sem) else sem
        rows.append({
            "model_id": r.model_id, "role": role, "status": status, "note": note,
            "grid": f"{r.grid_lat:.3f}, {r.grid_lon:.3f}" if pd.notna(r.grid_lat) else None,
            "native_step_h": None if pd.isna(r.native_step_h) else float(r.native_step_h),
            "last_run_init_utc": None if pd.isna(r.last_run_init_utc) else str(r.last_run_init_utc),
            "availability_delay_h": None if pd.isna(r.availability_delay_h) else float(r.availability_delay_h),
            "previous_runs_from": None if pd.isna(r.previous_runs_first_utc) else str(r.previous_runs_first_utc)[:10],
            "previous_runs_to": None if pd.isna(r.previous_runs_last_utc) else str(r.previous_runs_last_utc)[:10],
            "single_runs_from": None if pd.isna(r.single_runs_first_12z) else str(r.single_runs_first_12z)[:10],
            "run_mapping_check": [{"lead_day": int(x.lead_day), "compared": int(x.compared), "matched": int(x.matched)}
                                  for x in s.itertuples()] if len(s) else None,
            "licence": "Open-Meteo API, CC BY 4.0, free for non-commercial use; no key"})
    return rows


def live_view() -> dict:
    recs = ledger.read_ledger()
    if not recs:
        return {"issue_id": None, "rows": []}
    df = pd.DataFrame(recs)
    last = df.issue_id.max()
    d = df[df.issue_id == last]
    scored = ROOT / "data" / "challenge" / "prospective_scored.csv"
    obs = {}
    if scored.exists():
        s = pd.read_csv(scored, dtype={"gauge_id": str})
        obs = {(r.gauge_id, r.label_date): r.observed_precip_mm for r in s.itertuples() if pd.notna(r.observed_precip_mm)}
    rows = []
    for (loc, label, lead), g in d.groupby(["location", "label_date", "lead"], sort=True):
        comp = []
        for r in g.itertuples():
            comp.append({"competitor": scoring.LABELS.get(r.competitor, r.competitor) if r.competitor != "champion"
                         else "Our forecast", "key": r.competitor, "kind": r.kind, "status": r.status,
                         "missing_reason": r.missing_reason,
                         "amount_mm": r.total_mm if r.competitor != "champion" else r.expected_mm,
                         "median_mm": getattr(r, "median_mm", None) if r.competitor == "champion" else None,
                         "probabilities": r.probabilities if r.competitor == "champion" else None,
                         "latest_run_utc": getattr(r, "latest_run_utc", None),
                         "lead_hours": getattr(r, "lead_hours_to_window_start", None),
                         "timing": competitor_timing(r._asdict()) if r.status == "ok" else None})
        first = g.iloc[0]
        rows.append({"location": loc, "gauge_id": first.gauge_id, "gauge_name": first.gauge_name,
                     "gauge_distance_km": first.gauge_distance_km, "label_date": label, "lead": lead,
                     "window_start_utc": first.window_start_utc, "window_end_utc": first.window_end_utc,
                     "observed_mm": obs.get((first.gauge_id, label)), "competitors": comp})
    return {"issue_id": last, "issued_at_utc": d.issued_at_utc.iloc[0], "schedule_slot_utc": d.schedule_slot_utc.iloc[0],
            "ledger_records": len(recs), "rows": rows}


def _clean(x):
    """JSON has no NaN: missing values become null."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, float) and x != x:
        return None
    return x


def export() -> Path:
    freeze = champion.verify()
    out = {"schema_version": 1, "exported_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "champion": {s: {"name": L["name"], "run_id": L["run_id"],
                            "selected": {lead: d["selected"]["feature_set"] for lead, d in L["leads"].items()}}
                        for s, L in freeze["locations"].items()},
           "sources": sources_table(), "historical": _load("historical"), "prospective": _load("prospective"),
           "ledger_head": ledger.head() if ledger.LEDGER.exists() else None,
           "live": live_view(),
           "rules": {"fair_contest": "Same gauge, same 9am-9am window, same lead group and the same as-of rule (a "
                                     "forecast counts only if its model run was published before the cutoff). "
                                     "Leaderboards use only windows where every listed competitor has a forecast; "
                                     "missing forecasts are counted as coverage gaps, never as zero rain or wins.",
                     "probabilities": "Only our forecast issues probabilities. Raw model amounts are deterministic, "
                                      "so their probability scores are unavailable. CRPS of a point forecast equals "
                                      "its absolute error.",
                     "satellite": "Satellite cloud images are never used to verify rainfall; only gauge readings are.",
                     "timing": "Competitors share the lead group and the as-of cutoff, but not the age of their "
                               "information: explicit 12 UTC single runs, and Previous Runs inputs stitched from several "
                               "runs, can start at different times. The tables show the latest model run used and its "
                               "age. Observed availability means the run answered an explicit request or the model's own "
                               "metadata listed it at issue time; estimated means only the fixed 6-hour rule was used."}}
    REP.mkdir(parents=True, exist_ok=True)
    p = REP / "challenge_site.json"
    p.write_text(json.dumps(_clean(out), indent=1, default=str, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return p
