"""Export the public website data (versioned JSON) from the pipeline's OUTPUT files.

Reads only CSV / JSON / parquet outputs: prediction records, verification tables, held-out metrics, station
summaries. It never opens model artifacts (.joblib), raw archives or credentials, and it refuses to write output
that contains local filesystem paths.

Publishing is a two-step, human-approved process:

    python website/tools/export_site_data.py            # writes website/data/pending/ and validates it
    python website/tools/export_site_data.py --approve  # re-validates pending/ and copies it to website/data/v1/

The site only reads website/data/v1/. Nothing here runs the forecasting pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "website"
PENDING = SITE / "data" / "pending"
PUBLIC = SITE / "data" / f"v{SCHEMA_VERSION}"
TZ = "Australia/Perth"

MODEL_LABEL = {"sr_ecmwf_ifs": "ECMWF IFS (single run)", "pr_ecmwf_ifs025": "ECMWF IFS 0.25°",
               "pr_jma_gsm": "JMA GSM", "pr_ncep_gfs_global": "NOAA GFS", "pr_bom_access_global": "BoM ACCESS-G"}
FEATURE_LABEL = {"ecmwf_ifs_sr": "ECMWF IFS only", "ecmwf_ifs025_pr": "ECMWF IFS 0.25° only",
                 "jma_gsm_pr": "JMA GSM only", "ecmwf_composite": "ECMWF composite",
                 "all_composite": "All-model composite"}
_RUN = re.compile(r"(\w+): (?:run|runs up to) (\d{4}-\d{2}-\d{2} \d{2})Z \((explicit|inferred)\)")
_PATH = re.compile(r"[A-Za-z]:\\|[A-Za-z]:/|/Users/|/home/|\\Users\\|AppData|\.joblib|\.venv", re.I)


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def num(x, nd=4):
    if x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x):
        return None
    return round(float(x), nd)


def iso(ts) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_local(ts) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).tz_convert(TZ).isoformat()


def parse_runs(detail) -> list[dict]:
    if not isinstance(detail, str):
        return []
    out = []
    for col, run, kind in _RUN.findall(detail):
        out.append({"input": col, "model": MODEL_LABEL.get(col, col),
                    "run_init_utc": pd.Timestamp(run + ":00", tz="UTC").strftime("%Y-%m-%dT%H:%MZ"),
                    "run_identification": "explicit run requested" if kind == "explicit"
                    else "latest contributing run, inferred from the offset rule"})
    return out


# ------------------------------------------------------------------------------------------------- forecast
def export_location(name, s, status, verif):
    st = json.loads((ROOT / "reports" / s / "run_summary.json").read_text())["station"]
    recs = sorted((ROOT / "data" / s / "predictions").glob("forecast_*.csv"))
    rec = recs[-1]
    issued = pd.Timestamp(pd.to_datetime(rec.stem.split("_")[1][:14], format="%Y%m%dT%H%MZ"), tz="UTC")
    pred = pd.read_csv(rec)
    obs = pd.read_parquet(ROOT / "data" / s / "clean" / "observations.parquet",
                          columns=["label_date_local", "observed_precip_mm", "retrieved_at_utc"])
    have = obs[obs.observed_precip_mm.notna()]
    manifest = json.loads((ROOT / "data" / s / "models" / "manifest.json").read_text())["meta"]
    # the fit selected for each lead is the only one with a held-out score in this release
    selection = json.loads((ROOT / "data" / s / "models" / "selection.json").read_text())
    scored_fit = {lead: selection[lead]["selected"]["feature_set"] for lead in ("day1", "day2", "day3") if lead in selection}

    days = []
    for day, grp in pred.groupby("label_date_local", sort=True):
        w = grp.iloc[0]
        ws, we = pd.Timestamp(w.window_start_utc), pd.Timestamp(w.window_end_utc)
        forecasts, withdrawn = [], []
        for _, r in grp.sort_values("lead_group").iterrows():
            if r.status != "ok":
                continue
            runs = parse_runs(r.source_detail)
            latest = max((pd.Timestamp(x["run_init_utc"]) for x in runs), default=None)
            vs = status[(status.location == name) & (status.gauge_day == day) & (status.lead == r.lead_group)]
            vstat = vs.row_status.iloc[0] if len(vs) else "UNVERIFIED"
            vd = verif[(verif.location == name) & (verif.gauge_day == day) & (verif.lead == r.lead_group)]
            # Previous Runs inputs were checked against the archived run when the forecast was made, but the
            # downloaded response that was checked is not retained, so the check cannot be repeated later.
            if vstat == "VERIFIED" and (vd.input.astype(str).str.startswith("pr_") & (vd.status == "VERIFIED")).any():
                vstat = "CHECKED_AT_ISSUE"
            if vstat == "FAILED":
                # never publish numbers from a forecast whose inputs failed verification (e.g. an input credited to a
                # model run that had not been published when the forecast was made)
                withdrawn.append({"lead_group": r.lead_group,
                                  "reason": "; ".join(f"{x.input}: {x.detail}" for x in vd.itertuples() if x.status == "FAILED")})
                continue
            raw = [{"input": c[len("input_"):-len("_mm")], "model": MODEL_LABEL.get(c[len("input_"):-len("_mm")]),
                    "total_mm": num(r[c], 2)} for c in pred.columns if c.startswith("input_") and pd.notna(r[c])]
            forecasts.append({
                "lead_group": r.lead_group, "status": "ok",
                "verification": vstat.lower(),
                "verification_detail": [{"input": x.input, "status": x.status.lower(), "detail": x.detail}
                                        for x in vd.itertuples()],
                "latest_model_run_utc": iso(latest),
                "lead_hours_to_window_start": num((ws - latest).total_seconds() / 3600, 1) if latest is not None else None,
                "lead_hours_to_window_end": num((we - latest).total_seconds() / 3600, 1) if latest is not None else None,
                "probabilities": {"ge_0_2mm": num(r.p_rain_ge_0_2mm), "ge_1mm": num(r.p_ge_1mm),
                                  "ge_5mm": num(r.p_ge_5mm), "ge_10mm": num(r.p_ge_10mm)},
                "amounts_mm": {"median": num(r.q50_mm, 2), "expected": num(r.mean_estimate_mm, 3),
                               "p10": num(r.q10_mm, 2), "p25": num(r.q25_mm, 2), "p75": num(r.q75_mm, 2),
                               "p90": num(r.q90_mm, 2)},
                "route": r.route, "is_fallback": str(r.route).startswith("fallback"),
                "heldout_scored": scored_fit.get(r.lead_group) == r.feature_set_used,
                "calibrated_model": r.feature_set_used, "calibrated_model_label": FEATURE_LABEL.get(r.feature_set_used),
                "models_available": [m for m in str(r.models_available).split(",") if m and m != "nan"],
                "inputs_blocked_by_timing": [m for m in str(r.inputs_blocked_by_timing).split(",") if m and m != "nan"],
                "input_runs": runs, "raw_model_totals_uncalibrated": raw,
                "model_trained_to": r.model_train_last, "model_training_days": None if pd.isna(r.model_n_train) else int(r.model_n_train),
                "model_run_id": r.run_id, "artifact_sha256": r.artifact_sha256,
                "inputs_retrieved_utc": r.retrieved_at_utc if isinstance(r.retrieved_at_utc, str) else None})
        blocked = sorted({m for x in grp.inputs_blocked_by_timing.dropna() for m in str(x).split(",") if m})
        days.append({"label_date": day, "window_start_local": iso_local(ws), "window_end_local": iso_local(we),
                     "window_start_utc": iso(ws), "window_end_utc": iso(we),
                     "availability": "available" if forecasts else "unavailable",
                     "unavailable_reason": None if forecasts else
                     ("A forecast was made but withdrawn: one of its inputs was credited to a model run that had not "
                      "been published when the forecast was made." if withdrawn else
                      "The model runs needed for a calibrated forecast of this window had not been published when "
                      "the forecast was made."),
                     "withdrawn": withdrawn,
                     "inputs_blocked_by_timing": blocked, "forecasts": forecasts})
    return {"id": s, "name": name,
            "gauge": {"id": st["station_id"], "name": st["station_name"].title(),
                      "distance_km": round(float(st["distance_km"]), 1)},
            "forecast_made_utc": iso(issued),
            "latest_observation": {"label_date": str(have.label_date_local.max()),
                                   "retrieved_utc": str(have.retrieved_at_utc.max())},
            "model_release": {"run_id": manifest["run_id"], "created_utc": manifest["created_utc"],
                              "enso_status": manifest["enso_status"]},
            "days": days}


# ---------------------------------------------------------------------------------------------- performance
def export_performance(locs):
    T = pd.read_csv(ROOT / "reports" / "acceptance_heldout_table.csv")
    gauges, seen = [], set()
    for name, s in locs:
        st = json.loads((ROOT / "reports" / s / "run_summary.json").read_text())["station"]
        if st["station_id"] in seen:
            next(g for g in gauges if g["gauge_id"] == st["station_id"])["serves"].append(name)
            continue
        seen.add(st["station_id"])
        m = pd.read_csv(ROOT / "reports" / s / "calibration_metrics.csv")
        p = pd.read_csv(ROOT / "reports" / s / "paired_differences.csv")
        leads = []
        for lead in ("day1", "day2", "day3"):
            h = m[(m.lead == lead) & (m.split == "holdout")]
            cal = h[h.method == "hurdle"].iloc[0]
            clim = h[h.method == "climatology"].iloc[0]
            rows = [{"method": "Calibrated median", "kind": "calibrated", "mae": num(cal.median_MAE_mm, 3),
                     "bias": num(cal.median_bias_mm, 3), "rmse": num(cal.median_RMSE_mm, 3)},
                    {"method": "Calibrated expected total", "kind": "calibrated", "mae": num(cal.mean_est_MAE_mm, 3),
                     "bias": num(cal.mean_est_bias_mm, 3), "rmse": num(cal.mean_est_RMSE_mm, 3)}]
            for _, r in h[h.method.astype(str).str.startswith("raw:")].iterrows():
                key = r.method[4:]
                rows.append({"method": "Equal-weight blend (raw)" if key == "equal_weight_blend"
                             else f"{MODEL_LABEL.get(key, key)} (raw)", "kind": "raw", "mae": num(r.MAE_mm, 3),
                             "bias": num(r.bias_fc_minus_obs_mm, 3), "rmse": num(r.RMSE_mm, 3)})
            rows.append({"method": "Season-aware climatology (median)", "kind": "climatology",
                         "mae": num(clim.median_MAE_mm, 3), "bias": num(clim.median_bias_mm, 3),
                         "rmse": num(clim.median_RMSE_mm, 3)})
            pl = p[(p.lead == lead) & p.mean_diff.notna()]
            leads.append({
                "lead_group": lead, "n_days": int(cal.n), "test_first": cal.test_first, "test_last": cal.test_last,
                "trained_to": cal.train_last, "selected_model": cal.feature_set,
                "selected_model_label": FEATURE_LABEL.get(cal.feature_set),
                "identical_rows_for_all_methods": bool(cal.keys_identical_across_methods),
                "pinball_skill_vs_seasonal_climatology": num(cal.pinball_skill_vs_clim),
                "brier_ge_0_2mm": {"calibrated": num(cal["brier_0.2"]), "seasonal_climatology": num(clim["brier_0.2"])},
                "observed_rain_day_frequency": num(cal["freq_0.2"]),
                "point_accuracy_mm": rows,
                "paired_differences": [{"comparison": r.comparison.replace("  ", " "), "mean": num(r.mean_diff),
                                        "ci95": [num(r.ci95_lo), num(r.ci95_hi)],
                                        "significant": bool(r.significant_at_95)} for r in pl.itertuples()]})
        gauges.append({"gauge_id": st["station_id"], "gauge_name": st["station_name"].title(), "serves": [name],
                       "leads": leads})
    return {"gauges": gauges,
            "method": {"holdout": "Last 20 % of dates, never used for fitting or model choice; scored once.",
                       "event_definition": "Rain day = 9am-9am total >= 0.2 mm (inclusive).",
                       "bias_sign": "forecast minus observed",
                       "uncertainty": "95 % circular block bootstrap of per-day differences, 7-day blocks, 2000 resamples.",
                       "probability_comparison_with_raw_models": "Not possible: raw models give amounts, not probabilities."},
            "not_evaluated": ["The ECMWF-only fallback fit that currently serves most day-2 and day-3 forecasts was "
                              "not scored on the held-out days in this release; only its raw input was. Its own "
                              "accuracy is therefore unknown and is planned for the next version."]}


# ------------------------------------------------------------------------------------------------------ map
# Gauges chosen for the next version but not yet used by any published forecast (shown as "planned").
PLANNED_GAUGES = {"Clarkson": "009264"}


def _km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def export_map(entries):
    locations, gauges, links = [], {}, []
    for e in entries:
        s = slug(e["name"])
        locations.append({"id": s, "name": e["name"], "lat": round(e["latitude"], 5), "lon": round(e["longitude"], 5),
                          "coordinate_source": e.get("coordinate_source", "")})
        st = json.loads((ROOT / "reports" / s / "run_summary.json").read_text())["station"]
        gid = st["station_id"]
        g = gauges.setdefault(gid, {"id": gid, "name": st["station_name"].title(), "lat": round(st["latitude"], 4),
                                    "lon": round(st["longitude"], 4), "status": "current", "serves": []})
        g["serves"].append(e["name"])
        links.append({"location": e["name"], "gauge": gid, "status": "current",
                      "distance_km": round(_km(e["latitude"], e["longitude"], st["latitude"], st["longitude"]), 1)})
        if e["name"] in PLANNED_GAUGES:
            pid = PLANNED_GAUGES[e["name"]]
            allst = pd.read_csv(ROOT / "reports" / s / "nearby_bom_stations_all_IDCJMC0014.csv", dtype={"station_id": str})
            r = allst[allst.station_id.str.zfill(6) == pid].iloc[0]
            gauges.setdefault(pid, {"id": pid, "name": r.station_name.title(), "lat": round(float(r.latitude), 4),
                                    "lon": round(float(r.longitude), 4), "status": "planned", "serves": []})
            gauges[pid]["serves"].append(e["name"])
            links.append({"location": e["name"], "gauge": pid, "status": "planned",
                          "distance_km": round(_km(e["latitude"], e["longitude"], float(r.latitude), float(r.longitude)), 1)})
    return {"locations": locations, "gauges": list(gauges.values()), "links": links,
            "notes": {"planned": "Planned gauge: chosen for the next version; its readings are not yet used by any "
                                 "forecast on this site."}}


# ------------------------------------------------------------------------------------------------ validate
def validate(folder: Path) -> list[str]:
    problems = []
    need = ["manifest.json", "forecast.json", "performance.json", "map.json"]
    for n in need:
        if not (folder / n).exists():
            problems.append(f"missing {n}")
    if problems:
        return problems
    for f in folder.glob("*.json"):
        text = f.read_text(encoding="utf-8")
        if _PATH.search(text):
            problems.append(f"{f.name}: contains a local path or artifact reference: {_PATH.search(text).group(0)!r}")
        if re.search(r"@[a-z0-9-]+\.[a-z]{2,}", text, re.I):
            problems.append(f"{f.name}: contains an e-mail address")
        if re.search(r"(api[_-]?key|token|secret|password)\s*[:=]", text, re.I):
            problems.append(f"{f.name}: contains a credential-like field")
        def _reject(c):
            raise ValueError(f"non-standard JSON constant {c}")
        try:
            json.loads(text, parse_constant=_reject)     # browsers reject NaN / Infinity
        except ValueError as exc:
            problems.append(f"{f.name}: invalid JSON for browsers ({exc})")
    fc = json.loads((folder / "forecast.json").read_text(encoding="utf-8"))
    if fc.get("schema_version") != SCHEMA_VERSION:
        problems.append("forecast.json: wrong schema_version")
    for loc in fc["locations"]:
        for d in loc["days"]:
            if d["availability"] == "unavailable" and d["forecasts"]:
                problems.append(f"{loc['id']} {d['label_date']}: unavailable day carries numbers")
            for f in d["forecasts"]:
                pr = f["probabilities"]
                if any(v is None or not 0 <= v <= 1 for v in pr.values()):
                    problems.append(f"{loc['id']} {d['label_date']}: probability out of range")
                if f.get("verification") not in {"verified", "checked_at_issue", "unverified", "failed"}:
                    problems.append(f"{loc['id']} {d['label_date']}: unknown verification state {f.get('verification')!r}")
                if not isinstance(f.get("heldout_scored"), bool):
                    problems.append(f"{loc['id']} {d['label_date']}: heldout_scored missing")
                if not pr["ge_0_2mm"] >= pr["ge_1mm"] >= pr["ge_5mm"] >= pr["ge_10mm"]:
                    problems.append(f"{loc['id']} {d['label_date']}: probabilities not decreasing")
    ol = folder / "outlook.json"
    if ol.exists():
        o = json.loads(ol.read_text(encoding="utf-8"))
        for day in o.get("days", []):
            if not day.get("calibrated") and "%" in day.get("rain", ""):
                problems.append(f"outlook.json {day.get('date')}: raw model guidance shown as a percentage chance")
            if not isinstance(day.get("calibrated"), bool):
                problems.append(f"outlook.json {day.get('date')}: missing calibrated flag")
        for k in ("generated_utc", "model_guidance_retrieved_utc", "headline", "method", "sources"):
            if not o.get(k):
                problems.append(f"outlook.json: missing {k}")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--approve", action="store_true", help="validate pending/ and publish it to data/v1/")
    a = ap.parse_args(argv)
    if a.approve:
        problems = validate(PENDING)
        if problems:
            sys.exit("NOT approved:\n  " + "\n  ".join(problems))
        PUBLIC.mkdir(parents=True, exist_ok=True)
        for f in PENDING.glob("*.json"):
            shutil.copy2(f, PUBLIC / f.name)
        print(f"approved: {sorted(p.name for p in PUBLIC.glob('*.json'))} -> website/data/v{SCHEMA_VERSION}/")
        return
    locs = [(e["name"], slug(e["name"])) for e in tomllib.loads((ROOT / "config.toml").read_text())["locations"]]
    status = pd.read_csv(ROOT / "reports" / "served_row_status.csv")
    verif = pd.read_csv(ROOT / "reports" / "served_row_verification.csv")
    exported = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    forecast = {"schema_version": SCHEMA_VERSION, "exported_utc": exported, "timezone": TZ,
                "locations": [export_location(n, s, status, verif) for n, s in locs]}
    performance = {"schema_version": SCHEMA_VERSION, "exported_utc": exported, **export_performance(locs)}
    entries = tomllib.loads((ROOT / "config.toml").read_text())["locations"]
    mapdata = {"schema_version": SCHEMA_VERSION, "exported_utc": exported, **export_map(entries)}
    manifest = {"schema_version": SCHEMA_VERSION, "exported_utc": exported,
                "release": "Frozen audited release (no El Niño input), models trained 2026-09-30",
                "forecast_made_utc": max(l["forecast_made_utc"] for l in forecast["locations"]),
                "files": ["forecast.json", "performance.json", "map.json"],
                "attribution": "Forecast data: Open-Meteo.com (CC BY 4.0), from ECMWF, JMA and NOAA NCEP model output. "
                               "Observations: Commonwealth of Australia, Bureau of Meteorology."}
    if PENDING.exists():
        shutil.rmtree(PENDING)
    PENDING.mkdir(parents=True)
    challenge_src = ROOT / "reports" / "challenge" / "challenge_site.json"   # written by `python -m challenge export`
    extra = []
    if challenge_src.exists():
        extra = [("challenge.json", json.loads(challenge_src.read_text(encoding="utf-8")))]
        manifest["files"].append("challenge.json")
    outlook_src = ROOT / "reports" / "outlook" / "outlook_site.json"        # written by scripts/seven_day_outlook.py
    if outlook_src.exists():
        extra.append(("outlook.json", json.loads(outlook_src.read_text(encoding="utf-8"))))
        manifest["files"].append("outlook.json")
    for n, obj in (("manifest.json", manifest), ("forecast.json", forecast), ("performance.json", performance),
                   ("map.json", mapdata), *extra):
        (PENDING / n).write_text(json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    problems = validate(PENDING)
    if problems:
        sys.exit("export written to website/data/pending/ but FAILED validation:\n  " + "\n  ".join(problems))
    print("exported to website/data/pending/ and validated. Review it, then run with --approve to publish to "
          f"website/data/v{SCHEMA_VERSION}/.")


if __name__ == "__main__":
    main()
