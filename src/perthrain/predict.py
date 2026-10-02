"""Predict rainfall totals for upcoming gauge windows from LIVE archived-run forecasts.

Features are built exactly as in training: the same Single Runs / Previous Runs sources, the same
lead-group definitions, the same 09:00-09:00 local windows and the same completeness / publication rules.
A window/lead is only predicted when every hour of every model a fit needs is present.

Default mode is the **no-ENSO baseline**: only artifacts with ``enso_status == "no_enso"`` and a clean
feature schema can be loaded (see enso_guard.py), and no climate-index value is ever read. The unverified
ENSO feature is available only in the opt-in experimental mode, which uses separate directories
(models_enso_experimental/, predictions_enso_experimental/).
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import build, climate, enso_guard, forecasts, observations, provenance
from .calibrate import (CLIMATE_COLS, LEVELS, THRESHOLDS, exceed_prob, fit_X, mean_estimate, mixture_quantile)
from .pipeline import make_fetcher

log = logging.getLogger("perthrain.predict")

LIVE_PR_MODELS = ["jma_gsm", "ncep_gfs_global", "ecmwf_ifs025"]
LIVE_SR_MODELS = ["ecmwf_ifs"]
# Open-Meteo metadata domains behind each live Previous Runs model (GFS: both domains; the earlier run counts)
MODEL_META_IDS = {"jma_gsm": ["jma_gsm"], "ecmwf_ifs025": ["ecmwf_ifs025"],
                  "ncep_gfs_global": ["ncep_gfs013", "ncep_gfs025"]}
ESTIMATED = "estimated (initialisation + 6 h)"
META_URL = "https://api.open-meteo.com/data/{}/static/meta.json"


def fetch_model_meta(fetcher, refresh: bool = True) -> dict:
    """Latest run each live Previous Runs model has actually published, from Open-Meteo's model metadata. A model
    whose metadata cannot be read is left out (its inputs then fall back to the labelled 6-hour estimate)."""
    out = {}
    for model, ids in MODEL_META_IDS.items():
        lasts, checked = [], None
        for mid in ids:
            res = fetcher.get_json(META_URL.format(mid), {}, "open_meteo/live/meta", refresh=refresh,
                                   label=f"LIVE meta {mid}")
            t = (res.body or {}).get("last_run_initialisation_time") if res.ok else None
            if t:
                lasts.append(pd.Timestamp(t, unit="s", tz="UTC"))
                checked = res.retrieved_at_utc
        if lasts and len(lasts) == len(ids):
            out[model] = {"last_run_init_utc": min(lasts), "checked_utc": checked}
    return out
LEADS = {"day1": 1, "day2": 2, "day3": 3}
ALL_MODEL_COLS = ["sr_ecmwf_ifs", "pr_ecmwf_ifs025", "pr_jma_gsm", "pr_ncep_gfs_global"]

ENSO_UNVERIFIED_NOTE = (
    "**ENSO / IOD: not used (enso_status = no_enso).** These predictions are the no-ENSO baseline. The "
    "climate-index feature is marked UNVERIFIED: the BoM index file's definition could not be confirmed from BoM's "
    "own documentation (blocked to automated access) and its historical values may have been revised since first "
    "publication. See reports/enso_source_verification.md.")


def future_windows(now_utc: pd.Timestamp, tz: str, horizon_days: int) -> pd.DataFrame:
    """09:00-09:00 local windows that have NOT started yet at ``now_utc``."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    today = now_utc.tz_convert(tz).date()
    rows = []
    for d in pd.date_range(today, periods=horizon_days + 2, freq="D").date:
        s, e = observations.obs_window(d, tz)
        if s > now_utc:
            rows.append({"label_date_local": d, "window_start_utc": s, "window_end_utc": e})
    cols = ["label_date_local", "window_start_utc", "window_end_utc"]
    return pd.DataFrame(rows, columns=cols).sort_values("window_start_utc").reset_index(drop=True)


def _fetch_live(cfg, fetcher, lat: float, lon: float, windows: pd.DataFrame, now: pd.Timestamp,
                refresh: bool = True):
    """Download the live forecasts. ``refresh=False`` is an OFFLINE replay: cached responses only, a cache miss
    is reported as a problem and NEVER triggers a download."""
    fetcher.offline = not refresh
    sr_frames, pr_frames, problems = [], [], []
    needed: set[pd.Timestamp] = set()
    for n in LEADS.values():
        needed |= set(build.single_run_issue_time(windows.window_start_utc, n, cfg.single_run_hour_utc,
                                                  cfg.publication_latency_hours))
    for model in LIVE_SR_MODELS:
        for run in sorted(r for r in needed if r + pd.Timedelta(hours=cfg.publication_latency_hours) <= now):
            p = forecasts.single_run_params(lat, lon, model, run, forecast_days=6)
            res = fetcher.get_json(forecasts.SINGLE_RUNS_URL, p, "open_meteo/live/single_runs", refresh=refresh,
                                   label=f"LIVE single_runs {model} {run:%Y-%m-%dT%H}")
            if not refresh and not res.ok and res.error and res.error.startswith("offline"):
                problems.append(f"frozen replay: single_runs {model} {run} is not in the cache")
            if res.ok:
                sr_frames.append(forecasts.parse_single_run(res.body, model, run, res.url, res.params,
                                                            res.retrieved_at_utc))
            else:
                problems.append(f"single_runs {model} {run}: {res.error}")
    start = (windows.window_start_utc.min() - pd.Timedelta(days=1)).date()
    end = (windows.window_end_utc.max() + pd.Timedelta(days=1)).date()
    for model in LIVE_PR_MODELS:
        p = forecasts.previous_runs_params(lat, lon, model, start, end, list(LEADS.values()))
        res = fetcher.get_json(forecasts.PREVIOUS_RUNS_URL, p, "open_meteo/live/previous_runs", refresh=refresh,
                               label=f"LIVE previous_runs {model}")
        if not refresh and not res.ok and res.error and res.error.startswith("offline"):
            problems.append(f"frozen replay: previous_runs {model} is not in the cache")
        if res.ok:
            pr_frames.append(forecasts.parse_previous_runs(res.body, model, list(LEADS.values()), res.url,
                                                           res.params, res.retrieved_at_utc))
        else:
            problems.append(f"previous_runs {model}: {res.error}")
    sr = pd.concat(sr_frames, ignore_index=True) if sr_frames else pd.DataFrame()
    pr = pd.concat(pr_frames, ignore_index=True) if pr_frames else pd.DataFrame()
    return sr, pr, problems


def live_feature_table(cfg, sr: pd.DataFrame, pr: pd.DataFrame, windows: pd.DataFrame,
                       now: pd.Timestamp, model_meta: dict | None = None) -> pd.DataFrame:
    """Aggregate live hourly forecasts to the windows with the training-time rules.

    An input is usable only if every model run it depends on was issued and (by the documented latency) published
    BEFORE ``now``. For previous-runs inputs those runs follow from the offset rule (valid time - 24*N h); the API
    also returns values for hours whose offset run does not exist yet, so a complete response is not sufficient."""
    lead_days = list(LEADS.values())
    parts = []
    if not pr.empty:
        parts.append(build.aggregate_previous_runs(pr, windows, lead_days, cfg.publication_latency_hours))
    if not sr.empty:
        parts.append(build.aggregate_single_runs(sr, windows, lead_days, cfg.single_run_hour_utc,
                                                 cfg.publication_latency_hours, LIVE_SR_MODELS))
    if not parts:
        return pd.DataFrame()
    fw = build.finalize_forecast_windows(pd.concat(parts, ignore_index=True))
    fw["col"] = np.where(fw.source_api == "single_runs", "sr_", "pr_") + fw.model
    align_ok = (fw.alignment_status.eq("exact_hourly_boundaries") if getattr(cfg, "alignment_policy", "") == "strict"
                else fw.alignment_status.str.startswith(("exact_hourly_boundaries", "uniform_disaggregation")))
    ok = (fw.forecast_complete & fw.available_before_window_start_est.astype(bool) & align_ok
          & (fw.negative_hours.fillna(0) == 0))          # same exclusions as the training join
    latest_issue = fw.forecast_issue_time_utc.where(fw.forecast_issue_time_utc.notna(), fw.issue_time_inferred_max_utc)
    fw["latest_issue_utc"] = latest_issue
    fw["timing_ok"] = (latest_issue + pd.Timedelta(hours=cfg.publication_latency_hours)) <= now
    fw["timing_reason"] = np.where(fw.timing_ok, "",
                                   "required run not issued/published before the prediction time")
    # Availability evidence. Single runs were fetched by explicit run= and answered: observed. Previous Runs inputs
    # are checked against the latest run the model's own metadata lists as published; without metadata they rely on
    # the fixed 6-hour estimate, and say so.
    fw["availability_basis"] = np.where(fw.source_api == "single_runs", "observed", ESTIMATED)
    for model, m in (model_meta or {}).items():
        sel = (fw.source_api == "previous_runs") & (fw.model == model)
        if not sel.any():
            continue
        fw.loc[sel, "availability_basis"] = "observed"
        late = sel & (fw.latest_issue_utc > m["last_run_init_utc"])
        fw.loc[late, "timing_ok"] = False
        fw.loc[late, "timing_reason"] = (f"named run not yet published (model's latest published run "
                                         f"{m['last_run_init_utc']:%Y-%m-%d %HZ}, checked {m.get('checked_utc')})")
    fw["usable"] = ok & fw.timing_ok
    return fw


def _issue_text(rows: pd.DataFrame) -> str:
    out = []
    for _, r in rows.iterrows():
        if pd.notna(r.forecast_issue_time_utc):
            out.append(f"{r.col}: run {r.forecast_issue_time_utc:%Y-%m-%d %H}Z (explicit)")
        else:
            out.append(f"{r.col}: runs up to {r.issue_time_inferred_max_utc:%Y-%m-%d %H}Z (inferred)")
    return "; ".join(out)


def live_climate(indices: dict, windows: pd.DataFrame) -> dict:
    """EXPERIMENTAL ONLY. label date -> row with clim_nino34, clim_iod (pre-audit label-date rule)."""
    frame = pd.DataFrame({"label_date_local": pd.to_datetime(windows.label_date_local)})
    att = climate.attach_climate(frame, indices)
    return {pd.Timestamp(r.label_date_local).date(): r for r in att.itertuples()}


def climate_cols(fit: dict) -> list[str]:
    return CLIMATE_COLS.get(fit.get("climate"), [])


def routing_order(bundle: dict) -> list[str]:
    if bundle.get("routing_order"):
        return list(bundle["routing_order"])
    return list(dict.fromkeys(r["feature_set"] for r in bundle.get("ranking", [])))


def route(bundle: dict, available: set[str]) -> tuple[str | None, dict | None, int | None]:
    """First fit in the bundle's routing order whose input models are all available.

    Returns (feature_set, fit, rank); rank 0 is the cross-validation-selected primary model, rank >= 1 a
    fallback. (None, None, None) when no fit can be served with the available models."""
    for rank, name in enumerate(routing_order(bundle)):
        fit = bundle.get("fits", {}).get(name)
        if fit is not None and set(fit["cols"]) <= set(available):
            return name, fit, rank
    return None, None, None


def availability_subsets(cols: list[str] = ALL_MODEL_COLS) -> list[frozenset]:
    from itertools import combinations
    return [frozenset(c) for k in range(len(cols) + 1) for c in combinations(cols, k)]


def routing_table(bundles: dict) -> pd.DataFrame:
    """Which fit serves each lead for every combination of available forecast models (2^4 = 16 per lead)."""
    rows = []
    for lead, b in sorted(bundles.items()):
        for avail in availability_subsets():
            name, fit, rank = route(b, set(avail))
            rows.append({"lead": lead, "models_available": ",".join(sorted(avail)) or "(none)",
                         "feature_set": name or "NOT PREDICTED", "route": None if name is None else
                         ("primary" if rank == 0 else f"fallback#{rank}"),
                         "feature_schema": " | ".join(fit.get("feature_schema") or []) if fit else "",
                         "climate": (fit.get("climate") or "none") if fit else "",
                         "run_id": (b.get("meta") or {}).get("run_id")})
    return pd.DataFrame(rows)


def predict_windows(bundles: dict, fw: pd.DataFrame, windows: pd.DataFrame, tz: str,
                    clim: dict | None = None, experimental: bool = False) -> pd.DataFrame:
    rows = []
    for _, w in windows.iterrows():
        for lead, n in LEADS.items():
            b = bundles.get(lead)
            if b is None:
                # no artifact for this lead: say so explicitly; no raw (uncalibrated) forecast is served instead
                rows.append({"label_date_local": w.label_date_local, "window_start_utc": w.window_start_utc,
                             "window_end_utc": w.window_end_utc, "lead_group": lead,
                             "status": "no_model_artifact", "feature_set_used": None, "route": None})
                continue
            rec = {"label_date_local": w.label_date_local,
                   "window_start_local": w.window_start_utc.tz_convert(tz),
                   "window_end_local": w.window_end_utc.tz_convert(tz),
                   "window_start_utc": w.window_start_utc, "window_end_utc": w.window_end_utc,
                   "lead_group": lead, "status": "no_complete_forecast", "feature_set_used": None,
                   "route": None, "run_id": (b.get("meta") or {}).get("run_id"),
                   "enso_status": b.get("enso_status"), "artifact_sha256": b.get("_artifact_sha256")}
            if fw.empty:
                rec.update(route="exhausted", models_available="", inputs_blocked_by_timing="")
                rows.append(rec)
                continue
            sub = fw[(fw.label_date_local == w.label_date_local) & (fw.lead_day == n)]
            usable = sub[sub.usable]
            vals = dict(zip(usable.col, usable.forecast_precip_mm))
            rec["models_available"] = ",".join(sorted(vals))
            if "timing_ok" in sub.columns:        # complete inputs withheld because their runs did not exist yet
                blocked = sub[sub.forecast_complete.astype(bool) & ~sub.timing_ok.astype(bool)].col
                rec["inputs_blocked_by_timing"] = ",".join(sorted(blocked))
            name, fit, rank = route(b, set(vals))
            if name is None:
                # fallback exhausted: nothing calibrated can serve this lead. No raw-forecast value is substituted.
                rec["route"] = "exhausted"
                rows.append(rec)
                continue
            if not experimental and (fit.get("climate") not in (None, "none") or climate_cols(fit)):
                raise enso_guard.EnsoGuardError(f"{lead}: routed fit {name} uses climate {fit.get('climate')!r} "
                                                "in no-ENSO mode")
            frame = pd.DataFrame({"label_date_local": [pd.Timestamp(w.label_date_local)],
                                  **{f"fc_{c}_mm": [vals[c]] for c in fit["cols"]}})
            clipped = []
            if experimental:
                missing = False
                for c in climate_cols(fit):
                    v = getattr(clim[w.label_date_local], c) if clim else np.nan
                    lo, hi = fit["clim_range"][c]
                    if not np.isfinite(v):
                        missing = True
                        break
                    if v < lo or v > hi:
                        clipped.append(f"{c}={v:.2f} outside training range [{lo:.2f}, {hi:.2f}]")
                    frame[c] = float(np.clip(v, lo, hi))
                if missing:
                    rows.append(rec)
                    continue
            p, lq = fit["model"].parts(fit_X(frame, fit))
            rec.update(status="ok", feature_set_used=name,
                       route="primary" if rank == 0 else f"fallback#{rank}",
                       feature_schema=" | ".join(fit.get("feature_schema") or []),
                       model_train_last=fit["train_last"], model_n_train=fit["n_train"],
                       p_rain_ge_0_2mm=float(p[0]), mean_estimate_mm=float(mean_estimate(fit["model"], p, lq)[0]))
            if experimental:
                rec.update(climate_used=fit.get("climate") or "none", climate_clipped="; ".join(clipped))
            for T in THRESHOLDS[1:]:
                rec[f"p_ge_{T:g}mm"] = float(exceed_prob(p, lq, T)[0])
            for t in LEVELS:
                rec[f"q{int(t * 100)}_mm"] = float(mixture_quantile(p, lq, t)[0])
            rec["cond_median_if_rain_mm"] = float(np.exp(lq[0, 9]))
            rec["cond_q90_if_rain_mm"] = float(np.exp(lq[0, 17]))
            for c in fit["cols"]:
                rec[f"input_{c}_mm"] = float(vals[c])   # NWP input total (uncalibrated), provenance only
            rec["source_detail"] = _issue_text(usable[usable.col.isin(fit["cols"])])
            used = usable[usable.col.isin(fit["cols"])]
            rec["availability_basis"] = ("observed" if "availability_basis" in used and
                                         (used.availability_basis == "observed").all() else ESTIMATED)
            rec["retrieved_at_utc"] = str(usable.retrieved_at_utc.max())
            rows.append(rec)
    return pd.DataFrame(rows)


def load_bundles(cfg, experimental: bool = False) -> dict:
    """Load and VALIDATE the artifacts. Default mode refuses anything that is not a clean no-ENSO artifact
    of a single identified run (pre-audit artifacts have no run_id / enso_status and are rejected)."""
    d = Path(cfg.data_dir) / ("models_enso_experimental" if experimental else "models")
    quarantine_root = provenance.archive_root(cfg)
    bad_hashes = enso_guard.quarantined_hashes(quarantine_root)
    listed = _manifest_artifacts(d, experimental)
    out = {}
    for lead in LEADS:
        p = d / f"hurdle_{lead}.joblib"
        if lead not in listed:
            continue              # only possible for experimental sets; reported as 'no_model_artifact' downstream
        if not experimental:
            rp = enso_guard.assert_servable_path(p, d, quarantine_root)
        else:
            rp = p
        data = rp.read_bytes()                # read ONCE: the bytes that are hashed are the bytes that are loaded
        digest = hashlib.sha256(data).hexdigest()
        if digest != listed[lead]["sha256"]:
            raise enso_guard.EnsoGuardError(f"{p}: bytes do not match manifest.json (the set was modified or mixed)")
        if not experimental and digest in bad_hashes:
            raise enso_guard.EnsoGuardError(f"{p}: byte-identical to a quarantined file (copy or hard link)")
        try:
            b = joblib.load(io.BytesIO(data))
        except Exception as exc:  # truncated / corrupt / foreign file: fail loudly, never skip silently
            raise enso_guard.EnsoGuardError(f"{p}: unreadable or corrupt artifact ({type(exc).__name__}: {exc})") from exc
        if not isinstance(b, dict):
            raise enso_guard.EnsoGuardError(f"{p}: not a perthrain artifact (type {type(b).__name__})")
        if experimental:
            v = enso_guard.metadata_violations(b, lead, expected_status=enso_guard.EXPERIMENTAL)
            if v:
                raise enso_guard.EnsoGuardError(f"{p}: invalid experimental artifact: {v}")
        else:
            enso_guard.assert_no_enso(b, str(p), expected_lead=lead)
        b["_artifact_path"] = str(p).replace("\\", "/")
        b["_artifact_sha256"] = digest
        meta = b.get("meta") or {}
        if meta.get("location") != cfg.location_name or meta.get("gauge_data_dir") != Path(cfg.data_dir).name:
            raise enso_guard.EnsoGuardError(
                f"{p}: artifact was trained for {meta.get('location')!r}/{meta.get('gauge_data_dir')!r}, "
                f"not {cfg.location_name!r}/{Path(cfg.data_dir).name!r}")
        out[lead] = b
    ids = {b["meta"]["run_id"] for b in out.values()}
    if len(ids) > 1:
        raise enso_guard.EnsoGuardError(f"{d}: artifacts from different runs {sorted(ids)}")
    man_id = listed.get("_run_id")
    if out and man_id and ids != {man_id}:
        raise enso_guard.EnsoGuardError(f"{d}: artifacts are from run {sorted(ids)}, manifest.json from {man_id}")
    return out


def _manifest_artifacts(d: Path, experimental: bool) -> dict:
    """The model set as declared by its manifest.json. A directory with model files but no readable manifest, with
    files missing from or not listed in the manifest, or (production) without all leads, is refused: an interrupted
    promotion or a hand edit must never be served as a partial set."""
    present = sorted(p.name for p in d.glob("hurdle_*.joblib")) if d.exists() else []
    man_p = d / "manifest.json"
    if not present and not man_p.exists():
        return {}
    if not man_p.exists():
        raise enso_guard.EnsoGuardError(f"{d}: model files without manifest.json; refusing an incomplete or "
                                        "unverifiable model set")
    try:
        man = json.loads(man_p.read_text(encoding="utf-8"))
        arts = {a["lead"]: {"file": a["file"], "sha256": a["sha256"]} for a in man["artifacts"]}
    except (ValueError, KeyError, TypeError) as exc:
        raise enso_guard.EnsoGuardError(f"{man_p}: unreadable manifest ({exc})") from exc
    files = {a["file"] for a in arts.values()}
    missing, unlisted = sorted(files - set(present)), sorted(set(present) - files)
    if missing or unlisted:
        raise enso_guard.EnsoGuardError(f"{d}: model files do not match manifest.json (missing: {missing}, "
                                        f"not listed: {unlisted})")
    if not experimental and set(arts) != set(LEADS):
        raise enso_guard.EnsoGuardError(f"{d}: manifest.json covers {sorted(arts)}; all of {list(LEADS)} are required")
    arts["_run_id"] = (man.get("meta") or {}).get("run_id")
    return arts


def _fmt_pct(x: float) -> str:
    if x >= 0.995:
        return ">99%"
    if x < 0.005:
        return "<1%"
    return f"{x * 100:.0f}%"


def station_info(cfg) -> dict:
    obs = pd.read_parquet(Path(cfg.data_dir) / "clean" / "observations.parquet", columns=[
        "station_id", "station_name", "station_latitude", "station_longitude"])
    st = {"station_id": obs.station_id.iloc[0], "station_name": obs.station_name.iloc[0],
          "latitude": float(obs.station_latitude.iloc[0]), "longitude": float(obs.station_longitude.iloc[0])}
    summary = json.loads((Path(cfg.reports_dir) / "run_summary.json").read_text())
    st["distance_km"] = float(summary["station"]["distance_km"])
    return st


def compute_predictions(cfg, horizon_days: int = 4, now: pd.Timestamp | None = None, experimental: bool = False,
                        refresh_live: bool = True, bundles: dict | None = None) -> dict:
    now = now or pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    if bundles is None:
        bundles = load_bundles(cfg, experimental)
    else:                                   # caller-supplied artifacts get the same validation as loaded ones
        for lead, b in bundles.items():
            if experimental:
                v = enso_guard.metadata_violations(b, lead, expected_status=enso_guard.EXPERIMENTAL)
                if v:
                    raise enso_guard.EnsoGuardError(f"supplied {lead} bundle invalid: {v}")
            else:
                enso_guard.assert_no_enso(b, f"supplied {lead} bundle", expected_lead=lead)
    if not bundles:
        raise SystemExit(f"no calibrated models for {cfg.location_name}; run `calibrate` first")
    st = station_info(cfg)
    windows = future_windows(now, cfg.timezone, horizon_days)
    fetcher = make_fetcher(cfg)
    sr, pr, problems = _fetch_live(cfg, fetcher, st["latitude"], st["longitude"], windows, now, refresh_live)
    model_meta = fetch_model_meta(fetcher, refresh_live)
    for m in MODEL_META_IDS:
        if m not in model_meta:
            problems.append(f"model metadata for {m} unavailable: its availability is the 6-hour estimate")
    fw = live_feature_table(cfg, sr, pr, windows, now, model_meta)
    clim, climate_info = None, {"status_only": ENSO_UNVERIFIED_NOTE}
    if experimental:
        indices = climate.load_indices(fetcher, cfg.raw_dir)
        clim = live_climate(indices, windows)
        climate_info = {"readings": climate.latest_reading(indices),
                        "phase": climate.describe_phase(indices["nino34"].value.iloc[-1], indices["iod"].value.iloc[-1]),
                        "note": "EXPERIMENTAL run with an UNVERIFIED climate-index feature; not the baseline."}
    pred = predict_windows(bundles, fw, windows, cfg.timezone, clim, experimental)
    return {"pred": pred, "station": st, "problems": problems, "bundles": bundles, "climate_info": climate_info,
            "now": now, "windows": windows, "features": fw}


def write_outputs(cfg, station: dict, pred: pd.DataFrame, now: pd.Timestamp, problems: list[str],
                  bundles: dict, climate_info: dict | None = None, experimental: bool = False) -> Path:
    out_dir = Path(cfg.data_dir) / ("predictions_enso_experimental" if experimental else "predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.tz_convert("UTC").strftime("%Y%m%dT%H%MZ")
    base = out_dir / f"forecast_{stamp}"
    k = 1
    while base.with_suffix(".csv").exists() or base.with_suffix(".md").exists():    # never overwrite a record
        k += 1
        base = out_dir / f"forecast_{stamp}_{k}"
    with open(base.with_suffix(".csv"), "x", newline="", encoding="utf-8") as fh:  # exclusive creation
        pred.to_csv(fh, index=False)
    tz = cfg.timezone
    any_b = next(iter(bundles.values()))
    meta = any_b.get("meta") or {}
    lines = [f"# Rain forecast for {cfg.location_name} - gauge {station['station_id']} {station['station_name']}",
             f"\nIssued {now.tz_convert(tz):%a %d %b %Y %H:%M %Z} ({now.tz_convert('UTC'):%Y-%m-%d %H:%M}Z). "
             f"Gauge is {float(station['distance_km']):.1f} km from {cfg.location_name}; totals describe "
             f"**that gauge**, not every street in the suburb.\n",
             f"Models: run `{meta.get('run_id')}` (created {meta.get('created_utc')}, enso_status "
             f"`{any_b.get('enso_status')}`, code sha256 `{str(meta.get('code_sha256'))[:12]}`).\n",
             "**How to read the dates:** the gauge day runs **9 am to 9 am**. Rain that falls between 9 am Friday and "
             "9 am Saturday is reported under **Saturday**.\n"]
    if climate_info and "status_only" in climate_info:
        lines.append(climate_info["status_only"] + "\n")
    elif climate_info:
        r = climate_info["readings"]
        lines.append(f"**Climate-index readings (BoM traditional Nino3.4 file, week to {r['nino34']['period_end']}):** "
                     f"Nino 3.4 {r['nino34']['value']:+.2f} C, IOD {r['iod']['value']:+.2f} C - "
                     f"{climate_info['phase']}. {climate_info['note']}\n")
    ok = pred[pred.status == "ok"] if len(pred) else pred
    if ok.empty:
        lines.append("_No complete forecast could be assembled for any upcoming window._")
    else:
        lines.append("## Best available prediction per gauge day (shortest lead)\n")
        lines.append("| gauge day (9am-9am) | lead (model, route) | chance of rain (>=0.2 mm) | >=1 mm | >=5 mm | >=10 mm | "
                     "median total | likely range (25-75%) | 80% range (10-90%) | expected total |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for d, g in ok.groupby("label_date_local"):
            r = g.sort_values("lead_group").iloc[0]
            lines.append(f"| {pd.Timestamp(d):%a %d %b} ({r.window_start_local:%a 9am} -> {r.window_end_local:%a 9am}) "
                         f"| {r.lead_group} ({r.feature_set_used}, {r.route}) | {_fmt_pct(r.p_rain_ge_0_2mm)} | "
                         f"{_fmt_pct(r.p_ge_1mm)} | {_fmt_pct(r.p_ge_5mm)} | {_fmt_pct(r.p_ge_10mm)} | {r.q50_mm:.1f} mm | "
                         f"{r.q25_mm:.1f}-{r.q75_mm:.1f} mm | {r.q10_mm:.1f}-{r.q90_mm:.1f} mm | "
                         f"{r.mean_estimate_mm:.1f} mm |")
        lines.append("\n`median total` counts dry outcomes: it is 0 whenever the chance of rain is under about 50%. "
                     "`expected total` is the probability-weighted average, which is pulled up by the small chance "
                     "of heavy rain. `fallback#k` means the primary model's inputs were incomplete and the k-th model "
                     "in the routing order was used.\n")
        lines.append("## All leads (longer leads are less certain; compare them to see how the forecast is changing)\n")
        cols = ["label_date_local", "lead_group", "feature_set_used", "route", "p_rain_ge_0_2mm", "p_ge_1mm",
                "p_ge_5mm", "p_ge_10mm", "q25_mm", "q50_mm", "q75_mm", "mean_estimate_mm"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for _, r in ok.sort_values(["label_date_local", "lead_group"]).iterrows():
            cells = []
            for c in cols:
                v = r[c]
                cells.append(f"{v:%d %b}" if c == "label_date_local" else
                             (f"{v:.2f}" if isinstance(v, float) else str(v)))
            lines.append("| " + " | ".join(cells) + " |")
    miss = pred[pred.status != "ok"] if len(pred) else pred
    if len(miss):
        lines.append("\n## Not predicted\n")
        lines.append("No calibrated model could serve these (inputs incomplete or not yet published, or no artifact). "
                     "No uncalibrated raw-forecast value is substituted.\n")
        for _, r in miss.iterrows():
            lines.append(f"* {pd.Timestamp(r.label_date_local):%a %d %b} {r.lead_group}: {r.status}"
                         + (f" ({r.route})" if isinstance(r.get('route'), str) else ""))
    skill = {lead: (b.get("holdout") or [{}])[0] for lead, b in bundles.items() if (b.get("holdout") or [{}])[0]}
    if skill:
        lines.append("\n## How much to trust this (held-out testing, days the model never saw)\n")
        for lead, h in skill.items():
            lines.append(f"* {lead}: primary model `{h.get('feature_set')}`; on {int(h['n'])} unseen days "
                         f"({h['test_first']} to {h['test_last']}) its rain-chance skill vs plain climatology was "
                         f"BSS {h.get('BSS_1', float('nan')):+.2f} for >=1 mm and its distribution skill "
                         f"{h.get('pinball_skill_vs_clim', float('nan')):+.0%}. Fallback models were not scored "
                         f"separately on these days. See calibration_report.md.")
    if problems:
        lines.append("\n## Fetch problems\n" + "\n".join(f"* {p}" for p in problems))
    lines.append("\n_Statistical calibration of model forecasts against a single gauge. Not an official forecast: for "
                 "warnings and decisions use the Bureau of Meteorology and radar._")
    text = "\n".join(lines) + "\n"
    with open(base.with_suffix(".md"), "x", encoding="utf-8") as fh:
        fh.write(text)
    # 'latest' is a copy of the record above, written last so it can never point at an incomplete record
    pred.to_csv(out_dir / "latest.csv", index=False)
    (out_dir / "latest.md").write_text(text, encoding="utf-8")
    return out_dir / "latest.md"


def run_predict(cfg, horizon_days: int = 4, now: pd.Timestamp | None = None,
                use_climate: bool = False, refresh_live: bool = True) -> pd.DataFrame:
    r = compute_predictions(cfg, horizon_days, now, experimental=use_climate, refresh_live=refresh_live)
    path = write_outputs(cfg, r["station"], r["pred"], r["now"], r["problems"], r["bundles"], r["climate_info"],
                         experimental=use_climate)
    log.info("%s: wrote %s (%d ok predictions)", cfg.location_name, path, int((r["pred"].status == "ok").sum()))
    return r["pred"]
