"""Prospective track: an append-only, hash-chained ledger of every forecast as issued, and later scoring against
gauge observations without touching the ledger.

Ledger: data/challenge/ledger.jsonl (one JSON object per line). Each record carries `prev_hash` and `record_hash`
(SHA-256 of the record without `record_hash`), so editing or deleting a record inside the chain is detected. The
file is only ever opened in append mode.

A hash chain alone cannot detect deletion of the FINAL records (the remaining prefix is still a valid chain). After
every append the record count and head hash are written to a separate append-only checkpoint file
(`ledger.jsonl.checkpoints`); reading fails if the ledger is shorter than, or diverges from, its last checkpoint. The
challenge export also publishes the head hash, so a copy exists outside this machine (deleting records and
checkpoints together locally would otherwise go unnoticed). Only one writer at a time: appends take an exclusive
lock file (`ledger.jsonl.lock`); a second writer is refused rather than allowed to fork the chain.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tomllib

from perthrain import observations
from perthrain.calibrate import THRESHOLDS
from perthrain.config import load_configs
from perthrain.pipeline import ftp_month_url, make_fetcher
from perthrain.predict import compute_predictions
from perthrain.stations import STATE_FOLDERS, ftp_folder_name
from challenge import champion, sources

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "challenge"
LEDGER = DATA / "ledger.jsonl"
SETTINGS = Path(__file__).resolve().parent / "settings.toml"


class LedgerError(RuntimeError):
    pass


def settings() -> dict:
    return tomllib.loads(SETTINGS.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ ledger I/O
def _hash(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != "record_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def checkpoint_path(path: Path) -> Path:
    return path.with_name(path.name + ".checkpoints")


def lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


def _read_checkpoints(path: Path) -> list[dict]:
    cp = checkpoint_path(path)
    if not cp.exists():
        return []
    out = []
    for i, line in enumerate(cp.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            c = json.loads(line)
            c = {"count": int(c["count"]), "head": str(c["head"]), "utc": c.get("utc")}
        except (ValueError, KeyError, TypeError) as exc:
            raise LedgerError(f"checkpoint line {i} is unreadable ({exc}); the checkpoint file was damaged") from exc
        if out and c["count"] < out[-1]["count"]:
            raise LedgerError(f"checkpoint line {i} goes backwards ({c['count']} < {out[-1]['count']})")
        out.append(c)
    return out


def read_ledger(path: Path = LEDGER) -> list[dict]:
    if not path.exists():
        if _read_checkpoints(path):
            raise LedgerError(f"{path.name} is missing but its checkpoints exist: the ledger was deleted")
        return []
    recs, prev = [], "genesis"
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LedgerError(f"ledger line {i} is not valid JSON (a write was interrupted?). Nothing has been "
                              f"changed; inspect the end of {path.name} before continuing") from exc
        if r.get("prev_hash") != prev or _hash(r) != r.get("record_hash"):
            raise LedgerError(f"ledger line {i} fails the hash chain: the file was edited or truncated")
        prev = r["record_hash"]
        recs.append(r)
    cps = _read_checkpoints(path)
    if cps:
        last = cps[-1]
        if len(recs) < last["count"]:
            raise LedgerError(f"ledger truncated: the last checkpoint records {last['count']} records, "
                              f"the file holds {len(recs)}")
        if last["count"] and recs[last["count"] - 1]["record_hash"] != last["head"]:
            raise LedgerError(f"ledger diverges from its checkpoint at record {last['count']}")
    return recs


def head(path: Path = LEDGER) -> dict:
    """Record count and head hash of a verified ledger (published with the challenge export)."""
    recs = read_ledger(path)
    return {"count": len(recs), "head_hash": recs[-1]["record_hash"] if recs else None}


def ensure_checkpoint(path: Path = LEDGER) -> dict:
    """Write a checkpoint for an existing ledger that has none (verifies the chain first; changes no record)."""
    with writer_lock(path):
        recs = read_ledger(path)
        cps = _read_checkpoints(path)
        h = {"count": len(recs), "head": recs[-1]["record_hash"] if recs else "genesis",
             "utc": f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}"}
        if not cps or cps[-1]["count"] != h["count"]:
            with open(checkpoint_path(path), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(h) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        return h


@contextlib.contextmanager
def writer_lock(path: Path = LEDGER):
    """Exclusive single-writer lock (lock file created atomically). Refuses, never waits or steals."""
    lp = lock_path(path)
    lp.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        info = lp.read_text(encoding="utf-8", errors="replace") if lp.exists() else ""
        raise LedgerError(f"ledger is locked by another writer ({lp.name}: {info.strip()}). If no other collection "
                          "is running, the lock is stale: remove it by hand") from None
    try:
        os.write(fd, f"pid {os.getpid()} since {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}".encode())
        os.close(fd)
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(lp)


def append(records: list[dict], path: Path = LEDGER, *, lock: bool = True) -> int:
    """Append records as one buffered write (flushed and synced), then a checkpoint. ``lock=False`` only for callers
    that already hold :func:`writer_lock`."""
    ctx = writer_lock(path) if lock else contextlib.nullcontext()
    with ctx:
        existing = read_ledger(path)                   # verifies chain and checkpoint before writing anything
        prev = existing[-1]["record_hash"] if existing else "genesis"
        lines = []
        for r in records:
            r = dict(r, prev_hash=prev)
            r["record_hash"] = _hash(r)
            lines.append(json.dumps(r, sort_keys=True, default=str) + "\n")
            prev = r["record_hash"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:  # append only; never rewritten
            fh.write("".join(lines))
            fh.flush()
            os.fsync(fh.fileno())
        with open(checkpoint_path(path), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"count": len(existing) + len(records), "head": prev,
                                 "utc": f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}"}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    return len(records)


# ------------------------------------------------------------------------------------------------ collection
def schedule_slot(now: pd.Timestamp, times_utc: list[str]) -> pd.Timestamp:
    """Latest scheduled issue time at or before now."""
    cands = []
    for d in (now.normalize() - pd.Timedelta(days=1), now.normalize()):
        for t in times_utc:
            hh, mm = map(int, t.split(":"))
            cands.append(d + pd.Timedelta(hours=hh, minutes=mm))
    return max(c for c in cands if c <= now)


def _model_meta(model_meta_id: str) -> dict:
    res = sources.fetcher().get_json(f"https://api.open-meteo.com/data/{model_meta_id}/static/meta.json", {},
                                     "open_meteo/challenge/meta", refresh=True, label=f"meta {model_meta_id}")
    b = res.body if res.ok else {}
    ts = lambda k: pd.Timestamp(b[k], unit="s", tz="UTC") if b.get(k) else None  # noqa: E731
    return {"last_run_init_utc": ts("last_run_initialisation_time"), "last_run_available_utc": ts("last_run_availability_time"),
            "retrieved_at_utc": res.retrieved_at_utc}


def _base(issue, cfg, st, w, lead):
    return {"issue_id": issue["issue_id"], "issued_at_utc": issue["issued_at_utc"], "schedule_slot_utc": issue["slot"],
            "location": cfg.location_name, "gauge_id": st["station_id"], "gauge_name": st["station_name"],
            "gauge_distance_km": round(float(st["distance_km"]), 1), "label_date": str(w.label_date_local),
            "window_start_utc": str(w.window_start_utc), "window_end_utc": str(w.window_end_utc), "lead": lead,
            "scheduled_issue": issue["scheduled"]}


def collect(now: pd.Timestamp | None = None, force: bool = False) -> int:
    """Collect one issue. The writer lock is held for the whole collection, so two collections cannot both decide
    a slot is still open and then both append."""
    with writer_lock(LEDGER):
        return _collect(now, force)


def _collect(now: pd.Timestamp | None = None, force: bool = False) -> int:
    s = settings()["schedule"]
    freeze = champion.verify()
    now = (now or pd.Timestamp.now(tz="UTC")).floor("s")
    slot = schedule_slot(now, s["issue_times_utc"])
    late = (now - slot) / pd.Timedelta(minutes=1)
    if late > s["max_delay_minutes"] and not force:
        raise LedgerError(f"now is {late:.0f} min after the {slot:%H:%M} UTC slot (limit {s['max_delay_minutes']}); "
                          "run at the scheduled time or pass --force (the record shows the actual issue time)")
    done = {(r["schedule_slot_utc"], r["location"]) for r in read_ledger()}
    # an off-schedule (forced) issue never occupies the scheduled slot; it is labelled and kept separately
    slot_label = f"{slot:%Y-%m-%dT%H:%MZ}" if late <= s["max_delay_minutes"] else f"unscheduled {now:%Y-%m-%dT%H:%MZ}"
    issue = {"issue_id": f"{now:%Y%m%dT%H%M%SZ}", "issued_at_utc": f"{now:%Y-%m-%dT%H:%M:%SZ}", "slot": slot_label,
             "scheduled": slot_label == f"{slot:%Y-%m-%dT%H:%MZ}"}
    metas = {m: _model_meta(v["metadata_id"]) for m, v in sources.CHALLENGERS.items()}
    records = []
    for cfg in load_configs(str(ROOT / "config.toml")):
        if (issue["slot"], cfg.location_name) in done:
            print(f"{cfg.location_name}: slot {issue['slot']} already issued; skipped (never re-issued)")
            continue
        r = compute_predictions(cfg, horizon_days=s["horizon_days"], now=now, refresh_live=True)
        st, pred, fw = r["station"], r["pred"], r["features"]
        loc = freeze["locations"][Path(cfg.data_dir).name]
        windows = r["windows"]
        for _, w in windows.iterrows():
            for lead, n in (("day1", 1), ("day2", 2), ("day3", 3)):
                base = _base(issue, cfg, st, w, lead)
                p = pred[(pred.label_date_local == w.label_date_local) & (pred.lead_group == lead)]
                p = p.iloc[0] if len(p) else None
                ok = p is not None and p.status == "ok"
                records.append({**base, "competitor": "champion", "kind": "champion",
                                "status": "ok" if ok else "missing",
                                "missing_reason": None if ok else (p.status if p is not None else "no row"),
                                "median_mm": float(p.q50_mm) if ok else None,
                                "expected_mm": float(p.mean_estimate_mm) if ok else None,
                                "p10_mm": float(p.q10_mm) if ok else None, "p90_mm": float(p.q90_mm) if ok else None,
                                "probabilities": {f"ge_{T:g}": float(p[c]) for T, c in zip(
                                    THRESHOLDS, ("p_rain_ge_0_2mm", "p_ge_1mm", "p_ge_5mm", "p_ge_10mm"))} if ok else None,
                                "route": p.route if p is not None else None,
                                "model_used": p.feature_set_used if ok else None,
                                "model_run_id": loc["run_id"], "artifact_sha256": p.artifact_sha256 if p is not None else None,
                                "inputs": p.source_detail if ok else None,
                                "availability_basis": p.get("availability_basis") if ok else None,
                                "retrieved_at_utc": p.retrieved_at_utc if ok else None})
                # existing raw benchmarks: the champion's own live inputs (same as-of gate)
                for col in ["sr_ecmwf_ifs", "pr_ecmwf_ifs025", "pr_jma_gsm", "pr_ncep_gfs_global"]:
                    f = fw[(fw.label_date_local == w.label_date_local) & (fw.lead_day == n) & (fw.col == col)]
                    f = f.iloc[0] if len(f) else None
                    usable = f is not None and bool(f.usable)
                    records.append({**base, "competitor": col, "kind": "raw benchmark",
                                    "status": "ok" if usable else "missing",
                                    "missing_reason": None if usable else ("not requested at this lead" if f is None else
                                                                          (f.timing_reason or "incomplete or not eligible")),
                                    "total_mm": float(f.forecast_precip_mm) if usable else None,
                                    "latest_run_utc": str(f.latest_issue_utc) if f is not None else None,
                                    "availability_basis": (f.get("availability_basis") if f is not None else None),
                                    "retrieved_at_utc": f.retrieved_at_utc if f is not None else None})
                records += _challenger_records(base, st, w, n, now, metas)
        print(f"{cfg.location_name}: {sum(1 for x in records if x['location'] == cfg.location_name)} records")
    return append(records, LEDGER, lock=False) if records else 0


def _challenger_records(base, st, w, n, now, metas) -> list[dict]:
    out = []
    win = pd.DataFrame({"window_start_utc": [pd.Timestamp(w.window_start_utc)], "window_end_utc": [pd.Timestamp(w.window_end_utc)]})
    for model, meta in sources.CHALLENGERS.items():
        rec = {**base, "competitor": model, "kind": "challenger", "status": "missing", "total_mm": None}
        mm = metas[model]
        if n == 1:
            run = sources.sr_issue_time(win.window_start_utc, 1).iloc[0]
            rec["named_run_utc"] = str(run)
            if run + pd.Timedelta(hours=sources.LATENCY_H) > now:
                rec["missing_reason"] = "required 12 UTC run not issued/published before the issue time"
            else:
                sr = sources.fetch_single_run(model, float(st["latitude"]), float(st["longitude"]), run, 5, refresh=True,
                                              subdir="open_meteo/challenge/live_single_runs")
                if sr is None:
                    rec["missing_reason"] = "single run not available from the archive at issue time"
                else:
                    t = sources.window_totals_single_run(sr, win, 1).iloc[0]
                    rec.update(_fill(t, "explicit run requested; HTTP 200 at issue time"))
        else:
            pr = sources.fetch_previous_runs(model, float(st["latitude"]), float(st["longitude"]),
                                             win.window_start_utc[0].tz_localize(None).normalize(),
                                             win.window_end_utc[0].tz_localize(None).normalize(), [n], refresh=True,
                                             subdir="open_meteo/challenge/live_previous_runs")
            t = sources.window_totals_previous_runs(pr, win, cutoff_rule="as_of", as_of=now).iloc[0] if len(pr) else None
            if t is None or pd.isna(t.latest_run_utc):
                rec["missing_reason"] = "no response"
            else:
                rec["named_run_utc"] = str(t.latest_run_utc)
                after_meta = mm["last_run_init_utc"] is not None and t.latest_run_utc > mm["last_run_init_utc"]
                if not t.published_before_cutoff or after_meta:
                    rec["missing_reason"] = "required run not issued/published before the issue time"
                else:
                    rec.update(_fill(t, f"run named by the offset rule; model metadata last run "
                                        f"{mm['last_run_init_utc']} (checked {mm['retrieved_at_utc']})"))
        out.append(rec)
        if n > 1:                                   # same 12 UTC single-run rule as the champion's sr input
            r12 = {**base, "competitor": f"{model}_12z", "kind": "challenger", "status": "missing", "total_mm": None}
            run = sources.sr_issue_time(win.window_start_utc, n).iloc[0]
            r12["named_run_utc"] = str(run)
            if run + pd.Timedelta(hours=sources.LATENCY_H) > now:
                r12["missing_reason"] = "required 12 UTC run not issued/published before the issue time"
            else:
                sr = sources.fetch_single_run(model, float(st["latitude"]), float(st["longitude"]), run, 5, refresh=True,
                                              subdir="open_meteo/challenge/live_single_runs")
                if sr is None:
                    r12["missing_reason"] = "single run not available from the archive at issue time"
                else:
                    r12.update(_fill(sources.window_totals_single_run(sr, win, n).iloc[0],
                                     "explicit run requested; HTTP 200 at issue time"))
            out.append(r12)
    return out


def _fill(t, evidence) -> dict:
    if pd.isna(t.total_mm):
        return {"status": "missing", "missing_reason": "incomplete window or negative hour", "latest_run_utc": str(t.latest_run_utc)}
    return {"status": "ok", "missing_reason": None, "total_mm": float(t.total_mm), "latest_run_utc": str(t.latest_run_utc),
            "lead_hours_to_window_start": float(t.lead_hours_to_window_start), "availability_evidence": evidence,
            "grid_lat": t.grid_latitude, "grid_lon": t.grid_longitude, "retrieved_at_utc": t.retrieved_at_utc,
            "source_url": t.source_url}


# ------------------------------------------------------------------------------------------------ observations
def refresh_observations() -> pd.DataFrame:
    """Recent gauge readings (current and previous month) for every gauge in the ledger, from BoM's public FTP files."""
    frames = []
    seen = set()
    for cfg in load_configs(str(ROOT / "config.toml")):
        st = json.loads((Path(cfg.reports_dir) if Path(cfg.reports_dir).is_absolute() else ROOT / cfg.reports_dir)
                        .joinpath("run_summary.json").read_text())["station"]
        if st["station_id"] in seen:
            continue
        seen.add(st["station_id"])
        f = make_fetcher(cfg)
        today = pd.Timestamp.now(tz=cfg.timezone).date()
        first = (pd.Timestamp(today) - pd.offsets.MonthBegin(2)).date()
        state, folder = STATE_FOLDERS.get(st["state"], st["state"].lower()), ftp_folder_name(st["station_name"])
        monthly = []
        for ym in [p.strftime("%Y%m") for p in pd.period_range(first, today, freq="M")]:
            url = ftp_month_url(state, folder, ym)
            res = f.get_file(url, ROOT / "data" / "challenge" / "obs" / folder / f"{folder}-{ym}.csv", refresh=True,
                             label=f"challenge obs {folder} {ym}")
            monthly.append((url, res.body if res.ok else None, res.retrieved_at_utc, ym))
        obs, _ = observations.build_ftp_observations(monthly, st, cfg.timezone, first, today, cfg.suspicious_daily_mm)
        frames.append(obs)
    out = pd.concat(frames, ignore_index=True)
    DATA.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DATA / "observations_recent.parquet", index=False)
    return out


def scored_table(recs: list[dict], obs: pd.DataFrame) -> pd.DataFrame:
    """One row per (issue, gauge, window, lead): every competitor's forecast as issued, plus the observation once
    available. Perth and Ocean Reef share a gauge: each gauge is scored once (the first location issued)."""
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    first_loc = df.groupby("gauge_id").location.first()
    df = df[df.location == df.gauge_id.map(first_loc)]
    keys = ["issue_id", "gauge_id", "gauge_name", "location", "label_date", "window_start_utc", "window_end_utc", "lead"]
    wide = df[keys].drop_duplicates().reset_index(drop=True)
    for comp, g in df.groupby("competitor"):
        g = g.set_index(keys)
        if comp == "champion":
            for c, src in (("champion_median_mm", "median_mm"), ("champion_expected_mm", "expected_mm"),
                           ("champion_p10_mm", "p10_mm"), ("champion_p90_mm", "p90_mm")):
                wide = wide.join(g[src].rename(c), on=keys)
            probs = g.probabilities.apply(lambda d: d if isinstance(d, dict) else {})
            for T in THRESHOLDS:
                wide = wide.join(probs.apply(lambda d: d.get(f"ge_{T:g}")).rename(f"champion_p_ge_{T:g}"), on=keys)
        else:
            wide = wide.join(g.total_mm.rename(f"fc_{comp}_mm"), on=keys)
    o = obs.copy()
    o["gauge_id"] = o.station_id.astype(str).str.zfill(6)
    o["window_start_utc"] = pd.to_datetime(o.window_start_utc, utc=True).astype(str)
    o["window_end_utc"] = pd.to_datetime(o.window_end_utc, utc=True).astype(str)
    wide["window_start_utc"] = pd.to_datetime(wide.window_start_utc, utc=True).astype(str)
    wide["window_end_utc"] = pd.to_datetime(wide.window_end_utc, utc=True).astype(str)
    wide = wide.merge(o[["gauge_id", "window_start_utc", "window_end_utc", "observed_precip_mm"]],
                      on=["gauge_id", "window_start_utc", "window_end_utc"], how="left")
    return wide
