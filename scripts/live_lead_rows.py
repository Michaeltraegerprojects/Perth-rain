"""DIAGNOSTIC: per served row of the 30 Sep 14:56-14:57Z predictions - source API/variable, issue time (explicit, or
the verified offset rule), valid window, retrieval time, lead hours, and whether every contributing run had been
issued and (est.) published before the prediction was made. Rows whose previous-runs inputs pass that timing test are
then checked value-by-value against the Single Runs API archive for the exact runs the offset rule names.

Replays cached live inputs (no network) for the timing table; the verification step makes GET requests to the
Single Runs API and caches them under data/raw/open_meteo/verify_live/ (separate from the live cache).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import os  # noqa: E402

os.chdir(ROOT)
from perthrain import forecasts  # noqa: E402
from perthrain import predict as P  # noqa: E402
from perthrain.config import load_configs  # noqa: E402
from perthrain.calibrate import FEATURE_SETS  # noqa: E402
from perthrain.pipeline import make_fetcher  # noqa: E402

LATENCY_H = 6.0
rows, verify = [], []
cfgs = {c.data_dir.name: c for c in load_configs("config.toml")}
for loc, cfg in cfgs.items():
    served_file = sorted((cfg.data_dir / "predictions").glob("forecast_*.csv"))[-1]
    now = pd.Timestamp(pd.to_datetime(served_file.stem.split("_")[1], format="%Y%m%dT%H%MZ"), tz="UTC")
    served = pd.read_csv(served_file)
    r = P.compute_predictions(cfg, now=now, refresh_live=False)
    fw = r["features"]
    for _, s in served[served.status == "ok"].iterrows():
        n = int(s.lead_group[-1])
        for col in FEATURE_SETS[s.feature_set_used]["cols"]:
            f = fw[(fw.label_date_local.astype(str) == s.label_date_local) & (fw.lead_day == n) & (fw.col == col)].iloc[0]
            explicit = pd.notna(f.forecast_issue_time_utc)
            issue_max = f.forecast_issue_time_utc if explicit else f.issue_time_inferred_max_utc
            issue_min = f.forecast_issue_time_utc if explicit else f.issue_time_inferred_min_utc
            ok_time = bool(issue_max + pd.Timedelta(hours=LATENCY_H) <= now)
            rows.append({
                "location": loc, "gauge_day_label": s.label_date_local, "lead": s.lead_group,
                "model_used": s.feature_set_used, "input": col,
                "source_api_variable": ("Single Runs API / precipitation (run=" + f"{issue_max:%Y-%m-%dT%H:%MZ})")
                if explicit else f"Previous Runs API / precipitation_{f.source_lead_offset}",
                "issue_time": (f"{issue_max:%Y-%m-%d %H:%MZ} (explicit run)" if explicit else
                               f"not provided; offset rule gives runs {issue_min:%m-%d %HZ}..{issue_max:%m-%d %HZ}"),
                "window_utc": f"{f.window_start_utc:%Y-%m-%d %H:%M} -> {f.window_end_utc:%Y-%m-%d %H:%M}",
                "retrieved_utc": str(f.retrieved_at_utc), "predicted_at_utc": f"{now:%Y-%m-%d %H:%M}",
                "lead_h_to_window_start": round((f.window_start_utc - issue_max) / pd.Timedelta(hours=1), 1),
                "lead_h_to_window_end": round((f.window_end_utc - issue_max) / pd.Timedelta(hours=1), 1),
                "latest_run_issued_and_published_before_prediction": ok_time})
            if ok_time and not explicit:
                verify.append((loc, cfg, s.label_date_local, n, col, f.model, f.window_start_utc, f.window_end_utc, now))

tab = pd.DataFrame(rows)
# row verdict from timing
tab["row_timing_ok"] = tab.groupby(["location", "gauge_day_label", "lead"])[
    "latest_run_issued_and_published_before_prediction"].transform("all")

# ---- value check of timing-OK previous-runs inputs against the Single Runs archive
st_cache = {}
checks = []
for loc, cfg, day, n, col, model, ws, we, now in verify:
    st = P.station_info(cfg)
    f = make_fetcher(cfg)
    live_meta = [m for m in (ROOT / "data/raw/open_meteo/live/previous_runs").glob("*.meta.json")
                 if json.loads(m.read_text())["params"]["models"] == model
                 and abs(json.loads(m.read_text())["params"]["latitude"] - st["latitude"]) < 1e-6]
    body = json.loads(live_meta[0].with_name(live_meta[0].name.replace(".meta", "")).read_text())
    h = pd.DataFrame(body["hourly"])
    h["time"] = pd.to_datetime(h.time).dt.tz_localize("UTC")
    h = h.set_index("time")[f"precipitation_previous_day{n}"]
    hours = pd.date_range(ws + pd.Timedelta(hours=1), we, freq="h")
    runs = forecasts.inferred_run_time(pd.Series(hours, index=hours), model, n)
    tested = matched = 0
    for run in sorted(set(runs)):
        p = forecasts.single_run_params(st["latitude"], st["longitude"], model, run, 6)
        res = f.get_json(forecasts.SINGLE_RUNS_URL, p, "open_meteo/verify_live", label=f"verify {model} {run}")
        if not res.ok:
            checks.append({"location": loc, "gauge_day_label": day, "lead": f"day{n}", "input": col,
                           "run": str(run), "status": f"run unavailable: {res.error}"[:120]})
            continue
        sr = forecasts.parse_single_run(res.body, model, run, res.url, res.params, res.retrieved_at_utc)
        sr = sr.set_index("valid_end_utc").precip_mm
        for t in runs.index[runs == run]:
            if t in sr.index and pd.notna(sr[t]) and pd.notna(h.get(t, np.nan)):
                tested += 1
                matched += int(np.isclose(sr[t], h[t], atol=0.051))
    checks.append({"location": loc, "gauge_day_label": day, "lead": f"day{n}", "input": col,
                   "hours_tested": tested, "hours_matching_named_run": matched,
                   "status": "verified" if tested == len(hours) and matched == tested else "NOT verified"})
chk = pd.DataFrame(checks)
out = ROOT / "reports"
tab.to_csv(out / "live_lead_audit_rows.csv", index=False)
chk.to_csv(out / "live_lead_audit_value_checks.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70); pd.set_option("display.max_rows", 200)
print(tab[tab.location.isin(["perth", "clarkson", "fremantle"])][
    ["location", "gauge_day_label", "lead", "input", "source_api_variable", "issue_time", "lead_h_to_window_start",
     "lead_h_to_window_end", "latest_run_issued_and_published_before_prediction"]].to_string(index=False))
print()
print(chk.to_string(index=False))
