"""DIAGNOSTIC provenance audit of Previous Runs inputs (live and historical). Writes reports/live_provenance_audit.md.

Part A  observed behaviour of the LIVE responses (preserved copy in the withdrawal archive): for FUTURE valid hours
        (offset-rule run issued after retrieval) compare the returned values, on NON-ZERO hours only, with every
        archived single run that existed at retrieval.
Part B  value verification of the TIMING-PASS live rows against the exact runs the offset rule names.
Part C  historical sample: rainy training/held-out windows since 2026-04-02 (start of the single-run archive for these
        models), same comparison against the named runs.
Single-run requests are GETs to the public Single Runs API, cached under data/raw/open_meteo/verify_live|verify_hist.
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
from perthrain.config import load_configs  # noqa: E402
from perthrain.pipeline import make_fetcher  # noqa: E402

ARCH = sorted(ROOT.glob("archive/WITHDRAWN_*_pre_timing_fix_predictions"))[-1]
LIVE = ARCH / "inputs" / "open_meteo_live_cache" / "previous_runs"
TOL = 0.051
cfg = load_configs("config.toml")[0]
F = make_fetcher(cfg)
_runs = {}


def single_run(model, lat, lon, run, sub):
    key = (model, lat, lon, run)
    if key not in _runs:
        p = forecasts.single_run_params(lat, lon, model, run, 6)
        res = F.get_json(forecasts.SINGLE_RUNS_URL, p, sub, label=f"audit {model} {run}")
        _runs[key] = (forecasts.parse_single_run(res.body, model, run, res.url, res.params, res.retrieved_at_utc)
                      .set_index("valid_end_utc").precip_mm if res.ok else None)
    return _runs[key]


def load(meta_p):
    meta = json.loads(meta_p.read_text())
    body = json.loads(meta_p.with_name(meta_p.name.replace(".meta", "")).read_text())
    h = pd.DataFrame(body["hourly"])
    h.index = pd.to_datetime(h.pop("time")).dt.tz_localize("UTC")
    return meta, h


def full_url(meta):
    from urllib.parse import urlencode
    return meta["url"] + "?" + urlencode(meta["params"])


out = ["# Provenance audit of Previous Runs inputs\n",
       "## 1. Documented semantics (Open-Meteo Previous Runs API page, captured 2026-09-30)\n",
       "> \"_previous_day0 is the current model run (equivalent to the live Forecast API). _previous_day1 is the value "
       "that was predicted 24 hours before valid time, _previous_day2 48 hours before, and so on up to day 7. For local "
       "models with shorter forecast horizons (2–5 days), only offsets within that horizon are populated.\"\n",
       "The page does **not** say what is returned when valid time − 24·N h is later than the request time. The run "
       "that produces a given hour (floor_6h(ceil_to_native_step(t)) − 24·N h) is our rule, verified historically "
       "against the Single Runs archive (probe, 100 % of comparable hours) - it is not stated by the API.\n",
       "## 2. Observed behaviour of the live responses (preserved evidence, no inference)\n"]

# ---------------------------------------------------------------- Part A: future hours in the live responses
metas = [m for m in sorted(LIVE.glob("*.meta.json"))
         if json.loads(m.read_text())["params"]["start_date"] >= "2026-09-30"]
a_rows, examples = [], []
for mp in metas:
    meta, h = load(mp)
    p = meta["params"]
    model, lat, lon = p["models"], p["latitude"], p["longitude"]
    ret = pd.Timestamp(meta["retrieved_at_utc"])
    cands = [pd.Timestamp("2026-09-29T12:00Z") + pd.Timedelta(hours=6 * k) for k in range(5)]   # 29/12Z .. 30/12Z
    cands = [c for c in cands if c <= ret]
    for n in (1, 2, 3):
        col = f"precipitation_previous_day{n}"
        rule = forecasts.inferred_run_time(pd.Series(h.index, index=h.index), model, n)
        fut = h[(rule > ret) & h[col].notna()]
        nz = fut[fut[col] > 0]
        rec = {"model": model, "lat": lat, "offset": f"day{n}", "retrieved_utc": meta["retrieved_at_utc"],
               "future_hours_returned": len(fut), "future_nonzero_hours": len(nz)}
        for c in cands:
            sr = single_run(model, lat, lon, c, "open_meteo/verify_live")
            if sr is None:
                rec[f"match_{c:%d%HZ}"] = "not archived"
                continue
            common = [t for t in nz.index if t in sr.index and pd.notna(sr[t])]
            m = int(sum(np.isclose(nz.loc[t, col], sr[t], atol=TOL) for t in common))
            rec[f"match_{c:%d%HZ}"] = f"{m}/{len(common)}"
        a_rows.append(rec)
        for t in nz.index[:2]:
            ex = {"model": model, "lat": lat, "offset": f"day{n}", "valid_end_utc": f"{t:%Y-%m-%d %H:%MZ}",
                  "rule_run": f"{rule[t]:%Y-%m-%d %HZ}", "returned_mm": nz.loc[t, col]}
            for c in cands:
                sr = single_run(model, lat, lon, c, "open_meteo/verify_live")
                ex[f"run_{c:%d%HZ}"] = None if sr is None or t not in sr.index else sr[t]
            examples.append(ex)
A = pd.DataFrame(a_rows)
E = pd.DataFrame(examples)
out.append("Requests (exact, from the cache metadata):\n")
for mp in metas:
    meta, _ = load(mp)
    out.append(f"* retrieved {meta['retrieved_at_utc']}: `{full_url(meta)}`")
out.append("\n**Observed:** for valid hours whose rule-named run was issued AFTER the retrieval time, the API still "
           "returned values. Counts, and - on NON-ZERO returned hours only - how many equal (|Δ| ≤ 0.05 mm) each "
           "archived single run that existed at retrieval (runs 29 Sep 12Z … 30 Sep 12Z; 'not archived' = HTTP 400 "
           "from the Single Runs API):\n")


def md(df):
    if df.empty:
        return "_(none)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else
                                       (f"{v:.2f}" if isinstance(v, float) else str(v)) for v in r.values) + " |")
    return "\n".join(lines) + "\n"


out.append(md(A))
out.append("Non-zero examples (returned value vs each candidate run at the same valid hour):\n")
out.append(md(E))

# ---------------------------------------------------------------- Part B: timing-pass rows, named-run check
rows = pd.read_csv(ROOT / "reports" / "live_lead_audit_rows.csv")
tp = rows[rows.latest_run_issued_and_published_before_prediction & rows.input.str.startswith("pr_")]
tp = tp[tp.groupby(["location", "gauge_day_label", "lead"]).latest_run_issued_and_published_before_prediction
        .transform("all")]
cfgs = {c.data_dir.name: c for c in load_configs("config.toml")}
b_rows = []
for (loc, day, lead, inp), _g in tp.groupby(["location", "gauge_day_label", "lead", "input"]):
    model, n = inp[3:], int(lead[-1])
    obs = pd.read_parquet(cfgs[loc].data_dir / "clean" / "observations.parquet", columns=["station_latitude", "station_longitude"])
    lat, lon = float(obs.station_latitude.iloc[0]), float(obs.station_longitude.iloc[0])
    mp = [m for m in metas if json.loads(m.read_text())["params"]["models"] == model
          and abs(json.loads(m.read_text())["params"]["latitude"] - lat) < 1e-6]
    meta, h = load(mp[0])
    ws = pd.Timestamp(day).tz_localize("Australia/Perth") - pd.Timedelta(days=1) + pd.Timedelta(hours=9)
    hours = pd.date_range(ws.tz_convert("UTC") + pd.Timedelta(hours=1), ws.tz_convert("UTC") + pd.Timedelta(hours=24), freq="h")
    rule = forecasts.inferred_run_time(pd.Series(hours, index=hours), model, n)
    tested = matched = nz_t = nz_m = 0
    unarchived = set()
    for t in hours:
        sr = single_run(model, lat, lon, rule[t], "open_meteo/verify_live")
        v = h.loc[t, f"precipitation_previous_day{n}"]
        if sr is None or t not in sr.index or pd.isna(sr[t]):
            unarchived.add(f"{rule[t]:%d %HZ}")
            continue
        ok = bool(np.isclose(v, sr[t], atol=TOL))
        tested += 1
        matched += ok
        if v > 0 or sr[t] > 0:
            nz_t += 1
            nz_m += ok
    status = ("verified" if tested == 24 and matched == 24 else
              "UNVERIFIED (named run not archived)" if unarchived and matched == tested else "MISMATCH")
    b_rows.append({"location": loc, "gauge_day_label": day, "lead": lead, "input": inp,
                   "hours_matched": f"{matched}/{tested} of 24", "non_zero_hours_matched": f"{nz_m}/{nz_t}",
                   "unarchived_named_runs": ",".join(sorted(unarchived)) or "-", "status": status})
B = pd.DataFrame(b_rows)
out.append("\n## 3. Timing-pass live rows: each previous-runs input against the exact runs the offset rule names\n")
out.append("Single-run inputs (`sr_ecmwf_ifs`) are requested by explicit `run=` and need no inference. For each "
           "previous-runs input, all 24 hourly values of the window are compared with the named run:\n")
out.append(md(B))

# ---------------------------------------------------------------- Part C: historical sample
hist_rows = []
wide = pd.read_parquet(cfgs["perth"].joined_dir / "training_wide.parquet")
obs = pd.read_parquet(cfgs["perth"].data_dir / "clean" / "observations.parquet", columns=["station_latitude", "station_longitude"])
lat, lon = float(obs.station_latitude.iloc[0]), float(obs.station_longitude.iloc[0])
for model in ("jma_gsm", "ncep_gfs_global", "ecmwf_ifs025"):
    chunks = sorted((ROOT / "data/raw/open_meteo/previous_runs" / model).glob("*.meta.json"))
    sel = [c for c in chunks if json.loads(c.read_text())["params"]["start_date"] >= "2026-04-01"
           and abs(json.loads(c.read_text())["params"]["latitude"] - lat) < 1e-6
           and abs(json.loads(c.read_text())["params"]["longitude"] - lon) < 1e-6]
    hist = pd.concat([load(c)[1] for c in sel])
    # several cached requests overlap (probe window, runs with different end dates): they must agree exactly
    dup = hist[hist.index.duplicated(keep=False)]
    if len(dup):
        spread = dup.groupby(level=0).agg(lambda s: s.max() - s.min()).max().max()
        print(f"{model}: {dup.index.nunique()} hours present in >1 cached request; max disagreement {spread} mm")
        assert spread == 0, f"{model}: overlapping cached requests disagree"
    hist = hist[~hist.index.duplicated(keep="last")]
    for n in (2, 3):
        col = f"fc_pr_{model}_mm"
        w = wide[(wide.lead_group == f"day{n}") & (pd.to_datetime(wide.label_date_local) >= "2026-04-10")
                 & (wide[col] >= 2.0) & wide[f"eligible_pr_{model}"].astype(bool)]
        w = w.sort_values("label_date_local").iloc[:: max(1, len(w) // 3)].head(3)     # 3 rainy windows spread out
        for _, r in w.iterrows():
            hours = pd.date_range(r.window_start_utc + pd.Timedelta(hours=1), r.window_end_utc, freq="h")
            rule = forecasts.inferred_run_time(pd.Series(hours, index=hours), model, n)
            tested = matched = nz_t = nz_m = 0
            for t in hours:
                sr = single_run(model, lat, lon, rule[t], "open_meteo/verify_hist")
                if sr is None or t not in sr.index or pd.isna(sr[t]):
                    continue
                v = hist.loc[t, f"precipitation_previous_day{n}"]
                ok = bool(np.isclose(v, sr[t], atol=TOL))
                tested += 1
                matched += ok
                if v > 0 or sr[t] > 0:
                    nz_t += 1
                    nz_m += ok
            split = "held-out" if pd.Timestamp(r.label_date_local) >= pd.Timestamp("2026-03-13") else "training"
            hist_rows.append({"model": model, "lead": f"day{n}", "gauge_day_label": str(r.label_date_local)[:10],
                              "window_total_mm": round(r[col], 1), "split": split,
                              "hours_matched": f"{matched}/{tested}", "non_zero_hours_matched": f"{nz_m}/{nz_t}",
                              "status": "verified" if tested == 24 and matched == 24 else "NOT verified"})
C = pd.DataFrame(hist_rows)
out.append("\n## 4. Historical sample (Perth Metro coordinates, rainy windows, since the single-run archive began)\n")
out.append("Offset-rule runs named for historical valid hours were issued long before the data were retrieved "
           "(2026-09-30), so the future-run failure cannot occur there. Values compared hour by hour with the named runs:\n")
out.append(md(C))
(ROOT / "reports" / "live_provenance_audit.md").write_text("\n".join(out) + "\n", encoding="utf-8")
A.to_csv(ROOT / "reports" / "live_provenance_future_hours.csv", index=False)
B.to_csv(ROOT / "reports" / "live_provenance_timing_pass_rows.csv", index=False)
C.to_csv(ROOT / "reports" / "live_provenance_historical_sample.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 40)
print(A.to_string(index=False)); print(); print(E.head(12).to_string(index=False)); print()
print(B.to_string(index=False)); print(); print(C.to_string(index=False))
