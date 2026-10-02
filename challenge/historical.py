"""Historical contest table: the frozen champion versus raw benchmarks and challengers on the champion's original,
untouched held-out days.

* The champion's held-out forecasts are REPRODUCED with its frozen selection (feature set, C, seasonal term) and its
  own split rule; the reproduction must match the stored held-out score exactly or the run stops. Nothing is
  re-selected or tuned.
* Every forecast is at the gauge coordinates, for the exact 9am-9am window, at the same lead group and with the same
  as-of rule (latest contributing run + 6 h published before the window starts).
* Missing forecasts stay missing (NaN) and are counted as coverage gaps; they are never scored as zero.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from perthrain import calibrate as C
from challenge import champion, sources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "challenge"
RAW_BENCH = ["sr_ecmwf_ifs", "pr_ecmwf_ifs025", "pr_jma_gsm", "pr_ncep_gfs_global"]
TAU = np.round(np.arange(0.01, 0.991, 0.01), 2)      # quantile grid for the champion's CRPS


def unique_gauges():
    seen, out = {}, []
    for name, s in champion.locations():
        st = json.loads((ROOT / "reports" / s / "run_summary.json").read_text())["station"]
        if st["station_id"] in seen:
            seen[st["station_id"]]["serves"].append(name)
            continue
        g = {"slug": s, "station_id": st["station_id"], "station_name": st["station_name"].title(),
             "lat": float(st["latitude"]), "lon": float(st["longitude"]), "serves": [name]}
        seen[st["station_id"]] = g
        out.append(g)
    return out


def champion_holdout(slug: str, lead: str, freeze: dict):
    """Reproduce the champion's held-out forecasts for one gauge and lead (frozen config, original split)."""
    wide = pd.read_parquet(ROOT / "data" / slug / "joined" / "training_wide.parquet")
    wide = wide.drop(columns=[c for c in wide.columns if c.startswith("clim_")])
    cands = C._candidates(wide, lead)
    cands, common, _ = C._pick_common(cands)
    dates = np.array(sorted(common))
    hold_start = dates[int(len(dates) * (1 - C.HOLDOUT_FRAC))]
    sel = freeze["locations"][slug]["leads"][lead]["selected"]
    v = cands[sel["feature_set"]]
    gap = pd.Timedelta(days=C.GAP_DAYS)
    train = v["df"][v["df"].label_date_local < hold_start - gap]
    test = v["df"][(v["df"].label_date_local >= hold_start) & v["df"].label_date_local.isin(common)].copy()
    m = C.HurdleModel(C=sel["C"]).fit(C.make_X(train, v["cols"], v["combine"], sel["season"], None),
                                      train.observed_precip_mm.to_numpy(float))
    p, lq = m.parts(C.make_X(test, v["cols"], v["combine"], sel["season"], None))
    y = test.observed_precip_mm.to_numpy(float)
    q50 = C.mixture_quantile(p, lq, 0.5)
    _check_against_stored(slug, lead, test, p, lq, y, m)
    ytr = train.observed_precip_mm.to_numpy(float)
    ps, lqs, ss = C.climatology_seasonal(ytr, train.label_date_local, test.label_date_local)
    out = test[["label_date_local", "window_start_utc", "window_end_utc", "station_id", "observed_precip_mm"]
               + [f"fc_{c}_mm" for c in RAW_BENCH if f"fc_{c}_mm" in test.columns]
               + [f"eligible_{c}" for c in RAW_BENCH if f"eligible_{c}" in test.columns]].copy()
    out["lead"] = lead
    out["champion_median_mm"] = q50
    out["champion_expected_mm"] = C.mean_estimate(m, p, lq)
    out["champion_p10_mm"] = C.mixture_quantile(p, lq, 0.1)
    out["champion_p90_mm"] = C.mixture_quantile(p, lq, 0.9)
    for T in C.THRESHOLDS:
        out[f"champion_p_ge_{T:g}"] = C.exceed_prob(p, lq, T)
        out[f"clim_p_ge_{T:g}"] = C.exceed_prob(ps, lqs, T)
    out["champion_crps"] = _crps_from_quantiles(np.column_stack([C.mixture_quantile(p, lq, t) for t in TAU]), y)
    out["clim_crps"] = _crps_from_quantiles(np.column_stack([C.mixture_quantile(ps, lqs, t) for t in TAU]), y)
    # information age at the window start = hours from the latest contributing model run to the cutoff
    for c in RAW_BENCH:
        if f"lead_hours_{c}" in test.columns:
            out[f"age_{c}_h"] = test[f"lead_hours_{c}"].to_numpy(float) - 24.0
    member_ages = [f"age_{c}_h" for c in C.FEATURE_SETS[sel["feature_set"]]["cols"] if f"age_{c}_h" in out]
    out["age_champion_h"] = out[member_ages].min(axis=1) if member_ages else np.nan
    out["champion_model"] = sel["feature_set"]
    out["champion_run_id"] = freeze["locations"][slug]["run_id"]
    out["champion_trained_to"] = str(train.label_date_local.max().date())
    out = out.reset_index(drop=True)
    _check_against_reference(slug, lead, out)
    return out, {"hold_start": str(pd.Timestamp(hold_start).date()), "n": len(y)}


# Every held-out aggregate the audited calibration stored for the selected model; all must be reproduced.
STORED_AGGREGATES = ("pinball_mean", "median_MAE_mm", "median_bias_mm", "median_RMSE_mm", "mean_est_MAE_mm",
                     "mean_est_bias_mm", "mean_est_RMSE_mm", "brier_0.2", "brier_1", "brier_5", "brier_10",
                     "exceed_q50", "exceed_q75", "exceed_q90")
AGG_TOL = 1e-9
ROW_TOL = 1e-12
REFERENCE_COLUMNS = ["champion_median_mm", "champion_expected_mm", "champion_p10_mm", "champion_p90_mm",
                     "champion_p_ge_0.2", "champion_p_ge_1", "champion_p_ge_5", "champion_p_ge_10"]


def _check_against_stored(slug, lead, test, p, lq, y, model):
    """The reproduction must equal the audited held-out evaluation: same rows (station, window and observed
    target, by fingerprint), same count and every stored aggregate."""
    stored = pd.read_csv(ROOT / "reports" / slug / "calibration_metrics.csv")
    st = stored[(stored.lead == lead) & (stored.split == "holdout") & (stored.method == "hurdle")].iloc[0]
    problems = []
    n_rows, fp = C.frame_fingerprint(test)
    if n_rows != int(st.n):
        problems.append(f"n {n_rows} vs stored {int(st.n)}")
    if "key_fingerprint" not in st or fp != st.key_fingerprint:
        problems.append(f"evaluation key fingerprint {fp[:12]}... vs stored {str(st.get('key_fingerprint'))[:12]}...")
    got = C.prob_metrics(p, lq, y, model.tail_scale_)
    for k in STORED_AGGREGATES:
        if k in st and pd.notna(st[k]) and not np.isclose(got.get(k, np.nan), st[k], rtol=0, atol=AGG_TOL):
            problems.append(f"{k} {got.get(k)} vs stored {st[k]}")
    if problems:
        raise champion.ChampionChanged(f"{slug} {lead}: reproduced held-out forecasts do not match the audited "
                                       f"evaluation: {problems}")


def reference_path(slug: str, lead: str) -> Path:
    return ROOT / "challenge" / "reference" / f"holdout_{slug}_{lead}.parquet"


def _row_keys(df: pd.DataFrame) -> pd.Series:
    """Per-row key as a hash of station, window and observed target (the reference stores no observation values)."""
    import hashlib
    raw = (df.station_id.astype(str) + "|" + pd.to_datetime(df.window_start_utc, utc=True).astype(str) + "|" +
           pd.to_datetime(df.window_end_utc, utc=True).astype(str) + "|" + df.observed_precip_mm.round(6).astype(str))
    return raw.map(lambda x: hashlib.sha256(x.encode()).hexdigest())


def _check_against_reference(slug, lead, out):
    """Per-row check against the saved reference (written on the first verified run, then frozen by `refreeze`).
    Aggregates can agree while individual forecasts differ; this catches that."""
    ref_p = reference_path(slug, lead)
    cur = pd.DataFrame({"row_key": _row_keys(out), **{c: out[c].to_numpy(float) for c in REFERENCE_COLUMNS}})
    if not ref_p.exists():
        ref_p.parent.mkdir(parents=True, exist_ok=True)
        cur.to_parquet(ref_p, index=False)
        return
    ref = pd.read_parquet(ref_p)
    if set(ref.row_key) != set(cur.row_key) or len(ref) != len(cur):
        raise champion.ChampionChanged(f"{slug} {lead}: per-row reference has different evaluation rows")
    m = cur.merge(ref, on="row_key", suffixes=("", "_ref"))
    bad = [c for c in REFERENCE_COLUMNS if not np.allclose(m[c], m[f"{c}_ref"], rtol=0, atol=ROW_TOL)]
    if bad:
        n_bad = int(sum((~np.isclose(m[c], m[f"{c}_ref"], rtol=0, atol=ROW_TOL)).sum() for c in bad))
        raise champion.ChampionChanged(f"{slug} {lead}: per-row forecasts differ from the reference in {bad} "
                                       f"({n_bad} values)")


def _crps_from_quantiles(Q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS = 2 * integral of the pinball loss over quantile levels (midpoint rule on TAU)."""
    d = y[:, None] - Q
    pin = np.maximum(TAU * d, (TAU - 1) * d)
    return 2 * pin.mean(axis=1)


def add_challengers(tab: pd.DataFrame, g: dict, lead: str) -> pd.DataFrame:
    n = int(lead[-1])
    windows = tab[["window_start_utc", "window_end_utc"]].drop_duplicates().reset_index(drop=True)
    windows["window_start_utc"] = pd.to_datetime(windows.window_start_utc, utc=True)
    windows["window_end_utc"] = pd.to_datetime(windows.window_end_utc, utc=True)
    for model, meta in sources.CHALLENGERS.items():
        if n > 1:
            tab = _attach(tab, _sr_totals(model, g, windows, n), f"{model}_12z")
        if n == 1:
            # same semantics as the champion's day-1 input: the 12 UTC single run published (6 h) before the window
            runs = sources.sr_issue_time(windows.window_start_utc, 1)
            rows = []
            for run, grp in windows.assign(run=runs).groupby("run"):
                sr = sources.fetch_single_run(model, g["lat"], g["lon"], run, 3)
                if sr is None:
                    continue
                wt = sources.window_totals_single_run(sr, grp[["window_start_utc", "window_end_utc"]].reset_index(drop=True), 1)
                rows.append(wt)
            wt = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["window_start_utc"])
        else:
            pr = sources.fetch_previous_runs(model, g["lat"], g["lon"], windows.window_start_utc.min().tz_localize(None).normalize(),
                                             windows.window_end_utc.max().tz_localize(None).normalize(), [n])
            wt = sources.window_totals_previous_runs(pr, windows) if len(pr) else pd.DataFrame(columns=["window_start_utc"])
        keep = {"total_mm": f"fc_{model}_mm", "published_before_cutoff": f"eligible_{model}",
                "latest_run_utc": f"run_{model}_utc", "grid_latitude": f"grid_lat_{model}",
                "grid_longitude": f"grid_lon_{model}", "retrieved_at_utc": f"retrieved_{model}"}
        if len(wt):
            wt = wt[["window_start_utc", *keep]].rename(columns=keep)
            wt["window_start_utc"] = pd.to_datetime(wt.window_start_utc, utc=True)
            tab = tab.merge(wt, on="window_start_utc", how="left")
        else:
            for c in keep.values():
                tab[c] = np.nan
        # only forecasts that were published before the window started count; others are missing, not zero
        tab.loc[~tab[f"eligible_{model}"].fillna(False).astype(bool), f"fc_{model}_mm"] = np.nan
        tab = _add_age(tab, model)
    return tab


def _sr_totals(model, g, windows, n):
    """Window totals from the 12 UTC single run the champion's rule names for lead n (same run rule as its
    sr_ecmwf_ifs input)."""
    runs = sources.sr_issue_time(windows.window_start_utc, n)
    rows = []
    for run, grp in windows.assign(run=runs).groupby("run"):
        sr = sources.fetch_single_run(model, g["lat"], g["lon"], run, 5)
        if sr is not None:
            rows.append(sources.window_totals_single_run(sr, grp[["window_start_utc", "window_end_utc"]].reset_index(drop=True), n))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["window_start_utc"])


def _add_age(tab, name):
    run = pd.to_datetime(tab.get(f"run_{name}_utc"), utc=True, errors="coerce")
    tab[f"age_{name}_h"] = (pd.to_datetime(tab.window_start_utc, utc=True) - run).dt.total_seconds() / 3600
    return tab


def _attach(tab, wt, name):
    keep = {"total_mm": f"fc_{name}_mm", "published_before_cutoff": f"eligible_{name}", "latest_run_utc": f"run_{name}_utc",
            "grid_latitude": f"grid_lat_{name}", "grid_longitude": f"grid_lon_{name}", "retrieved_at_utc": f"retrieved_{name}"}
    if len(wt):
        wt = wt[["window_start_utc", *keep]].rename(columns=keep)
        wt["window_start_utc"] = pd.to_datetime(wt.window_start_utc, utc=True)
        tab = tab.merge(wt, on="window_start_utc", how="left")
    else:
        for c in keep.values():
            tab[c] = np.nan
    tab.loc[~tab[f"eligible_{name}"].fillna(False).astype(bool), f"fc_{name}_mm"] = np.nan
    return _add_age(tab, name)


def build() -> pd.DataFrame:
    freeze = champion.verify()
    frames, info = [], []
    for g in unique_gauges():
        for lead in ("day1", "day2", "day3"):
            tab, meta = champion_holdout(g["slug"], lead, freeze)
            tab["window_start_utc"] = pd.to_datetime(tab.window_start_utc, utc=True)
            tab = add_challengers(tab, g, lead)
            for c in RAW_BENCH:
                if f"eligible_{c}" in tab:
                    tab.loc[~tab[f"eligible_{c}"].fillna(False).astype(bool), f"fc_{c}_mm"] = np.nan
            tab["gauge_id"], tab["gauge_name"], tab["gauge_slug"] = g["station_id"], g["station_name"], g["slug"]
            tab["serves"] = ", ".join(g["serves"])
            frames.append(tab)
            info.append({"gauge": g["station_name"], "lead": lead, **meta})
            print(f"{g['station_name']} {lead}: champion n={meta['n']} reproduced exactly; challengers attached", flush=True)
    T = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_parquet(OUT / "historical_contest.parquet", index=False)
    (OUT / "historical_contest_info.json").write_text(json.dumps(info, indent=1))
    return T


if __name__ == "__main__":
    build()
