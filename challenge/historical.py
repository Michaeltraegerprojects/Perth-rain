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
    # the reproduction must equal the stored, audited held-out score
    stored = pd.read_csv(ROOT / "reports" / slug / "calibration_metrics.csv")
    st = stored[(stored.lead == lead) & (stored.split == "holdout") & (stored.method == "hurdle")].iloc[0]
    q50 = C.mixture_quantile(p, lq, 0.5)
    if len(y) != int(st.n) or not np.isclose(np.abs(q50 - y).mean(), st.median_MAE_mm, rtol=0, atol=1e-9):
        raise champion.ChampionChanged(f"{slug} {lead}: reproduced held-out forecasts do not match the stored score "
                                       f"(n {len(y)} vs {int(st.n)}, MAE {np.abs(q50 - y).mean()} vs {st.median_MAE_mm})")
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
    out["champion_model"] = sel["feature_set"]
    out["champion_run_id"] = freeze["locations"][slug]["run_id"]
    out["champion_trained_to"] = str(train.label_date_local.max().date())
    return out.reset_index(drop=True), {"hold_start": str(pd.Timestamp(hold_start).date()), "n": len(y)}


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
    return tab


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
