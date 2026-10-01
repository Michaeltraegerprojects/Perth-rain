"""Routing: every lead x every combination of available models, through the REAL prediction entry point
(predict_windows / compute_predictions). Expected routes below are HARD-CODED literals, not derived from route(),
so reversing the routing order or changing the rule fails these tests."""
import copy
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from perthrain import observations
from perthrain import predict as P
from perthrain.config import slug

ROOT = Path(__file__).resolve().parents[1]
TZ = "Australia/Perth"
S, E, J, G = "sr_ecmwf_ifs", "pr_ecmwf_ifs025", "pr_jma_gsm", "pr_ncep_gfs_global"

# routing order of the day-2/day-3 fixture: all_composite(S,J,G,E) > ecmwf_composite(S,E) > ecmwf_ifs_sr(S)
#                                            > jma_gsm_pr(J) > ecmwf_ifs025_pr(E)
EXPECTED_DAY23 = {
    frozenset(): (None, None),
    frozenset({S}): ("ecmwf_ifs_sr", "fallback#2"),
    frozenset({E}): ("ecmwf_ifs025_pr", "fallback#4"),
    frozenset({J}): ("jma_gsm_pr", "fallback#3"),
    frozenset({G}): (None, None),
    frozenset({S, E}): ("ecmwf_composite", "fallback#1"),
    frozenset({S, J}): ("ecmwf_ifs_sr", "fallback#2"),
    frozenset({S, G}): ("ecmwf_ifs_sr", "fallback#2"),
    frozenset({E, J}): ("jma_gsm_pr", "fallback#3"),
    frozenset({E, G}): ("ecmwf_ifs025_pr", "fallback#4"),
    frozenset({J, G}): ("jma_gsm_pr", "fallback#3"),
    frozenset({S, E, J}): ("ecmwf_composite", "fallback#1"),
    frozenset({S, E, G}): ("ecmwf_composite", "fallback#1"),
    frozenset({S, J, G}): ("ecmwf_ifs_sr", "fallback#2"),
    frozenset({E, J, G}): ("jma_gsm_pr", "fallback#3"),
    frozenset({S, E, J, G}): ("all_composite", "primary"),
}
IDENTITY_COLS = {"label_date_local", "window_start_local", "window_end_local", "window_start_utc", "window_end_utc",
                 "lead_group", "status", "feature_set_used", "route", "run_id", "enso_status", "artifact_sha256",
                 "models_available", "inputs_blocked_by_timing"}


def _windows(first="2026-10-02", n=1):
    rows = []
    for d in pd.date_range(first, periods=n, freq="D").date:
        s, e = observations.obs_window(d, TZ)
        rows.append({"label_date_local": d, "window_start_utc": s, "window_end_utc": e})
    return pd.DataFrame(rows)


def _features(windows, lead_day, available):
    rows = []
    for _, w in windows.iterrows():
        for col in P.ALL_MODEL_COLS:
            rows.append({"label_date_local": w.label_date_local, "lead_day": lead_day, "col": col,
                         "usable": col in available, "timing_ok": True, "forecast_complete": True,
                         "forecast_precip_mm": 2.5 if col in available else np.nan,
                         "forecast_issue_time_utc": pd.Timestamp("2026-09-29T12:00Z") if col.startswith("sr_") else pd.NaT,
                         "issue_time_inferred_max_utc": pd.Timestamp("2026-09-29T06:00Z"),
                         "retrieved_at_utc": "2026-09-30T14:00:00Z"})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("lead", ["day2", "day3"])
@pytest.mark.parametrize("avail", list(EXPECTED_DAY23), ids=lambda a: ",".join(sorted(a)) or "none")
def test_day23_route_matches_the_hard_coded_expectation(clean_bundles, lead, avail):
    w = _windows()
    n = int(lead[-1])
    pred = P.predict_windows(clean_bundles, _features(w, n, set(avail)), w, TZ)
    row = pred[pred.lead_group == lead].iloc[0]
    table = P.routing_table(clean_bundles)
    trow = table[(table.lead == lead) & (table.models_available == (",".join(sorted(avail)) or "(none)"))].iloc[0]
    name, route = EXPECTED_DAY23[avail]
    if name is None:
        assert row.status == "no_complete_forecast" and row.route == "exhausted"
        assert trow.feature_set == "NOT PREDICTED"
        # NO number of any kind may be served: every non-identity column is empty
        numeric = [c for c in pred.columns if c not in IDENTITY_COLS]
        assert row[numeric].isna().all(), row[numeric].dropna().to_dict()
    else:
        assert (row.status, row.feature_set_used, row.route) == ("ok", name, route)
        assert (trow.feature_set, trow.route) == (name, route)
        assert 0.0 <= row.p_rain_ge_0_2mm <= 1.0 and row.q10_mm <= row.q50_mm <= row.q90_mm
    assert not [c for c in pred.columns if c.startswith(("clim_", "climate", "nino", "raw_"))]


@pytest.mark.parametrize("avail", [frozenset(), frozenset({J}), frozenset({G}), frozenset({S}), frozenset({S, J, G, E})])
def test_day1_only_the_ecmwf_single_run_can_serve(clean_bundles, avail):
    w = _windows()
    pred = P.predict_windows(clean_bundles, _features(w, 1, set(avail)), w, TZ)
    row = pred[pred.lead_group == "day1"].iloc[0]
    if S in avail:
        assert (row.status, row.feature_set_used, row.route) == ("ok", "ecmwf_ifs_sr", "primary")
    else:
        assert row.status == "no_complete_forecast" and row.route == "exhausted"
        assert row[[c for c in pred.columns if c not in IDENTITY_COLS]].isna().all()


def test_reversing_the_routing_order_would_be_caught(clean_bundles):
    """Mutation check: with the order reversed the hard-coded table no longer holds (so it has teeth)."""
    b = copy.deepcopy(clean_bundles)
    b["day2"]["routing_order"] = list(reversed(b["day2"]["routing_order"]))
    b["day2"]["feature_set"] = b["day2"]["routing_order"][0]
    w = _windows()
    pred = P.predict_windows(b, _features(w, 2, {S, E, J, G}), w, TZ)
    assert pred[pred.lead_group == "day2"].iloc[0].feature_set_used != "all_composite"   # the hard-coded table would fail


def test_day3_jma_only_routes_to_the_clean_jma_fallback(clean_bundles):
    w = _windows()
    pred = P.predict_windows(clean_bundles, _features(w, 3, {J}), w, TZ)
    row = pred[pred.lead_group == "day3"].iloc[0]
    assert row.status == "ok" and row.feature_set_used == "jma_gsm_pr" and row.route == "fallback#3"
    assert row.feature_schema == "log1p(fc_pr_jma_gsm_mm) | season_sin(day_of_year) | season_cos(day_of_year)"


def test_missing_artifact_for_a_lead_is_reported_not_skipped(clean_bundles):
    b = {k: v for k, v in clean_bundles.items() if k != "day2"}
    w = _windows()
    pred = P.predict_windows(b, _features(w, 2, set(P.ALL_MODEL_COLS)), w, TZ)
    row = pred[pred.lead_group == "day2"].iloc[0]
    assert row.status == "no_model_artifact" and pd.isna(row.feature_set_used)


def test_no_enso_mode_refuses_a_climate_fit_even_if_it_slipped_past_loading(clean_bundles):
    from conftest import make_fit
    from perthrain.enso_guard import EnsoGuardError
    b = copy.deepcopy(clean_bundles)
    b["day3"]["fits"]["jma_gsm_pr"] = make_fit(["pr_jma_gsm"], climate="nino34_iod", seed=3)
    w = _windows()
    with pytest.raises(EnsoGuardError):
        P.predict_windows(b, _features(w, 3, {J}), w, TZ)


# ------------------------------------------------------------------ ACTIVE artifacts (fail loudly if absent)
def _locations():
    cfg = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
    return [(e["name"], slug(e["name"])) for e in cfg["locations"]]


LOCS = _locations()


def _cfg(loc_slug):
    from perthrain.config import load_configs
    c = [c for c in load_configs(str(ROOT / "config.toml")) if c.data_dir.name == loc_slug][0]
    c.data_dir, c.reports_dir, c.raw_root = ROOT / c.data_dir, ROOT / c.reports_dir, ROOT / c.raw_root
    return c


@pytest.mark.parametrize("name,loc", LOCS)
def test_active_artifacts_exist_for_every_configured_location_and_lead(name, loc):
    for lead in ("day1", "day2", "day3"):
        assert (ROOT / "data" / loc / "models" / f"hurdle_{lead}.joblib").exists(), f"{loc} {lead} missing"


@pytest.mark.parametrize("name,loc", LOCS)
def test_active_artifacts_route_as_the_written_routing_table_says(name, loc):
    cfg = _cfg(loc)
    bundles = P.load_bundles(cfg)
    assert set(bundles) == {"day1", "day2", "day3"}
    written = pd.read_csv(cfg.reports_dir / "routing_table.csv").fillna("")
    fresh = P.routing_table(bundles).fillna("")
    cols = ["lead", "models_available", "feature_set", "route", "feature_schema", "climate", "run_id"]
    pd.testing.assert_frame_equal(written[cols].reset_index(drop=True), fresh[cols].reset_index(drop=True))
    w = _windows()
    for lead, n in P.LEADS.items():
        for avail in P.availability_subsets():
            row = P.predict_windows(bundles, _features(w, n, set(avail)), w, TZ)
            row = row[row.lead_group == lead].iloc[0]
            t = fresh[(fresh.lead == lead) & (fresh.models_available == (",".join(sorted(avail)) or "(none)"))].iloc[0]
            if t.feature_set == "NOT PREDICTED":
                assert row.status == "no_complete_forecast" and row.route == "exhausted"
                assert row[[c for c in row.index if c not in IDENTITY_COLS]].isna().all()
            else:
                assert (row.feature_set_used, row.route) == (t.feature_set, t.route)
    # day-3 with only JMA available: must be the JMA fallback or explicitly exhausted - never silently skipped
    r = P.predict_windows(bundles, _features(w, 3, {J}), w, TZ)
    r = r[r.lead_group == "day3"].iloc[0]
    if "jma_gsm_pr" in bundles["day3"]["fits"]:
        assert r.feature_set_used == "jma_gsm_pr" and "clim" not in r.feature_schema.lower()
    else:
        assert r.route == "exhausted"


@pytest.mark.parametrize("name,loc", LOCS)
def test_served_predictions_are_reproduced_exactly_by_the_real_entry_point(name, loc):
    """compute_predictions (real entry point, OFFLINE: cache only) on the cached live inputs of the newest served
    run reproduces latest.csv row for row, exactly (rtol=0)."""
    cfg = _cfg(loc)
    recs = sorted((cfg.data_dir / "predictions").glob("forecast_*.csv"))
    assert recs, f"no served predictions for {loc}"
    served = pd.read_csv(cfg.data_dir / "predictions" / "latest.csv")
    now = pd.Timestamp(pd.to_datetime(recs[-1].stem.split("_")[1], format="%Y%m%dT%H%MZ"), tz="UTC")
    r = P.compute_predictions(cfg, now=now, refresh_live=False)
    assert not [p for p in r["problems"] if "not in the cache" in p], r["problems"]
    got = r["pred"]
    got["label_date_local"] = got.label_date_local.astype(str)
    m = served.merge(got, on=["label_date_local", "lead_group"], suffixes=("_served", "_replay"))
    assert len(m) == len(served) == len(got)
    assert (m.status_served == m.status_replay).all()
    assert (m.feature_set_used_served.fillna("") == m.feature_set_used_replay.fillna("")).all()
    assert (m.route_served.fillna("") == m.route_replay.fillna("")).all()
    ok = m[m.status_served == "ok"]
    for c in ("p_rain_ge_0_2mm", "p_ge_1mm", "p_ge_5mm", "p_ge_10mm", "q50_mm", "q90_mm", "mean_estimate_mm"):
        assert np.allclose(ok[f"{c}_served"], ok[f"{c}_replay"], atol=1e-12, rtol=0), c
    table = P.routing_table(r["bundles"]).set_index(["lead", "models_available"])
    for _, x in ok.iterrows():
        exp = table.loc[(x.lead_group, x.models_available_served)]
        assert x.feature_set_used_served == exp.feature_set and x.route_served == exp.route
