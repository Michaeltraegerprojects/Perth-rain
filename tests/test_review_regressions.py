"""One regression test per CONFIRMED reviewer finding (calibration, loader/guard, prediction). Names say which."""
import copy
import os
import shutil
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import make_bundle, make_fit
from perthrain import calibrate as C
from perthrain import enso_guard as G
from perthrain import observations
from perthrain import predict as P
from perthrain.config import Config

TZ = "Australia/Perth"


# ----------------------------------------------------------------------- synthetic training table
def synth_wide(n=900, seed=1, ecmwf_from=0, jma_range=None, dry_until=0, leads=("day1", "day2", "day3")):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2023-01-01")
    rows = []
    days = pd.date_range(start, periods=n, freq="D")
    for lead in leads:
        base = rng.gamma(0.6, 3.0, n)
        y = np.where(rng.random(n) < 1 - np.exp(-base), rng.gamma(2.0, np.maximum(base, 0.3)), 0.0)
        y[:dry_until] = 0.0
        d = pd.DataFrame({"label_date_local": days, "lead_group": lead, "observed_precip_mm": y,
                          "station_id": "009225",
                          "window_start_utc": days.tz_localize("UTC") - pd.Timedelta(hours=23),
                          "window_end_utc": days.tz_localize("UTC") + pd.Timedelta(hours=1)})
        for k, c in enumerate(["sr_ecmwf_ifs", "pr_jma_gsm", "pr_ncep_gfs_global", "pr_ecmwf_ifs025"]):
            d[f"fc_{c}_mm"] = np.clip(base * rng.lognormal(0, 0.4, n), 0, None).round(2)
            ok = np.ones(n, bool)
            if c == "sr_ecmwf_ifs":
                ok[:ecmwf_from] = False
            if c == "pr_jma_gsm" and jma_range is not None:
                ok[:] = False
                ok[jma_range[0]:jma_range[1]] = True
            d[f"eligible_{c}"] = ok
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


# ============================================================================ calibration
def test_unequal_folds_are_never_scored_differently():
    """Reviewer: a fold that one candidate cannot score made cv_pinball average different days for different
    candidates. Now the fold is dropped for ALL configurations."""
    # ecmwf starts at the common start with < MIN_TRAIN training rows in fold 1; JMA has a long history
    wide = synth_wide(n=900, ecmwf_from=500, leads=("day2",))
    res, bundle = C.evaluate_lead(wide, "day2", only=("ecmwf_ifs_sr", "jma_gsm_pr"), use_climate=False)
    assert bundle, res.notes
    assert res.cv_folds == ["cv2", "cv3"], res.cv_folds
    assert {r["cv_folds"] for r in res.ranking} == {"cv2,cv3"}
    assert len({r["cv_n"] for r in res.ranking}) == 1, "every configuration must be scored on the same days"


def test_unusable_fold_from_too_few_wet_days_does_not_abort_calibration():
    """Reviewer: HurdleModel.fit ValueError (<40 wet days in fold-1 training) aborted calibrate."""
    wide = synth_wide(n=600, dry_until=250, leads=("day2",))
    res, bundle = C.evaluate_lead(wide, "day2", only=("ecmwf_ifs_sr", "jma_gsm_pr"), use_climate=False)
    assert bundle, res.notes
    assert "cv1" not in res.cv_folds and res.cv_folds


def test_disjoint_candidate_periods_exclude_a_candidate_instead_of_crashing():
    """Reviewer: empty/tiny common date set raised IndexError or produced no model."""
    wide = synth_wide(n=900, leads=("day2",), ecmwf_from=0)
    wide.loc[wide.label_date_local >= wide.label_date_local.min() + pd.Timedelta(days=300), "eligible_sr_ecmwf_ifs"] = False
    wide["eligible_pr_jma_gsm"] = wide.label_date_local >= wide.label_date_local.min() + pd.Timedelta(days=400)
    res, bundle = C.evaluate_lead(wide, "day2", only=("ecmwf_ifs_sr", "jma_gsm_pr"), use_climate=False)
    assert res.excluded_candidates == ["ecmwf_ifs_sr"], (res.excluded_candidates, res.notes)
    assert bundle and bundle["routing_order"] == ["jma_gsm_pr"]


def test_holdout_methods_share_identical_keys_and_targets():
    wide = synth_wide(n=900, leads=("day2",))
    res, _ = C.evaluate_lead(wide, "day2", use_climate=False)
    hold = [r for r in res.holdout if r["method"] in ("hurdle", "climatology", "climatology_flat") or
            str(r["method"]).startswith("raw:")]
    assert len({r["key_fingerprint"] for r in hold}) == 1
    assert {r["n"] for r in hold} == {hold[0]["n"]} and all(r["keys_identical_across_methods"] for r in hold)


def test_frame_fingerprint_detects_a_changed_window_or_target():
    d = synth_wide(n=50, leads=("day2",))
    a = C.frame_fingerprint(d)
    b = d.copy()
    b.loc[3, "observed_precip_mm"] += 0.2
    c = d.copy()
    c.loc[3, "window_start_utc"] += pd.Timedelta(hours=1)
    assert a != C.frame_fingerprint(b) and a != C.frame_fingerprint(c)
    assert a == C.frame_fingerprint(d.sample(frac=1, random_state=0))       # row order does not matter


def test_seasonal_climatology_differs_from_flat_and_uses_training_dates_only():
    dates = pd.date_range("2024-01-01", periods=730, freq="D")
    y = np.where(dates.month.isin([6, 7, 8]), 5.0, 0.0)                  # rain only in winter
    ps, lqs, ss = C.climatology_seasonal(y, dates, pd.DatetimeIndex(["2025-01-15", "2025-07-15"]))
    pf, lqf, sf = C.climatology_flat(y, 2)
    assert ps[0] < 0.05 and ps[1] > 0.9                                   # January dry, July wet
    assert pf[0] == pf[1] and 0.2 < pf[0] < 0.3                          # flat is one number
    # training-only: changing a value AFTER the training dates cannot change the fit
    ps2, _, _ = C.climatology_seasonal(y[:-10], dates[:-10], pd.DatetimeIndex(["2025-01-15"]))
    assert ps2[0] < 0.05


def test_paired_bootstrap_interval_and_significance():
    rng = np.random.default_rng(0)
    d = rng.normal(-0.3, 1.0, 400)                                        # clear negative mean
    r = C.paired_bootstrap(d)
    assert r["ci95_hi"] < 0 and r["significant_at_95"] and r["mean_diff"] < 0
    z = C.paired_bootstrap(rng.normal(0.0, 1.0, 100))
    assert z["ci95_lo"] < 0 < z["ci95_hi"] and not z["significant_at_95"]
    assert C.paired_bootstrap(d) == C.paired_bootstrap(d)                 # seeded: reproducible


def test_median_and_expected_total_are_scored_separately():
    p = np.array([0.3, 0.9, 0.6]); lq = np.tile(np.log(C.GRID * 10 + 0.2), (3, 1)); y = np.array([0.0, 1.0, 7.0])
    m = C.prob_metrics(p, lq, y, 1.0)
    for k in ("median_MAE_mm", "median_bias_mm", "median_RMSE_mm", "mean_est_MAE_mm", "mean_est_bias_mm",
              "mean_est_RMSE_mm"):
        assert k in m
    assert not np.isclose(m["median_MAE_mm"], m["mean_est_MAE_mm"])
    # event definition is inclusive: an observation of exactly 0.2 mm is an event at the 0.2 mm threshold
    m2 = C.prob_metrics(np.array([1.0]), lq[:1], np.array([0.2]), 1.0)
    assert m2["freq_0.2"] == 1.0


def _project(tmp_path, wide):
    cfg = Config(data_dir=tmp_path / "data" / "loc", reports_dir=tmp_path / "reports" / "loc",
                 raw_root=tmp_path / "data", location_name="Loc")
    (cfg.data_dir / "joined").mkdir(parents=True)
    wide.to_parquet(cfg.joined_dir / "training_wide.parquet")
    (cfg.data_dir / "clean").mkdir()
    return cfg


def test_failed_calibration_leaves_existing_artifacts_untouched(tmp_path):
    """Reviewer: old artifacts were moved aside BEFORE fitting; a later failure left models/ half-written."""
    wide = synth_wide(n=700, leads=("day2", "day3"))                        # no day1 rows -> day1 cannot be built
    cfg = _project(tmp_path, wide)
    mdir = cfg.data_dir / "models"
    mdir.mkdir()
    (mdir / "hurdle_day1.joblib").write_bytes(b"previous artifact")
    with pytest.raises(C.CalibrationError):
        C.run_calibration(cfg)
    assert (mdir / "hurdle_day1.joblib").read_bytes() == b"previous artifact"
    assert sorted(p.name for p in mdir.iterdir()) == ["hurdle_day1.joblib"]
    assert not list(cfg.data_dir.glob("staging_*")), "staging directory must be cleaned up"
    assert not (tmp_path / "archive").exists(), "nothing may be moved to the archive on failure"


def test_successful_calibration_swaps_moves_old_aside_and_is_loadable(tmp_path):
    wide = synth_wide(n=700)
    cfg = _project(tmp_path, wide)
    mdir = cfg.data_dir / "models"
    mdir.mkdir()
    (mdir / "hurdle_day1.joblib").write_bytes(b"previous artifact")
    sel = C.run_calibration(cfg)
    assert sorted(p.name for p in mdir.iterdir()) == ["hurdle_day1.joblib", "hurdle_day2.joblib", "hurdle_day3.joblib",
                                                     "manifest.json", "selection.json"]
    old = list((tmp_path / "archive" / "superseded_models").rglob("hurdle_day1.joblib"))
    assert len(old) == 1 and old[0].read_bytes() == b"previous artifact"
    bundles = P.load_bundles(cfg)
    assert set(bundles) == {"day1", "day2", "day3"}
    b = bundles["day2"]
    assert b["enso_status"] == "no_enso" and b["meta"]["location"] == "Loc" and b["meta"]["source_data"]
    assert (cfg.reports_dir / "routing_table.csv").exists() and (cfg.reports_dir / "paired_differences.csv").exists()
    assert sel["_notes"]["day2"] == [] or isinstance(sel["_notes"]["day2"], list)


def test_default_calibration_ignores_climate_columns_in_the_wide_table(tmp_path):
    """Reviewer: clim_* columns present in training_wide switched the default run into a climate search."""
    wide = synth_wide(n=700)
    wide["clim_nino34"] = 1.0
    wide["clim_iod"] = 0.1
    cfg = _project(tmp_path, wide)
    C.run_calibration(cfg)
    b = P.load_bundles(cfg)["day2"]
    assert all(f["climate"] in (None, "none") for f in b["fits"].values())
    assert all("clim" not in s for f in b["fits"].values() for s in f["feature_schema"])


# ================================================================================ loader / guard
def _cfg_here(tmp_path, name="Perth", data="perth"):
    return Config(data_dir=tmp_path / "data" / data, reports_dir=tmp_path / "reports" / data,
                  raw_root=tmp_path / "data", location_name=name)


def _put(cfg, bundle, lead="day1"):
    d = cfg.data_dir / "models"
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, d / f"hurdle_{lead}.joblib")
    return d / f"hurdle_{lead}.joblib"


def _with_loc(bundle, name, data):
    b = copy.deepcopy(bundle)
    b["meta"]["location"], b["meta"]["gauge_data_dir"] = name, data
    return b


def test_artifact_for_another_location_is_refused(tmp_path, clean_bundles):
    """Reviewer: Clarkson-labelled artifacts placed in data/perth/models loaded for Perth."""
    from conftest import install_models
    cfg = _cfg_here(tmp_path)
    bundles = {k: _with_loc(b, "Perth", "perth") for k, b in clean_bundles.items()}
    bundles["day1"] = _with_loc(clean_bundles["day1"], "Clarkson", "clarkson")
    install_models(cfg.data_dir / "models", bundles)
    with pytest.raises(G.EnsoGuardError, match="not 'Perth'"):
        P.load_bundles(cfg)


def test_project_under_a_folder_named_archive_loads_through_load_bundles(tmp_path, clean_bundles):
    """Reviewer: the 'parent named archive' allowance was only tested on assert_servable_path."""
    from conftest import install_models
    root = tmp_path / "archive" / "proj"
    cfg = _cfg_here(root)
    install_models(cfg.data_dir / "models", {k: _with_loc(b, "Perth", "perth") for k, b in clean_bundles.items()})
    assert set(P.load_bundles(cfg)) == {"day1", "day2", "day3"}


@pytest.mark.parametrize("field", list(G.REQUIRED_META))
def test_every_required_meta_field_is_enforced(tmp_path, clean_bundles, field):
    b = _with_loc(clean_bundles["day1"], "Perth", "perth")
    b["meta"].pop(field)
    v = G.validate_artifact(b, "day1")
    assert any(f"meta.{field}" in x or (field == "lead" and "lead mismatch" in x) or
               (field == "enso_status" and "enso_status" in x) for x in v), v


@pytest.mark.parametrize("field,bad", [("run_id", 0), ("code_sha256", " "), ("source_data", []), ("versions", []),
                                       ("created_utc", None), ("run_id", "")])
def test_meta_field_types_are_checked(clean_bundles, field, bad):
    b = copy.deepcopy(clean_bundles["day1"])
    b["meta"][field] = bad
    assert any(f"meta.{field}" in x for x in G.validate_artifact(b, "day1"))


def test_fit_without_a_model_is_rejected_by_the_loader_not_at_prediction(clean_bundles):
    b = copy.deepcopy(clean_bundles["day2"])
    del b["fits"]["jma_gsm_pr"]["model"]
    assert any("missing or not a HurdleModel" in x for x in G.validate_artifact(b, "day2"))
    b2 = copy.deepcopy(clean_bundles["day2"])
    b2["fits"]["jma_gsm_pr"]["model"] = object()
    assert any("not a HurdleModel" in x for x in G.validate_artifact(b2, "day2"))
    b3 = copy.deepcopy(clean_bundles["day2"])
    b3["fits"]["jma_gsm_pr"]["model"].q_ = []
    assert any("not fitted" in x for x in G.validate_artifact(b3, "day2"))


def test_directory_named_like_an_artifact_is_refused(tmp_path):
    root = tmp_path / "proj"
    models = root / "data" / "perth" / "models"
    (models / "hurdle_day2.joblib").mkdir(parents=True)
    with pytest.raises(G.EnsoGuardError, match="not a regular file"):
        G.assert_servable_path(models / "hurdle_day2.joblib", models, root / "archive")


def test_hard_link_to_an_artifact_is_refused(tmp_path):
    root = tmp_path / "proj"
    models = root / "data" / "perth" / "models"
    models.mkdir(parents=True)
    (models / "hurdle_day1.joblib").write_bytes(b"x")
    try:
        os.link(models / "hurdle_day1.joblib", models / "hurdle_day2.joblib")
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")
    with pytest.raises(G.EnsoGuardError, match="hard links"):
        G.assert_servable_path(models / "hurdle_day2.joblib", models, root / "archive")


def test_junction_or_symlinked_models_dir_is_refused_even_when_target_is_not_quarantine(tmp_path):
    """Reviewer: the junction check was masked by the quarantine check (and skipped on Python 3.11)."""
    root = tmp_path / "proj"
    (root / "data" / "perth").mkdir(parents=True)
    other = root / "data" / "perth" / "models_enso_experimental"
    other.mkdir()
    (other / "hurdle_day1.joblib").write_bytes(b"x")
    link = root / "data" / "perth" / "models"
    try:
        if sys.platform == "win32":
            import _winapi
            _winapi.CreateJunction(str(other), str(link))
        else:
            os.symlink(other, link, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.fail(f"cannot create a directory link on this platform, so the linked-directory guard is untested: {exc}")
    assert G._is_link(link)
    with pytest.raises(G.EnsoGuardError, match="linked directory"):
        G.assert_servable_path(link / "hurdle_day1.joblib", link, root / "archive")


def test_manifest_hashes_are_normalised_and_bom_tolerant(tmp_path):
    q = tmp_path / "archive" / "INVALIDATED_x"
    q.mkdir(parents=True)
    h = "AB" * 32
    (q / "manifest.csv").write_bytes(b"\xef\xbb\xbfsha256,status\n" + h.encode() + b",x\n")
    assert ("ab" * 32) in G.quarantined_hashes(tmp_path / "archive")


def test_hash_check_covers_archived_models_without_a_manifest(tmp_path):
    q = tmp_path / "archive" / "superseded_models" / "perth" / "models_before_x"
    q.mkdir(parents=True)
    (q / "hurdle_day1.joblib").write_bytes(b"superseded bytes")
    import hashlib
    assert hashlib.sha256(b"superseded bytes").hexdigest() in G.quarantined_hashes(tmp_path / "archive")


def test_every_archived_model_file_in_this_project_is_hash_blocked():
    root = Path(__file__).resolve().parents[1]
    files = [f for f in (root / "archive").rglob("*.joblib") if f.is_file()]
    assert files, "expected archived artifacts in the project"
    blocked = G.quarantined_hashes(root / "archive")
    from perthrain.provenance import sha256_file
    assert all(sha256_file(f) in blocked for f in files)


def test_scan_active_flag_uses_resolved_paths_with_an_absolute_root(tmp_path, clean_bundles):
    d = tmp_path / "data" / "perth" / "models"
    d.mkdir(parents=True)
    joblib.dump(clean_bundles["day1"], d / "hurdle_day1.joblib")
    rows = G.scan_artifacts(tmp_path.resolve())          # absolute root
    assert rows and rows[0]["active"] is True
    # and a violating active artifact is REPORTED, not hidden
    b = copy.deepcopy(clean_bundles["day1"])
    b["fits"]["ecmwf_ifs_sr"]["climate"] = "nino34"
    joblib.dump(b, d / "hurdle_day1.joblib")
    assert G.scan_artifacts(tmp_path.resolve())[0]["violations"]


def test_caller_supplied_bundles_are_validated(clean_bundles):
    cfg = Config(location_name="Perth", data_dir=Path("data/perth"), reports_dir=Path("reports/perth"))
    bad = copy.deepcopy(clean_bundles)
    bad["day1"]["enso_status"] = "experimental_unverified"
    bad["day1"]["meta"]["enso_status"] = "experimental_unverified"
    root = Path(__file__).resolve().parents[1]
    cfg.data_dir, cfg.reports_dir, cfg.raw_root = root / "data" / "perth", root / "reports" / "perth", root / "data"
    with pytest.raises(G.EnsoGuardError):
        P.compute_predictions(cfg, bundles=bad, refresh_live=False)


# ==================================================================================== prediction
NOW = pd.Timestamp("2026-09-30T14:56Z")


def test_no_upcoming_window_returns_an_empty_frame_not_a_crash():
    now = pd.Timestamp("2026-09-30T02:00Z")          # 10:00 AWST
    w = P.future_windows(now, TZ, 1)
    assert list(w.columns) == ["label_date_local", "window_start_utc", "window_end_utc"]
    with pytest.raises(ValueError):
        P.future_windows(now, TZ, 0)


def test_empty_live_features_give_exhausted_rows(clean_bundles):
    w = P.future_windows(NOW, TZ, 2)
    pred = P.predict_windows(clean_bundles, pd.DataFrame(), w, TZ)
    assert (pred.status == "no_complete_forecast").all() and (pred.route == "exhausted").all()
    assert (pred.models_available == "").all()


def test_negative_hourly_forecast_makes_the_input_unusable():
    """Training excludes windows with a negative forecast hour; the live path must too."""
    from perthrain import forecasts
    cfg = Config()
    w = P.future_windows(NOW, TZ, 4)
    times = pd.date_range("2026-09-30T00:00", "2026-10-06T23:00", freq="h")
    vals = [0.1] * len(times)
    body = {"latitude": 0, "longitude": 0, "elevation": 0,
            "hourly_units": {"time": "iso8601", "precipitation_previous_day2": "mm"},
            "hourly": {"time": times.strftime("%Y-%m-%dT%H:%M").tolist(), "precipitation_previous_day2": vals}}
    vals[list(times).index(pd.Timestamp("2026-10-01T10:00"))] = -0.2
    pr = forecasts.parse_previous_runs(body, "ncep_gfs_global", [2], "u", {}, "t")
    fw = P.live_feature_table(cfg, pd.DataFrame(), pr, w, NOW)
    r = fw[(fw.col == "pr_ncep_gfs_global") & (fw.lead_day == 2) & (fw.label_date_local == date(2026, 10, 2))].iloc[0]
    assert r.forecast_complete and not r.usable


def test_records_are_never_overwritten_and_use_utc_names(tmp_path, clean_bundles):
    cfg = _cfg_here(tmp_path)
    st = {"station_id": "1", "station_name": "X", "distance_km": 1.0, "latitude": 0, "longitude": 0}
    now = pd.Timestamp("2026-09-30T22:13+08:00")                                # 14:13Z
    pred = pd.DataFrame({"label_date_local": [date(2026, 10, 2)], "status": ["no_complete_forecast"],
                         "lead_group": ["day2"], "route": ["exhausted"]})
    a = P.write_outputs(cfg, st, pred, now, [], clean_bundles)
    b = P.write_outputs(cfg, st, pred, now, [], clean_bundles)
    names = sorted(p.name for p in (cfg.data_dir / "predictions").iterdir())
    assert "forecast_20260930T1413Z.csv" in names and "forecast_20260930T1413Z_2.csv" in names
    assert "(2026-09-30 14:13Z)" in (cfg.data_dir / "predictions" / "latest.md").read_text()


def test_offline_fetcher_never_uses_the_network(tmp_path):
    from perthrain.http import Fetcher
    f = Fetcher(tmp_path)
    f.offline = True
    r = f.get_json("https://example.invalid/x", {"a": 1}, "sub")
    assert not r.ok and r.error.startswith("offline") and f.stats["network"] == 0


def test_served_rows_do_not_carry_raw_forecast_columns(clean_bundles):
    """Served rows expose NWP totals only as clearly named provenance inputs (input_*), never as 'raw_' forecasts."""
    cfg = Config()
    w = P.future_windows(NOW, TZ, 4)
    fw = pd.DataFrame([{"label_date_local": r.label_date_local, "lead_day": n, "col": c, "usable": True,
                        "forecast_precip_mm": 1.0, "forecast_complete": True, "timing_ok": True,
                        "forecast_issue_time_utc": pd.NaT, "issue_time_inferred_max_utc": pd.Timestamp("2026-09-29T06:00Z"),
                        "retrieved_at_utc": "t"}
                       for r in w.itertuples() for n in (1, 2, 3) for c in P.ALL_MODEL_COLS])
    pred = P.predict_windows(clean_bundles, fw, w, TZ)
    ok = pred[pred.status == "ok"]
    assert len(ok) and not [c for c in pred.columns if c.startswith("raw_")]
    assert [c for c in pred.columns if c.startswith("input_")]
