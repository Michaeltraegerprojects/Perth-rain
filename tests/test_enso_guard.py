"""The no-ENSO guard: what it must reject, what it must allow (JMA rainfall), and the ACTIVE artifacts."""
import copy
import glob
from pathlib import Path

import joblib
import pandas as pd
import pytest

from conftest import make_bundle, make_fit
from perthrain import enso_guard as G
from perthrain.config import Config
from perthrain.predict import load_bundles

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ what is allowed
def test_clean_artifact_with_jma_rainfall_passes(clean_bundles):
    for lead, b in clean_bundles.items():
        assert G.validate_artifact(b, lead) == [], lead
    # JMA GSM rainfall is a legitimate forecast input; only climate-index inputs are prohibited
    assert "pr_jma_gsm" in clean_bundles["day3"]["fits"]["jma_gsm_pr"]["cols"]


@pytest.mark.parametrize("name", ["log1p(fc_pr_jma_gsm_mm)", "log1p(fc_sr_ecmwf_ifs_mm)",
                                  "mean(log1p(fc_sr_ecmwf_ifs_mm), log1p(fc_pr_jma_gsm_mm))",
                                  "season_sin(day_of_year)", "season_cos(day_of_year)",
                                  "period_days", "persistence_mm", "passthrough_mm", "resistance", "sensor", "gesture"])   # substrings inside words
def test_legitimate_feature_names_are_not_flagged(name):
    assert not G.forbidden(name)


# ------------------------------------------------------------------ what is rejected
@pytest.mark.parametrize("name", ["clim_nino34", "clim_iod", "Niño3.4", "nino_3.4", "rnino34", "enso_phase",
                                  "sst_anomaly", "SSTA_nino34", "iod", "soi_30d", "ONI", "iod_weekly", "oni_3m",
                                  "weekly_sst", "SOI", "roni_3m", "RONI", "dmi", "DMI_weekly", "MEI.v2", "sst34", "SSTs", "Indian_Ocean_Dipole",
                                  "southern oscillation index", "Niño3.4", "NIÑO4",
                                  "log1p(fc_pr_jma_gsm_mm)*clim_nino34", "interaction(nino34, season_sin)"])
def test_enso_and_sst_anomaly_features_and_interactions_are_flagged(name):
    assert G.forbidden(name)


def test_jma_fallback_trained_with_climate_index_is_rejected(clean_bundles):
    b = copy.deepcopy(clean_bundles["day3"])
    b["fits"]["jma_gsm_pr"] = make_fit(["pr_jma_gsm"], climate="nino34_iod", seed=7)
    v = G.validate_artifact(b, "day3")
    assert any("jma_gsm_pr: trained with climate configuration 'nino34_iod'" in x for x in v)
    assert any("jma_gsm_pr: forbidden feature 'clim_nino34'" in x for x in v)
    with pytest.raises(G.EnsoGuardError):
        G.assert_no_enso(b, "test", "day3")


def test_label_alone_is_not_trusted(clean_bundles):
    """enso_status says no_enso, but a fit's schema carries an ENSO interaction term -> rejected."""
    b = copy.deepcopy(clean_bundles["day2"])
    b["fits"]["jma_gsm_pr"]["feature_schema"] = ["log1p(fc_pr_jma_gsm_mm)*clim_nino34", "season_sin(day_of_year)",
                                                 "season_cos(day_of_year)"]
    assert b["enso_status"] == "no_enso"
    v = G.validate_artifact(b, "day2")
    assert any("forbidden feature" in x for x in v)
    assert any("!= expected ordered schema" in x for x in v)


def test_hidden_column_is_detected(clean_bundles):
    """Estimator fitted on 5 inputs (2 hidden climate columns) while the schema declares 3."""
    b = copy.deepcopy(clean_bundles["day3"])
    hidden = make_fit(["pr_jma_gsm"], climate="nino34_iod", seed=8)
    hidden.update(climate=None, clim_range={}, feature_schema=["log1p(fc_pr_jma_gsm_mm)", "season_sin(day_of_year)",
                                                               "season_cos(day_of_year)"])
    b["fits"]["jma_gsm_pr"] = hidden
    v = G.validate_artifact(b, "day3")
    assert any("expects 5 inputs but the schema declares 3" in x for x in v)


def test_reordered_schema_is_rejected(clean_bundles):
    b = copy.deepcopy(clean_bundles["day2"])
    b["fits"]["jma_gsm_pr"]["feature_schema"] = list(reversed(b["fits"]["jma_gsm_pr"]["feature_schema"]))
    assert any("!= expected ordered schema" in x for x in G.validate_artifact(b, "day2"))


@pytest.mark.parametrize("mutate,needle", [
    (lambda b: b.pop("meta"), "no 'meta' block"),
    (lambda b: b["meta"].pop("run_id"), "meta.run_id missing"),
    (lambda b: b["meta"].pop("source_data"), "meta.source_data missing"),
    (lambda b: b["meta"].pop("versions"), "meta.versions missing"),
    (lambda b: b.update(enso_status="experimental_unverified"), "expected 'no_enso'"),
    (lambda b: b["meta"].update(enso_status="experimental_unverified"), "meta.enso_status"),
    (lambda b: b.update(lead="day2"), "lead mismatch"),
    (lambda b: b.update(routing_order=["all_composite", "ghost_set"]), "has no fit"),
    (lambda b: b.pop("routing_order"), "routing_order missing"),
    (lambda b: b["fits"]["jma_gsm_pr"].update(cols=["pr_bom_access_global"]), "differ from the code's definition"),
    (lambda b: b.update(fits_noclim={"x": {}}), "legacy 'fits_noclim'"),
])
def test_missing_or_incompatible_metadata_is_rejected(clean_bundles, mutate, needle):
    b = copy.deepcopy(clean_bundles["day3"])
    mutate(b)
    v = G.validate_artifact(b, "day3")
    assert any(needle in x for x in v), v


def test_unknown_feature_set_is_rejected(clean_bundles):
    b = copy.deepcopy(clean_bundles["day3"])
    b["fits"]["mystery_set"] = b["fits"].pop("jma_gsm_pr")
    b["routing_order"] = [n if n != "jma_gsm_pr" else "mystery_set" for n in b["routing_order"]]
    assert any("unknown feature set" in x for x in G.validate_artifact(b, "day3"))


@pytest.mark.parametrize("name", ["mjo_rmm1", "mjo_amplitude", "sam_index", "log1p(fc_pr_bom_access_global_mm)",
                                  "log1p(fc_pr_jma_gsm_mm)^2"])
def test_features_outside_the_approved_baseline_set_are_rejected(clean_bundles, name):
    """MJO/SAM are not named by the ENSO prohibition, but they are not approved features either."""
    assert name not in G.approved_feature_names()
    b = copy.deepcopy(clean_bundles["day3"])
    b["fits"]["jma_gsm_pr"]["feature_schema"] = b["fits"]["jma_gsm_pr"]["feature_schema"] + [name]
    assert any("not in the approved baseline feature set" in x or "forbidden feature" in x
               for x in G.validate_artifact(b, "day3"))


EXPECTED_APPROVED = {
    "log1p(fc_sr_ecmwf_ifs_mm)", "log1p(fc_pr_ecmwf_ifs025_mm)", "log1p(fc_pr_jma_gsm_mm)",
    "mean(log1p(fc_sr_ecmwf_ifs_mm), log1p(fc_pr_ecmwf_ifs025_mm))",
    "mean(log1p(fc_sr_ecmwf_ifs_mm), log1p(fc_pr_jma_gsm_mm), log1p(fc_pr_ncep_gfs_global_mm), log1p(fc_pr_ecmwf_ifs025_mm))",
    "season_sin(day_of_year)", "season_cos(day_of_year)"}


def test_approved_baseline_feature_set_is_exactly_the_configured_one():
    ok = G.approved_feature_names()
    assert "log1p(fc_pr_jma_gsm_mm)" in ok and "season_sin(day_of_year)" in ok
    assert not any(G.forbidden(f) for f in ok)
    assert ok == EXPECTED_APPROVED


# ------------------------------------------------------------- loading from disk
@pytest.fixture()
def project(tmp_path):
    """A project that itself sits under a parent folder literally called 'archive'."""
    root = tmp_path / "archive" / "proj"
    (root / "data" / "perth" / "models").mkdir(parents=True)
    q = root / "archive" / "INVALIDATED_x" / "model_artifacts" / "perth"
    q.mkdir(parents=True)
    (q / "hurdle_day1.joblib").write_bytes(b"quarantined")
    (root / "data" / "perth" / "models" / "hurdle_day1.joblib").write_bytes(b"served")
    return root


def test_project_under_a_parent_folder_named_archive_is_accepted(project):
    models = project / "data" / "perth" / "models"
    rp = G.assert_servable_path(models / "hurdle_day1.joblib", models, project / "archive")
    assert rp == (models / "hurdle_day1.joblib").resolve()


def test_benign_traversal_inside_the_models_dir_is_accepted(project):
    models = project / "data" / "perth" / "models"
    G.assert_servable_path(models / ".." / "models" / "hurdle_day1.joblib", models, project / "archive")


def test_traversal_into_the_quarantine_root_is_refused(project):
    models = project / "data" / "perth" / "models"
    sneaky = models / ".." / ".." / ".." / "archive" / "INVALIDATED_x" / "model_artifacts" / "perth" / "hurdle_day1.joblib"
    with pytest.raises(G.EnsoGuardError, match="inside the quarantine root"):
        G.assert_servable_path(sneaky, models, project / "archive")


def test_artifact_outside_the_models_dir_is_refused(project):
    models = project / "data" / "perth" / "models"
    other = project / "data" / "perth" / "models_enso_experimental"
    other.mkdir()
    (other / "hurdle_day1.joblib").write_bytes(b"x")
    with pytest.raises(G.EnsoGuardError, match="not directly inside the models directory"):
        G.assert_servable_path(other / "hurdle_day1.joblib", models, project / "archive")


def test_file_symlink_to_a_quarantined_artifact_is_refused(project):
    import os
    models = project / "data" / "perth" / "models"
    link = models / "hurdle_day2.joblib"
    target = project / "archive" / "INVALIDATED_x" / "model_artifacts" / "perth" / "hurdle_day1.joblib"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"this OS/user may not create file symlinks: {exc}")
    with pytest.raises(G.EnsoGuardError, match="is a link"):
        G.assert_servable_path(link, models, project / "archive")


def test_linked_models_directory_into_quarantine_is_refused(project, tmp_path):
    """data/<loc>/models replaced by a directory junction (Windows) / symlink (POSIX) into the quarantine."""
    import os
    import sys
    loc = project / "data" / "clarkson"
    loc.mkdir()
    target = project / "archive" / "INVALIDATED_x" / "model_artifacts" / "perth"
    link = loc / "models"
    try:
        if sys.platform == "win32":
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
        else:
            os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"cannot create a directory link here: {exc}")
    with pytest.raises(G.EnsoGuardError, match="linked directory|inside the quarantine root"):
        G.assert_servable_path(link / "hurdle_day1.joblib", link, project / "archive")


def _cfg(tmp_path):
    return Config(data_dir=tmp_path / "data" / "perth", reports_dir=tmp_path / "reports" / "perth",
                  raw_root=tmp_path / "data")


def test_loader_refuses_byte_identical_copy_of_a_quarantined_file(tmp_path, clean_bundles):
    import csv
    cfg = _cfg(tmp_path)
    (cfg.data_dir / "models").mkdir(parents=True)
    art = cfg.data_dir / "models" / "hurdle_day1.joblib"
    joblib.dump(clean_bundles["day1"], art)
    q = tmp_path / "archive" / "INVALIDATED_test"
    q.mkdir(parents=True)
    from perthrain.provenance import sha256_file
    with open(q / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["archived_path", "sha256", "status"])
        w.writeheader()
        w.writerow({"archived_path": "x", "sha256": sha256_file(art), "status": "INVALIDATED_ENSO_FIT_PRESENT"})
    with pytest.raises(G.EnsoGuardError, match="byte-identical to a quarantined file"):
        load_bundles(cfg)


def test_loader_rejects_corrupt_foreign_and_mislabelled_artifacts(tmp_path, clean_bundles):
    cfg = _cfg(tmp_path)
    mdir = cfg.data_dir / "models"
    mdir.mkdir(parents=True)
    (mdir / "hurdle_day1.joblib").write_bytes(b"\x80\x04 this is not a pickle")
    with pytest.raises(G.EnsoGuardError, match="unreadable or corrupt"):
        load_bundles(cfg)
    joblib.dump(["not", "an", "artifact"], mdir / "hurdle_day1.joblib")
    with pytest.raises(G.EnsoGuardError, match="not a perthrain artifact"):
        load_bundles(cfg)
    joblib.dump(clean_bundles["day3"], mdir / "hurdle_day1.joblib")         # day-3 artifact in the day-1 slot
    with pytest.raises(G.EnsoGuardError, match="lead mismatch"):
        load_bundles(cfg)
    joblib.dump(clean_bundles["day1"], mdir / "hurdle_day1.joblib")
    assert set(load_bundles(cfg)) == {"day1"}                               # clean artifact loads


def test_loader_rejects_a_quarantined_pre_audit_artifact_copied_back(tmp_path):
    src = sorted(ROOT.glob("archive/INVALIDATED_*/model_artifacts/perth/hurdle_day3.joblib"))
    if not src:
        pytest.skip("quarantined artifact not present")
    cfg = _cfg(tmp_path)          # separate project: its quarantine manifest does not list this file
    (cfg.data_dir / "models").mkdir(parents=True)
    (cfg.data_dir / "models" / "hurdle_day3.joblib").write_bytes(src[0].read_bytes())
    with pytest.raises(G.EnsoGuardError, match="no 'meta' block|trained with climate"):
        load_bundles(cfg)         # rejected on CONTENT even without the hash list


# ---------------------------------------------------- the ACTIVE artifacts / payloads
ACTIVE = sorted(glob.glob(str(ROOT / "data" / "*" / "models" / "hurdle_*.joblib")))
assert ACTIVE, "no active artifacts: run `calibrate`"


@pytest.mark.parametrize("path", ACTIVE, ids=[Path(p).parent.parent.name + "/" + Path(p).stem for p in ACTIVE])
def test_every_active_artifact_is_a_clean_no_enso_artifact(path):
    b = joblib.load(path)
    lead = Path(path).stem.split("_")[-1]
    assert b["enso_status"] == "no_enso"
    assert G.validate_artifact(b, lead) == []
    for name, fit in b["fits"].items():
        assert fit["climate"] in (None, "none") and not fit["clim_range"], name
        assert not any(G.forbidden(f) for f in fit["feature_schema"]), name


PAYLOADS = sorted(glob.glob(str(ROOT / "data" / "*" / "predictions" / "*.csv")))


def test_active_prediction_payloads_exist():
    assert PAYLOADS, "no active prediction payloads: run `predict`"


@pytest.mark.parametrize("path", PAYLOADS, ids=[Path(p).parent.parent.name + "/" + Path(p).name for p in PAYLOADS])
def test_active_prediction_payloads_come_from_the_active_clean_run(path):
    """EVERY payload file (timestamped records and latest.csv), EVERY row, and the VALUES of the schema and
    provenance columns - not only the column names."""
    d = pd.read_csv(path)
    # 'enso_status' is the provenance label (value must be 'no_enso'), not a climate feature
    bad = [c for c in d.columns if c not in G.PROVENANCE_COLUMNS and G.forbidden(c)]
    assert not bad, f"climate columns in a no-ENSO payload: {bad}"
    assert (d.enso_status == "no_enso").all(), "every row (served or not) must carry enso_status=no_enso"
    served = d[d.status == "ok"]
    for schema in served.feature_schema.dropna():
        assert not any(G.forbidden(f) for f in schema.split(" | ")), schema
        assert all(f in G.approved_feature_names() for f in schema.split(" | ")), schema
    mdir = Path(path).parent.parent / "models"
    active_ids = {joblib.load(p)["meta"]["run_id"] for p in mdir.glob("hurdle_*.joblib")}
    assert len(active_ids) == 1
    assert set(d.run_id.dropna()) <= active_ids, "payload was not produced by the active artifacts"
    assert not d.run_id.isna().any() or (d[d.run_id.isna()].status == "no_model_artifact").all()
