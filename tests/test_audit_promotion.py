"""Finding 2 of the external review of 0de31ee: model promotion must be all-or-nothing, and the loader must refuse
an incomplete or inconsistent model directory. Written to fail on 0de31ee."""
import os
import shutil

import joblib
import pytest

from perthrain import calibrate as C
from perthrain import enso_guard as G
from perthrain import predict as P
from test_review_regressions import _project, synth_wide


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory):
    """A project with one complete, promoted model set."""
    root = tmp_path_factory.mktemp("promo")
    cfg = _project(root, synth_wide(n=700))
    C.run_calibration(cfg)
    return root, cfg


def _copy(calibrated, tmp_path):
    root, cfg = calibrated
    dst = tmp_path / "p"
    shutil.copytree(root, dst)
    from perthrain.config import Config
    return Config(data_dir=dst / "data" / "loc", reports_dir=dst / "reports" / "loc", raw_root=dst / "data",
                  location_name="Loc")


def _snapshot(d):
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


def test_failure_while_promoting_leaves_the_live_models_intact(calibrated, tmp_path, monkeypatch):
    cfg = _copy(calibrated, tmp_path)
    models = cfg.data_dir / "models"
    before = _snapshot(models)
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise PermissionError("simulated: file locked by another process")
        return real(src, dst)
    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(PermissionError):
        C.run_calibration(cfg)
    monkeypatch.setattr(os, "replace", real)
    assert _snapshot(models) == before, "live model directory changed after a failed promotion"
    assert not list(cfg.data_dir.glob("staging_*")), "staging directory left behind"
    assert set(P.load_bundles(cfg)) == {"day1", "day2", "day3"}


def test_failure_while_archiving_the_old_set_keeps_both_sets(calibrated, tmp_path, monkeypatch):
    cfg = _copy(calibrated, tmp_path)
    old = _snapshot(cfg.data_dir / "models")

    def broken_move(*a, **k):
        raise PermissionError("simulated: archive not writable")
    from perthrain import provenance
    monkeypatch.setattr(provenance, "archive_dir", broken_move, raising=False)
    C.run_calibration(cfg)                                    # promotion itself succeeded
    assert set(P.load_bundles(cfg)) == {"day1", "day2", "day3"}
    kept = [d for d in cfg.data_dir.iterdir() if d.is_dir() and d.name.startswith(".models_previous_")]
    assert len(kept) == 1 and _snapshot(kept[0]) == old, "the previous model set must not be lost"


def test_loader_refuses_a_directory_missing_a_lead(calibrated, tmp_path):
    cfg = _copy(calibrated, tmp_path)
    (cfg.data_dir / "models" / "hurdle_day2.joblib").unlink()
    with pytest.raises(G.EnsoGuardError, match="day2"):
        P.load_bundles(cfg)


def test_loader_refuses_a_directory_without_a_manifest(calibrated, tmp_path):
    cfg = _copy(calibrated, tmp_path)
    (cfg.data_dir / "models" / "manifest.json").unlink()
    with pytest.raises(G.EnsoGuardError, match="manifest"):
        P.load_bundles(cfg)


def test_loader_refuses_an_artifact_not_listed_in_the_manifest(calibrated, tmp_path):
    cfg = _copy(calibrated, tmp_path)
    d = cfg.data_dir / "models"
    b = joblib.load(d / "hurdle_day3.joblib")
    b["C"] = 123.0                                          # a valid-looking but different artifact
    joblib.dump(b, d / "hurdle_day3.joblib")
    with pytest.raises(G.EnsoGuardError, match="manifest"):
        P.load_bundles(cfg)
