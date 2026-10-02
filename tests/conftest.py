"""Shared fixtures: small but REAL fitted HurdleModels, wrapped into artifacts shaped exactly like calibrate's."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from perthrain import calibrate as C


def make_fit(cols, combine=None, season=True, climate=None, n=240, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"label_date_local": pd.date_range("2024-01-01", periods=n, freq="D")})
    for c in cols:
        df[f"fc_{c}_mm"] = rng.gamma(0.6, 3.0, n)
    for c in C.CLIMATE_COLS.get(climate, []):
        df[c] = rng.normal(0.0, 1.0, n)
    fc = df[[f"fc_{c}_mm" for c in cols]].mean(axis=1).to_numpy()
    y = np.where(rng.random(n) < 1 - np.exp(-fc), rng.gamma(2.0, np.maximum(fc, 0.3)), 0.0)
    model = C.HurdleModel().fit(C.make_X(df, cols, combine, season, climate), y)
    return {"model": model, "cols": list(cols), "combine": combine, "season": season, "climate": climate,
            "clim_range": {c: (-1.0, 1.0) for c in C.CLIMATE_COLS.get(climate, [])}, "C": 1.0, "n_train": n,
            "feature_schema": C.feature_names(list(cols), combine, season, climate),
            "train_first": "2024-01-01", "train_last": str((pd.Timestamp("2024-01-01") + pd.Timedelta(days=n - 1)).date())}


def make_bundle(fits: dict, lead: str, order=None, status: str = "no_enso"):
    order = list(order or fits)
    meta = {"run_id": "noenso-test-00000000", "created_utc": "2026-09-30T00:00:00Z", "enso_status": status,
            "code_sha256": "0" * 64, "source_data": {"training_wide": {"sha256": "1" * 64}},
            "versions": {"python": "3.12"}, "lead": lead, "location": "Perth", "gauge_data_dir": "perth"}
    top = fits[order[0]]
    return {"model": top["model"], "cols": top["cols"], "feature_set": order[0], "C": 1.0, "season": top["season"],
            "combine": top["combine"], "lead": lead, "n_train": top["n_train"], "train_first": top["train_first"],
            "train_last": top["train_last"], "fits": dict(fits), "routing_order": order,
            "climate": None, "enso_status": status, "meta": meta, "holdout": [],
            "ranking": [{"feature_set": n, "climate": fits[n]["climate"]} for n in order]}


# the five feature sets, fitted once per test session (quantile regression is the slow part)
@pytest.fixture(scope="session")
def clean_fits():
    return {name: make_fit(spec["cols"], spec.get("combine"), season=True, seed=i)
            for i, (name, spec) in enumerate(C.FEATURE_SETS.items())}


@pytest.fixture(scope="session")
def clean_bundles(clean_fits):
    """Artifacts for the three leads with a realistic routing order (composite first, single models after)."""
    order23 = ["all_composite", "ecmwf_composite", "ecmwf_ifs_sr", "jma_gsm_pr", "ecmwf_ifs025_pr"]
    return {"day1": make_bundle({"ecmwf_ifs_sr": clean_fits["ecmwf_ifs_sr"]}, "day1"),
            "day2": make_bundle({n: clean_fits[n] for n in order23}, "day2", order23),
            "day3": make_bundle({n: clean_fits[n] for n in order23}, "day3", order23)}


def write_manifest(models_dir, run_id="noenso-test-00000000"):
    """manifest.json listing the hurdle files currently in ``models_dir`` with their SHA-256 (as calibration does)."""
    import hashlib
    import json
    from pathlib import Path
    arts = [{"lead": p.stem.split("_")[1], "file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(Path(models_dir).glob("hurdle_*.joblib"))]
    (Path(models_dir) / "manifest.json").write_text(json.dumps({"meta": {"run_id": run_id}, "artifacts": arts}),
                                                    encoding="utf-8")


def install_models(models_dir, bundles: dict):
    """Write a complete model set (one artifact per lead) plus its manifest."""
    import joblib
    from pathlib import Path
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    for lead, b in bundles.items():
        joblib.dump(b, Path(models_dir) / f"hurdle_{lead}.joblib")
    write_manifest(models_dir)
