"""Self-contained fixtures for challenge tests: a synthetic project (code copy, config, calibrated models, reports)
built in a temporary directory, so the champion/historical logic can be tested on a clean checkout."""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REAL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REAL_ROOT / "src"))
sys.path.insert(0, str(REAL_ROOT))


def synth_wide(n=760, seed=1, leads=("day1", "day2", "day3")):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2023-01-01", periods=n, freq="D")
    rows = []
    for lead in leads:
        base = rng.gamma(0.6, 3.0, n)
        y = np.where(rng.random(n) < 1 - np.exp(-base), rng.gamma(2.0, np.maximum(base, 0.3)), 0.0)
        d = pd.DataFrame({"label_date_local": days, "lead_group": lead, "observed_precip_mm": y.round(1),
                          "station_id": "009225",
                          "window_start_utc": days.tz_localize("UTC") - pd.Timedelta(hours=23),
                          "window_end_utc": days.tz_localize("UTC") + pd.Timedelta(hours=1)})
        for c in ["sr_ecmwf_ifs", "pr_jma_gsm", "pr_ncep_gfs_global", "pr_ecmwf_ifs025"]:
            d[f"fc_{c}_mm"] = np.clip(base * rng.lognormal(0, 0.4, n), 0, None).round(2)
            d[f"eligible_{c}"] = True
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="session")
def synthetic_project_template(tmp_path_factory):
    """Calibrated synthetic project for location 'Perth' (slug 'perth'), built once per session."""
    from perthrain import calibrate
    from perthrain.config import Config
    root = tmp_path_factory.mktemp("proj")
    shutil.copytree(REAL_ROOT / "src" / "perthrain", root / "src" / "perthrain",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (root / "config.toml").write_text('[[locations]]\nname = "Perth"\nlatitude = -31.95\nlongitude = 115.86\n\n'
                                      '[forecast]\npublication_latency_hours = 6\n', encoding="utf-8")
    (root / "challenge").mkdir()
    cfg = Config(data_dir=root / "data" / "perth", reports_dir=root / "reports" / "perth", raw_root=root / "data",
                 location_name="Perth")
    (cfg.data_dir / "joined").mkdir(parents=True)
    (cfg.data_dir / "clean").mkdir()
    synth_wide().to_parquet(cfg.data_dir / "joined" / "training_wide.parquet")
    calibrate.run_calibration(cfg)
    return root


@pytest.fixture
def synthetic_project(synthetic_project_template, tmp_path, monkeypatch):
    """A private copy of the template, with the challenge modules pointed at it."""
    root = tmp_path / "proj"
    shutil.copytree(synthetic_project_template, root)
    from challenge import champion, historical
    monkeypatch.setattr(champion, "ROOT", root)
    monkeypatch.setattr(champion, "FREEZE", root / "challenge" / "champion_freeze.json")
    monkeypatch.setattr(historical, "ROOT", root)
    return root
