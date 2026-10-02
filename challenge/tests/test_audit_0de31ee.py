"""Regression tests for the external review of commit 0de31ee (findings 1, 3-7) and the timing-evidence request.
Each test was written to fail on 0de31ee before the fix."""
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from challenge import champion, historical, ledger, scoring


# ------------------------------------------------------------------------- finding 1: identical errors
def test_identical_errors_are_a_tie_not_a_significant_win():
    r = scoring.block_bootstrap(np.zeros(200))
    assert r["p"] == 1.0, "zero variance and zero mean difference is a tie, not p = 0"
    df = pd.DataFrame([{"first": "A", "second": "B", "first_key": "champion_median", "second_key": "x",
                        "n_days": 200, "n_wet_days": 60, "mean_abs_error_diff": r["mean_diff"],
                        "mean_diff_exact": r["mean_diff"], "p_bootstrap": r["p"]}])
    assert scoring.holm(df).verdict.iloc[0] == "inconclusive"


def test_identical_forecasts_through_the_full_paired_path():
    rng = np.random.default_rng(0)
    y = np.where(rng.random(200) < 0.4, rng.gamma(1.5, 3, 200), 0.0)
    t = pd.DataFrame({"window_start_utc": pd.date_range("2026-01-01T01:00Z", periods=200, freq="D"),
                      "observed_precip_mm": y, "champion_median_mm": y + 0.5, "champion_expected_mm": y + 0.7,
                      "fc_sr_ecmwf_ifs_mm": y + 0.5})              # identical to our median on every day
    cols = {"champion_median": "champion_median_mm", "champion_expected": "champion_expected_mm",
            "sr_ecmwf_ifs": "fc_sr_ecmwf_ifs_mm"}
    p = scoring.holm(scoring.paired(t, cols))
    row = p[(p.first_key == "champion_median") & (p.second_key == "sr_ecmwf_ifs")].iloc[0]
    assert row.verdict == "inconclusive"


def test_verdict_uses_the_unrounded_difference():
    """A consistent tiny advantage for the first forecast must not be reported as 'second better' because the
    stored difference was rounded to -0.0."""
    rng = np.random.default_rng(1)
    n = 200
    y = np.where(rng.random(n) < 0.5, rng.gamma(2, 3, n), 0.0)
    err = 1.0 + rng.random(n)
    t = pd.DataFrame({"window_start_utc": pd.date_range("2026-01-01T01:00Z", periods=n, freq="D"),
                      "observed_precip_mm": y, "champion_median_mm": y + err, "champion_expected_mm": y + err,
                      "fc_x_mm": y + err + 4e-5 * (1 + 0.01 * rng.random(n))})
    cols = {"champion_median": "champion_median_mm", "champion_expected": "champion_expected_mm", "x": "fc_x_mm"}
    p = scoring.holm(scoring.paired(t, cols))
    row = p[(p.first_key == "champion_median") & (p.second_key == "x")].iloc[0]
    assert row.p_holm < 0.05
    assert row.verdict == "first better"


# ------------------------------------------------------------------------- finding 6: verdict settings
def test_verdict_rules_come_from_settings(tmp_path):
    s = tmp_path / "settings.toml"
    s.write_text('[schedule]\nissue_times_utc=["21:00"]\nmax_delay_minutes=180\nhorizon_days=4\n'
                 '[verdicts]\nmin_days = 10\nmin_wet_days = 2\nalpha = 0.05\n', encoding="utf-8")
    rules = scoring.load_rules(s)
    assert rules == {"min_days": 10, "min_wet_days": 2, "alpha": 0.05}
    df = pd.DataFrame([{"first": "A", "second": "B", "first_key": "champion_median", "second_key": "x", "n_days": 20,
                        "n_wet_days": 5, "mean_abs_error_diff": -0.3, "mean_diff_exact": -0.3, "p_bootstrap": 0.001}])
    assert scoring.holm(df, rules).verdict.iloc[0] == "first better"
    assert scoring.holm(df, {"min_days": 60, "min_wet_days": 15, "alpha": 0.05}).verdict.iloc[0] == "insufficient evidence"


def test_default_rules_equal_the_settings_file():
    import tomllib
    cfg = tomllib.loads(ledger.SETTINGS.read_text(encoding="utf-8"))["verdicts"]
    assert scoring.load_rules() == cfg


@pytest.mark.parametrize("bad", ['min_days = 0', 'alpha = 1.5', 'min_wet_days = "x"'])
def test_invalid_verdict_settings_are_rejected(tmp_path, bad):
    s = tmp_path / "settings.toml"
    good = {"min_days": "min_days = 60", "min_wet_days": "min_wet_days = 15", "alpha": "alpha = 0.05"}
    good[bad.split(" =")[0]] = bad
    s.write_text("[verdicts]\n" + "\n".join(good.values()) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        scoring.load_rules(s)


# ------------------------------------------------------------------------- finding 5: ledger integrity
def _rec(i):
    return {"issue_id": f"2026100{i}T210000Z", "location": "Perth", "gauge_id": "009225", "competitor": "x",
            "label_date": "2026-10-03", "window_start_utc": "2026-10-02 01:00:00+00:00",
            "window_end_utc": "2026-10-03 01:00:00+00:00", "lead": "day1", "status": "ok", "total_mm": float(i)}


def test_ledger_detects_deletion_of_final_records(tmp_path):
    p = tmp_path / "ledger.jsonl"
    ledger.append([_rec(1), _rec(2), _rec(3)], p)
    ledger.append([_rec(4), _rec(5)], p)
    lines = p.read_text(encoding="utf-8").splitlines()
    p.write_text("\n".join(lines[:3]) + "\n", encoding="utf-8")          # drop the last two complete records
    with pytest.raises(ledger.LedgerError, match="truncated|checkpoint"):
        ledger.read_ledger(p)


def test_torn_final_line_is_a_ledger_error_not_a_crash(tmp_path):
    p = tmp_path / "ledger.jsonl"
    ledger.append([_rec(1), _rec(2)], p)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write('{"issue_id": "202610')                               # crash mid-append
    with pytest.raises(ledger.LedgerError):
        ledger.read_ledger(p)
    with pytest.raises(ledger.LedgerError):
        ledger.append([_rec(3)], p)


def test_a_second_writer_is_refused_while_the_lock_is_held(tmp_path):
    p = tmp_path / "ledger.jsonl"
    ledger.append([_rec(1)], p)
    with ledger.writer_lock(p):
        with pytest.raises(ledger.LedgerError, match="lock"):
            ledger.append([_rec(2)], p)
    ledger.append([_rec(2)], p)                                        # released: appending works again
    assert len(ledger.read_ledger(p)) == 2


# ------------------------------------------------------------------------- findings 3 and 4: champion freeze
class _Marker:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (open, (self.path, "w"))      # unpickling this object creates a file


def test_verify_compares_hashes_before_unpickling_anything(synthetic_project):
    champion.freeze()
    marker = synthetic_project / "UNPICKLED"
    art = synthetic_project / "data" / "perth" / "models" / "hurdle_day2.joblib"
    with open(art, "wb") as fh:
        pickle.dump(_Marker(str(marker)), fh)
    with pytest.raises(champion.ChampionChanged):
        champion.verify()
    assert not marker.exists(), "verify() deserialised an artifact whose bytes no longer match the freeze"


def test_freeze_covers_config_inference_code_and_runtime(synthetic_project):
    rec = json.loads(champion.freeze().read_text())
    files = rec["inference_files"]
    for f in ("config.toml", "src/perthrain/config.py", "src/perthrain/observations.py",
              "src/perthrain/provenance.py", "src/perthrain/http.py", "src/perthrain/predict.py",
              "src/perthrain/calibrate.py", "src/perthrain/build.py", "src/perthrain/forecasts.py"):
        assert f in files, f
    assert not any(f.endswith(("report.py", "calibrate_report.py", "noaa_cpc.py")) for f in files)
    for pkg in ("python", "numpy", "pandas", "scikit-learn", "scipy", "joblib"):
        assert pkg in rec["runtime"], pkg


def test_changing_config_is_detected(synthetic_project):
    champion.freeze()
    cfg = synthetic_project / "config.toml"
    cfg.write_text(cfg.read_text().replace("publication_latency_hours = 6", "publication_latency_hours = 9"))
    with pytest.raises(champion.ChampionChanged, match="config.toml"):
        champion.verify()


def test_runtime_mismatch_is_detected(synthetic_project, monkeypatch):
    champion.freeze()
    cur = champion.runtime_versions()
    monkeypatch.setattr(champion, "runtime_versions", lambda: dict(cur, **{"scikit-learn": "0.20.0"}))
    with pytest.raises(champion.ChampionChanged, match="scikit-learn"):
        champion.verify()


def test_existing_record_is_never_overwritten_by_a_different_one(synthetic_project):
    p = champion.freeze()
    before = p.read_bytes()
    cfg = synthetic_project / "config.toml"
    cfg.write_text(cfg.read_text() + "\n# changed\n")
    with pytest.raises(champion.ChampionChanged):
        champion.freeze()
    assert p.read_bytes() == before


# ------------------------------------------------------------------------- finding 7: exact reproduction
def _tamper(root, lead, **changes):
    p = root / "reports" / "perth" / "calibration_metrics.csv"
    m = pd.read_csv(p)
    sel = (m.lead == lead) & (m.split == "holdout") & (m.method == "hurdle")
    for k, v in changes.items():
        m.loc[sel, k] = v
    m.to_csv(p, index=False)


def test_reproduction_passes_on_untouched_outputs(synthetic_project):
    rec = champion.verify() if champion.FREEZE.exists() else json.loads(champion.freeze().read_text())
    tab, meta = historical.champion_holdout("perth", "day2", rec)
    assert meta["n"] == len(tab) > 0


def test_reproduction_checks_the_exact_evaluation_keys(synthetic_project):
    rec = json.loads(champion.freeze().read_text())
    _tamper(synthetic_project, "day2", key_fingerprint="0" * 64)        # n and median MAE unchanged
    with pytest.raises(champion.ChampionChanged, match="key"):
        historical.champion_holdout("perth", "day2", rec)


def test_reproduction_checks_more_than_one_aggregate(synthetic_project):
    rec = json.loads(champion.freeze().read_text())
    m = pd.read_csv(synthetic_project / "reports" / "perth" / "calibration_metrics.csv")
    b = m[(m.lead == "day3") & (m.split == "holdout") & (m.method == "hurdle")]["brier_0.2"].iloc[0]
    _tamper(synthetic_project, "day3", **{"brier_0.2": b + 0.01})
    with pytest.raises(champion.ChampionChanged, match="brier_0.2"):
        historical.champion_holdout("perth", "day3", rec)


def test_reproduction_checks_every_row_against_the_saved_reference(synthetic_project):
    rec = json.loads(champion.freeze().read_text())
    historical.champion_holdout("perth", "day1", rec)            # writes/validates the per-row reference
    ref = historical.reference_path("perth", "day1")
    r = pd.read_parquet(ref)
    r.loc[r.index[5], "champion_p_ge_0.2"] += 0.02               # a per-row change no aggregate check would see
    r.to_parquet(ref, index=False)
    with pytest.raises(champion.ChampionChanged, match="per-row"):
        historical.champion_holdout("perth", "day1", rec)


# ------------------------------------------------------------------------- timing evidence per competitor
def test_competitor_timing_separates_observed_and_estimated_availability():
    from challenge import export
    issued = "2026-10-01T21:00:00Z"
    sr = {"competitor": "icon_global", "issued_at_utc": issued, "latest_run_utc": "2026-10-01 12:00:00+00:00",
          "availability_evidence": "explicit run requested; HTTP 200 at issue time"}
    meta = {"competitor": "icon_global", "issued_at_utc": issued, "latest_run_utc": "2026-10-01 06:00:00+00:00",
            "availability_evidence": "run named by the offset rule; model metadata last run 2026-10-01 12:00:00+00:00"}
    est = {"competitor": "pr_jma_gsm", "issued_at_utc": issued, "latest_run_utc": "2026-10-01 06:00:00+00:00"}
    champ = {"competitor": "champion", "issued_at_utc": issued,
             "inputs": "pr_jma_gsm: runs up to 2026-10-01 06Z (inferred); sr_ecmwf_ifs: run 2026-09-30 12Z (explicit)"}
    a, b, c, d = (export.competitor_timing(x) for x in (sr, meta, est, champ))
    assert a["availability_basis"] == "observed" and a["info_age_h"] == 9.0
    assert b["availability_basis"] == "observed" and b["info_age_h"] == 15.0
    assert c["availability_basis"] == "estimated (initialisation + 6 h)"
    assert d["run_init_utc"] == "2026-10-01T06:00Z" and d["availability_basis"] == "estimated (initialisation + 6 h)"
