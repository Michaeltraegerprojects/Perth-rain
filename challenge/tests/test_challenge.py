"""Tests for the Forecast Challenge: fairness rules, ledger integrity, scoring and champion protection."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from challenge import champion, ledger, scoring, sources  # noqa: E402
from perthrain import forecasts  # noqa: E402


# ------------------------------------------------------------------------------------------- run semantics
@pytest.mark.parametrize("model,step", [("ecmwf_ifs025", 3), ("jma_gsm", 6), ("ncep_gfs_global", 1)])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_offset_rule_matches_the_champions_verified_rule(model, step, n):
    t = pd.Series(pd.date_range("2026-06-01T00:00Z", periods=72, freq="h"))
    mine = sources.rule_run_series(t, step, n)
    theirs = forecasts.inferred_run_time(t, model, n)
    assert (mine == theirs).all()
    assert sources.rule_run_time(t.iloc[5], step, n) == theirs.iloc[5]


def _hourly(values, start="2026-06-01T02:00Z", lead=2, step=1):
    t = pd.date_range(start, periods=len(values), freq="h")
    return pd.DataFrame({"model": "m", "lead_day": lead, "valid_end_utc": t, "precip_mm": values,
                         "grid_latitude": -31.9, "grid_longitude": 115.9, "source_url": "u", "retrieved_at_utc": "r",
                         "issue_time_inferred_utc": sources.rule_run_series(pd.Series(t), step, lead).tolist()})


WIN = pd.DataFrame({"window_start_utc": [pd.Timestamp("2026-06-01T01:00Z")],
                    "window_end_utc": [pd.Timestamp("2026-06-02T01:00Z")]})


def test_complete_window_total_and_lead():
    t = sources.window_totals_previous_runs(_hourly([0.5] * 24), WIN).iloc[0]
    assert t.complete and t.total_mm == pytest.approx(12.0)
    assert t.published_before_cutoff                       # latest run 2026-05-31 00Z + 6 h < window start


def test_missing_hour_gives_missing_total_not_zero():
    v = [0.0] * 24
    v[10] = np.nan
    t = sources.window_totals_previous_runs(_hourly(v), WIN).iloc[0]
    assert not t.complete and np.isnan(t.total_mm)


def test_negative_hour_makes_the_window_missing():
    v = [0.1] * 24
    v[3] = -0.1
    assert np.isnan(sources.window_totals_previous_runs(_hourly(v), WIN).iloc[0].total_mm)


def test_as_of_cutoff_blocks_runs_published_after_issue():
    h = _hourly([0.1] * 24, lead=1)                       # day-1 offset: latest run 2026-06-01 00Z
    early = sources.window_totals_previous_runs(h, WIN, cutoff_rule="as_of", as_of=pd.Timestamp("2026-06-01T03:00Z"))
    late = sources.window_totals_previous_runs(h, WIN, cutoff_rule="as_of", as_of=pd.Timestamp("2026-06-01T07:00Z"))
    assert not early.iloc[0].published_before_cutoff and late.iloc[0].published_before_cutoff


def test_day1_single_run_rule_is_the_champions_12z_rule():
    ws = pd.Series([pd.Timestamp("2026-06-02T01:00Z")])
    assert sources.sr_issue_time(ws, 1).iloc[0] == pd.Timestamp("2026-06-01T12:00Z")
    assert sources.sr_issue_time(ws, 2).iloc[0] == pd.Timestamp("2026-05-31T12:00Z")


# ------------------------------------------------------------------------------------------- scoring
def _tab(n=120, seed=0):
    rng = np.random.default_rng(seed)
    y = np.where(rng.random(n) < 0.4, rng.gamma(1.5, 3, n), 0.0)
    t = pd.DataFrame({"window_start_utc": pd.date_range("2026-03-01T01:00Z", periods=n, freq="D"),
                      "observed_precip_mm": y})
    t["window_end_utc"] = t.window_start_utc + pd.Timedelta(days=1)
    t["champion_median_mm"] = y * 0.9
    t["champion_expected_mm"] = y * 0.95 + 0.1
    t["fc_sr_ecmwf_ifs_mm"] = y + rng.normal(0, 2, n).clip(-y)
    t["fc_pr_jma_gsm_mm"] = y + rng.normal(0, 3, n).clip(-y)
    t["fc_icon_global_mm"] = y + rng.normal(0, 2.5, n).clip(-y)
    for T in scoring.THRESHOLDS:
        t[f"champion_p_ge_{T:g}"] = np.clip((y >= T) * 0.8 + 0.1, 0, 1)
        t[f"clim_p_ge_{T:g}"] = 0.3
    return t


def test_leaderboard_uses_identical_keys_and_reports_coverage_separately():
    t = scoring.add_blend(_tab())
    t.loc[:19, "fc_icon_global_mm"] = np.nan              # challenger missing on 20 days
    cols = scoring.amount_columns(t)
    cov = scoring.coverage(t, cols).set_index("key")
    assert cov.loc["icon_global", "missing"] == 20 and cov.loc["champion_median", "missing"] == 0
    amounts, common = scoring.amount_scores(t, cols)
    assert len(common) == 100 and set(amounts.n_days) == {100}  # every competitor scored on the same 100 days


def test_missing_forecasts_never_become_zero_or_wins():
    t = _tab()
    t["fc_icon_global_mm"] = np.nan
    cols = scoring.amount_columns(t)
    assert "icon_global" not in cols                       # absent everywhere: not ranked at all
    t2 = _tab()
    t2.loc[5, "fc_icon_global_mm"] = np.nan
    _, common = scoring.amount_scores(t2, scoring.amount_columns(t2))
    assert 5 not in common.index


def test_blend_requires_every_member():
    t = _tab()
    t.loc[3, "fc_pr_jma_gsm_mm"] = np.nan
    assert np.isnan(scoring.add_blend(t).loc[3, "fc_blend_mm"])


def test_deterministic_models_have_no_probability_scores():
    t = _tab()
    cols = scoring.amount_columns(t)
    _, common = scoring.amount_scores(t, cols)
    pr = scoring.probability_scores(common, cols)
    raw = pr[pr.key == "icon_global"]
    assert raw.Brier.isna().all() and raw.note.str.contains("unavailable").all()
    assert pr[pr.key == "champion"].Brier.notna().all()


def test_point_forecast_crps_equals_its_mae():
    t = _tab()
    a, _ = scoring.amount_scores(t, scoring.amount_columns(t))
    r = a[a.key == "sr_ecmwf_ifs"].iloc[0]
    assert r.CRPS == r.MAE


def test_holm_adjustment_and_verdicts():
    df = pd.DataFrame({"first": ["A"] * 4, "second": ["B", "C", "D", "E"], "first_key": ["champion_median"] * 4,
                       "second_key": list("bcde"), "n_days": [200, 200, 200, 30], "n_wet_days": [50, 50, 50, 10],
                       "mean_abs_error_diff": [-0.2, 0.1, -0.05, -0.5], "p_bootstrap": [0.001, 0.03, 0.04, 0.0005]})
    h = scoring.holm(df)
    assert (h.p_holm >= h.p_bootstrap).all()
    assert h.loc[0, "verdict"] == "first better"
    assert h.loc[2, "verdict"] == "inconclusive"            # raw p 0.04 < 0.05, but 0.06 after Holm
    assert h.loc[3, "verdict"] == "insufficient evidence"   # tiny sample: no winner however small the p-value


def test_paired_difference_sign():
    t = _tab()
    cols = scoring.amount_columns(t)
    _, common = scoring.amount_scores(t, cols)
    p = scoring.paired(common, cols)
    r = p[(p.first_key == "champion_median") & (p.second_key == "pr_jma_gsm")].iloc[0]
    assert r.mean_abs_error_diff < 0                       # the closer forecast has the negative difference


# ------------------------------------------------------------------------------------------- ledger
def _rec(**kw):
    base = {"issue_id": "20261001T210000Z", "issued_at_utc": "2026-10-01T21:00:00Z", "schedule_slot_utc": "2026-10-01T21:00Z",
            "location": "Perth", "gauge_id": "009225", "gauge_name": "PERTH METRO", "label_date": "2026-10-03",
            "window_start_utc": "2026-10-02 01:00:00+00:00", "window_end_utc": "2026-10-03 01:00:00+00:00",
            "lead": "day1", "competitor": "icon_global", "kind": "challenger", "status": "ok", "total_mm": 1.5}
    return {**base, **kw}


def test_ledger_is_hash_chained_and_tamper_evident(tmp_path):
    p = tmp_path / "ledger.jsonl"
    ledger.append([_rec(), _rec(competitor="ecmwf_aifs025_single", total_mm=2.0)], p)
    ledger.append([_rec(issue_id="20261002T210000Z", total_mm=0.0)], p)
    recs = ledger.read_ledger(p)
    assert len(recs) == 3 and recs[1]["prev_hash"] == recs[0]["record_hash"]
    lines = p.read_text().splitlines()
    lines[0] = lines[0].replace('"total_mm": 1.5', '"total_mm": 9.9')
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ledger.LedgerError):
        ledger.read_ledger(p)
    with pytest.raises(ledger.LedgerError):                # nothing is appended to a broken ledger
        ledger.append([_rec()], p)


def test_deleting_a_line_is_detected(tmp_path):
    p = tmp_path / "ledger.jsonl"
    ledger.append([_rec(total_mm=x) for x in (1.0, 2.0, 3.0)], p)
    lines = p.read_text().splitlines()
    p.write_text("\n".join([lines[0], lines[2]]) + "\n")
    with pytest.raises(ledger.LedgerError):
        ledger.read_ledger(p)


def test_scoring_keeps_each_issue_separate_and_never_overwrites(tmp_path):
    recs = [_rec(competitor="champion", kind="champion", median_mm=0.5, expected_mm=1.0, p10_mm=0, p90_mm=3,
                 probabilities={"ge_0.2": 0.4, "ge_1": 0.2, "ge_5": 0.05, "ge_10": 0.01}),
            _rec(),
            _rec(issue_id="20261002T210000Z", lead="day0x", total_mm=7.0)]
    obs = pd.DataFrame({"station_id": ["009225"], "window_start_utc": ["2026-10-02 01:00:00+00:00"],
                        "window_end_utc": ["2026-10-03 01:00:00+00:00"], "observed_precip_mm": [2.0]})
    tab = ledger.scored_table(recs, obs)
    assert len(tab) == 2                                   # two issues for the same window stay two rows
    first = tab[tab.issue_id == "20261001T210000Z"].iloc[0]
    assert first.fc_icon_global_mm == 1.5 and first.champion_expected_mm == 1.0 and first.observed_precip_mm == 2.0


def test_shared_gauge_is_scored_once():
    recs = [_rec(), _rec(location="Ocean Reef")]
    tab = ledger.scored_table(recs, pd.DataFrame(columns=["station_id", "window_start_utc", "window_end_utc",
                                                          "observed_precip_mm"]))
    assert len(tab) == 1 and tab.location.iloc[0] == "Perth"


def test_schedule_slot():
    now = pd.Timestamp("2026-10-01T22:30Z")
    assert ledger.schedule_slot(now, ["21:00"]) == pd.Timestamp("2026-10-01T21:00Z")
    assert ledger.schedule_slot(pd.Timestamp("2026-10-01T05:00Z"), ["21:00"]) == pd.Timestamp("2026-09-30T21:00Z")


# ------------------------------------------------------------------------------------------- champion
def test_champion_is_unchanged_and_tampering_is_detected(tmp_path, monkeypatch):
    rec = champion.verify()
    assert set(rec["locations"]) == {"perth", "clarkson", "ocean_reef", "fremantle"}
    bad = json.loads(champion.FREEZE.read_text())
    bad["locations"]["perth"]["leads"]["day2"]["sha256"] = "0" * 64
    fake = tmp_path / "freeze.json"
    fake.write_text(json.dumps(bad))
    monkeypatch.setattr(champion, "FREEZE", fake)
    with pytest.raises(champion.ChampionChanged):
        champion.verify()


def test_challenge_code_lives_outside_the_audited_package():
    pkg = ROOT / "src" / "perthrain"
    assert not any("challenge" in p.name for p in pkg.glob("*.py"))
    frozen = json.loads(champion.FREEZE.read_text())
    from perthrain.provenance import code_sha256
    assert all(L["code_sha256"] == code_sha256() for L in frozen["locations"].values())


def test_historical_champion_reproduction_matches_stored_scores():
    p = ROOT / "data" / "challenge" / "historical_contest.parquet"
    assert p.exists(), "run `python -m challenge historical` first"
    t = pd.read_parquet(p)
    for (slug, lead), g in t.groupby(["gauge_slug", "lead"]):
        m = pd.read_csv(ROOT / "reports" / slug / "calibration_metrics.csv")
        s = m[(m.lead == lead) & (m.split == "holdout") & (m.method == "hurdle")].iloc[0]
        assert len(g) == int(s.n)
        assert np.isclose(np.abs(g.champion_median_mm - g.observed_precip_mm).mean(), s.median_MAE_mm, atol=1e-9)


def test_bootstrap_p_value_is_not_floored_at_one_over_b():
    rng = np.random.default_rng(1)
    r = scoring.block_bootstrap(rng.normal(-1.0, 1.0, 300))
    assert r["p"] < 1 / 2000                     # a strong effect can survive Holm over hundreds of comparisons
    assert r["ci95_hi"] < 0
    z = scoring.block_bootstrap(rng.normal(0.0, 1.0, 300))
    assert z["p"] > 0.01 and z["ci95_lo"] < 0 < z["ci95_hi"]
