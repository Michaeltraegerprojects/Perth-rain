"""Second review of branch fix/audit-0de31ee (2026-10-04): defects in the audit fixes themselves, plus recovery checks.

R-1  refreeze silently recreated a deleted per-row reference, so the row check could be reset by the very operation
     it guards (a reviewed code change).
R-2  a challenger was labelled "observed" when the model metadata could not be read.
R-6  a single-day (or constant-difference) comparison reported p = 0.
R-3, R-4 and R-5 (timing gate order, offline replay, cached-400 expiry) are in tests/test_branch_review_timing.py.
Each R test was written to fail on 58d0730 before the fix."""
import json
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest

from challenge import champion, export, historical, ledger, scoring, sources
from challenge.__main__ import main
from perthrain.predict import ESTIMATED

LEADS = ("day1", "day2", "day3")


# ------------------------------------------------------------------------- R-1: per-row references
def _freeze_with_references(monkeypatch):
    monkeypatch.setattr(historical, "unique_gauges", lambda: [{"slug": "perth"}])
    champion.freeze()
    rec = json.loads(champion.FREEZE.read_text())
    for lead in LEADS:
        historical.champion_holdout("perth", lead, rec, create_reference=True)
    champion.refreeze("record the per-row references for the test")
    return champion.verify()


def test_historical_reproduction_never_creates_a_reference_implicitly(synthetic_project):
    rec = json.loads(champion.freeze().read_text())
    with pytest.raises(champion.ChampionChanged, match="reference"):
        historical.champion_holdout("perth", "day1", rec)
    assert not historical.reference_path("perth", "day1").exists()


def test_refreeze_refuses_a_deleted_reference_instead_of_recreating_it(synthetic_project, monkeypatch):
    rec = _freeze_with_references(monkeypatch)
    before = champion.FREEZE.read_bytes()
    ref = historical.reference_path("perth", "day1")
    assert ref.name in rec["holdout_references"]
    ref.unlink()
    with pytest.raises(champion.ChampionChanged, match="reference"):
        main(["refreeze", "--reason", "a reviewed change to the inference code"])
    assert not ref.exists(), "a recorded reference must never be recreated from the current code"
    assert champion.FREEZE.read_bytes() == before


def test_refreeze_refuses_a_changed_reference(synthetic_project, monkeypatch):
    _freeze_with_references(monkeypatch)
    ref = historical.reference_path("perth", "day2")
    r = pd.read_parquet(ref)
    r.loc[r.index[0], "champion_median_mm"] += 0.01
    r.to_parquet(ref, index=False)
    with pytest.raises(champion.ChampionChanged, match="reference"):
        champion.refreeze("a reviewed change to the inference code")


def test_freeze_command_records_references_for_a_new_champion(synthetic_project, monkeypatch):
    monkeypatch.setattr(historical, "unique_gauges", lambda: [{"slug": "perth"}])
    main(["freeze"])
    rec = champion.verify()
    assert sorted(rec["holdout_references"]) == [f"holdout_perth_{lead}.parquet" for lead in LEADS]


# ------------------------------------------------------------------------- R-2: challenger availability label
W = namedtuple("W", "window_start_utc window_end_utc")
NOW = pd.Timestamp("2026-10-04T21:00:00Z")
RUN = pd.Timestamp("2026-10-04T00:00:00Z")          # the run the offset rule names for the day-2 window


def _challenger(monkeypatch, meta):
    row = pd.DataFrame([{"total_mm": 1.2, "latest_run_utc": RUN, "published_before_cutoff": True,
                         "lead_hours_to_window_start": 25.0, "grid_latitude": -31.9, "grid_longitude": 115.9,
                         "retrieved_at_utc": "2026-10-04T21:00:05Z", "source_url": "u"}])
    monkeypatch.setattr(sources, "fetch_previous_runs", lambda *a, **k: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(sources, "window_totals_previous_runs", lambda *a, **k: row)
    monkeypatch.setattr(sources, "fetch_single_run", lambda *a, **k: None)
    w = W(pd.Timestamp("2026-10-06T01:00Z"), pd.Timestamp("2026-10-07T01:00Z"))
    metas = {m: meta for m in sources.CHALLENGERS}
    recs = ledger._challenger_records({"issue_id": "x", "issued_at_utc": f"{NOW:%Y-%m-%dT%H:%M:%SZ}"},
                                      {"latitude": -31.9, "longitude": 115.9}, w, 2, NOW, metas)
    return next(r for r in recs if r["competitor"] == "icon_global")


def test_challenger_without_model_metadata_is_labelled_estimated(monkeypatch):
    r = _challenger(monkeypatch, {"last_run_init_utc": None, "last_run_available_utc": None, "retrieved_at_utc": None})
    assert r["status"] == "ok"
    assert r["availability_basis"] == ESTIMATED
    assert "unavailable" in r["availability_evidence"]
    assert export.competitor_timing(r)["availability_basis"] == ESTIMATED


def test_old_records_naming_missing_metadata_are_not_shown_as_observed():
    r = {"competitor": "icon_global", "issued_at_utc": "2026-10-01T21:00:00Z", "latest_run_utc": "2026-10-01 06:00:00+00:00",
         "availability_evidence": "run named by the offset rule; model metadata last run None (checked None)"}
    assert export.competitor_timing(r)["availability_basis"] == ESTIMATED


def test_challenger_run_published_after_the_issue_time_is_missing(monkeypatch):
    r = _challenger(monkeypatch, {"last_run_init_utc": RUN, "last_run_available_utc": NOW + pd.Timedelta(minutes=10),
                                  "retrieved_at_utc": "2026-10-04T21:10:30Z"})
    assert r["status"] == "missing" and "published" in r["missing_reason"]


def test_challenger_with_metadata_proving_availability_is_observed(monkeypatch):
    r = _challenger(monkeypatch, {"last_run_init_utc": RUN, "last_run_available_utc": NOW - pd.Timedelta(hours=11),
                                  "retrieved_at_utc": "2026-10-04T20:59:50Z"})
    assert r["status"] == "ok" and r["availability_basis"] == "observed"


# ------------------------------------------------------------------------- R-6: degenerate p-values
def test_a_single_day_is_never_significant():
    assert scoring.block_bootstrap(np.array([0.5]))["p"] == 1.0


def test_a_constant_difference_uses_the_exact_sign_test():
    assert scoring.block_bootstrap(np.full(3, 0.5))["p"] == pytest.approx(0.25)
    assert scoring.block_bootstrap(np.full(10, -0.2))["p"] == pytest.approx(2 * 0.5 ** 10)   # not exactly representable


def test_fewer_days_than_one_block_is_not_significant():
    # with n < 7 every circular block resample is the whole series, so the bootstrap SE is 0 whatever the data
    assert scoring.block_bootstrap(np.array([0.5, -0.2, 0.3]))["p"] == 1.0


# ------------------------------------------------------------------------- recovery checks (pass before and after)
def _rec(i):
    return {"issue_id": f"2026100{i}T210000Z", "location": "Perth", "competitor": "x", "status": "ok", "total_mm": float(i)}


def test_crash_between_ledger_write_and_checkpoint_recovers_on_the_next_append(tmp_path):
    p = tmp_path / "ledger.jsonl"
    ledger.append([_rec(0)], p)
    prev = ledger.read_ledger(p)[-1]["record_hash"]
    r = dict(_rec(1), prev_hash=prev)
    r["record_hash"] = ledger._hash(r)
    with open(p, "a", encoding="utf-8") as fh:            # record written, process killed before the checkpoint
        fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    assert len(ledger.read_ledger(p)) == 2                # a ledger longer than its checkpoint is valid
    ledger.append([_rec(2)], p)
    assert ledger._read_checkpoints(p)[-1]["count"] == 3  # the checkpoint catches up


def test_stale_lock_is_refused_with_instructions_and_removal_restores_appends(tmp_path):
    p = tmp_path / "ledger.jsonl"
    ledger.append([_rec(0)], p)
    ledger.lock_path(p).write_text("pid 1 since 2026-10-04T21:00:00Z", encoding="utf-8")   # writer was killed
    with pytest.raises(ledger.LedgerError, match="stale"):
        ledger.append([_rec(1)], p)
    ledger.lock_path(p).unlink()
    ledger.append([_rec(1)], p)
    assert len(ledger.read_ledger(p)) == 2


# ------------------------------------------------------------------------- R-7: records loaded from the real ledger
NAN = float("nan")


def test_timing_handles_records_without_availability_fields():
    """Found on the first export of real data after PR #1: pandas loads absent fields as NaN, which is truthy and
    not iterable, so competitor_timing crashed (unit tests had only used plain dicts)."""
    issued = "2026-10-01T21:00:00Z"
    old_raw = {"competitor": "pr_jma_gsm", "issued_at_utc": issued, "latest_run_utc": "2026-10-01 06:00:00+00:00",
               "availability_evidence": NAN, "availability_basis": NAN}
    t = export.competitor_timing(old_raw)
    assert t["availability_basis"] == ESTIMATED and t["info_age_h"] == 15.0
    old_champ = {"competitor": "champion", "issued_at_utc": issued, "availability_basis": NAN, "availability_evidence": NAN,
                 "inputs": "pr_jma_gsm: runs up to 2026-10-01 06Z (inferred)", "latest_run_utc": NAN}
    assert export.competitor_timing(old_champ)["availability_basis"] == ESTIMATED
    no_run = {"competitor": "icon_global", "issued_at_utc": issued, "latest_run_utc": NAN, "named_run_utc": None,
              "availability_evidence": NAN, "availability_basis": None}
    assert export.competitor_timing(no_run)["info_age_h"] is None


def test_timing_works_on_rows_built_the_way_live_view_builds_them():
    df = pd.DataFrame([{"competitor": "pr_jma_gsm", "issued_at_utc": "2026-10-01T21:00:00Z", "status": "ok",
                        "latest_run_utc": "2026-10-01 06:00:00+00:00"},
                       {"competitor": "icon_global", "issued_at_utc": "2026-10-04T21:00:00Z", "status": "ok",
                        "latest_run_utc": "2026-10-04 00:00:00+00:00", "availability_basis": "observed",
                        "availability_evidence": "explicit run requested; HTTP 200 at issue time"}])
    out = [export.competitor_timing(r._asdict()) for r in df.itertuples(index=False)]
    assert [o["availability_basis"] for o in out] == [ESTIMATED, "observed"]
    json.dumps(out, allow_nan=False)                      # the export refuses NaN
