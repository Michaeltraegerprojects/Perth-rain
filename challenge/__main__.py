"""Forecast Challenge command line.

    python -m challenge freeze              record the champion (artifacts, selection rules, schemas)
    python -m challenge verify              check the champion is unchanged (hashes and versions only)
    python -m challenge refreeze --reason "..."   new record after a reviewed code/config/runtime change (artifacts
                                            must be unchanged and the held-out reproduction must still be exact)
    python -m challenge checkpoint          anchor the current ledger (count + head hash) without changing it
    python -m challenge probe               small availability probes of challenger sources
    python -m challenge verify-semantics    check previous_dayN values against the rule-named runs
    python -m challenge historical          build the historical contest table (champion's original holdout)
    python -m challenge collect [--force]   append today's forecasts (champion, raw, challengers) to the ledger
    python -m challenge score               score historical + completed prospective windows; write reports
    python -m challenge export              write the dashboard data (website/data/pending/challenge.json)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m challenge")
    ap.add_argument("command", choices=["freeze", "verify", "refreeze", "checkpoint", "probe", "verify-semantics",
                                        "historical", "collect", "score", "export"])
    ap.add_argument("--reason", default="", help="why the champion record is being renewed (refreeze)")
    ap.add_argument("--force", action="store_true", help="collect outside the scheduled window (time is recorded)")
    a = ap.parse_args(argv)
    from challenge import champion
    if a.command == "freeze":
        print(champion.freeze())
    elif a.command == "verify":
        champion.verify()
        print("champion unchanged")
    elif a.command == "refreeze":
        from challenge import historical
        # behavioural evidence first: every gauge x lead must still reproduce the audited held-out evaluation
        rec = json.loads(champion.FREEZE.read_text())
        for g in historical.unique_gauges():
            for lead in ("day1", "day2", "day3"):
                historical.champion_holdout(g["slug"], lead, rec)
        print(champion.refreeze(a.reason))
    elif a.command == "checkpoint":
        from challenge import ledger
        print(ledger.ensure_checkpoint())
    elif a.command == "probe":
        from challenge import probe_sources
        probe_sources.main()
    elif a.command == "verify-semantics":
        from challenge import verify_semantics
        verify_semantics.main()
    elif a.command == "historical":
        from challenge import historical
        historical.build()
    elif a.command == "collect":
        from challenge import ledger
        print(f"appended {ledger.collect(force=a.force)} records to {ledger.LEDGER.relative_to(ROOT)}")
    elif a.command == "score":
        score()
    elif a.command == "export":
        from challenge import export
        print(export.export())


def score():
    from challenge import champion, ledger, report
    champion.verify()
    hist = ROOT / "data" / "challenge" / "historical_contest.parquet"
    if hist.exists():
        tab = pd.read_parquet(hist)
        tab["window_start_utc"] = pd.to_datetime(tab.window_start_utc, utc=True)
        rep = report.build(tab, "historical (champion's original held-out days; exploratory for challengers)")
        report.write(rep, "historical")
        print(f"historical: {sum(g['common_days'] for g in rep['groups'])} common gauge-days across "
              f"{len(rep['groups'])} gauge x lead groups; {rep['n_comparisons']} paired comparisons")
    recs = ledger.read_ledger()
    if recs:
        obs = ledger.refresh_observations()
        tab = ledger.scored_table(recs, obs)
        tab.to_csv(ROOT / "data" / "challenge" / "prospective_scored.csv", index=False)
        done = tab[tab.observed_precip_mm.notna()]
        rep = report.build(done, "prospective (forecasts as issued, scored after the gauge reading arrived)",
                           key_cols=("window_start_utc", "issue_id"))
        rep["windows_issued"] = int(len(tab))
        rep["windows_scored"] = int(len(done))
        rep["ledger_records"] = len(recs)
        report.write(rep, "prospective")
        print(f"prospective: {len(recs)} ledger records; {len(done)} of {len(tab)} issued windows have observations")


if __name__ == "__main__":
    main()
