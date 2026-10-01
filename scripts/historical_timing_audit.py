"""DIAGNOSTIC: the as-of timing rule applied to the HISTORICAL paired rows used for training and held-out scoring.
For every training-eligible row: (a) the latest contributing run (explicit or offset rule) + 6 h publication latency
must be <= the window start (the as-of time of a forecast used for that window), and (b) that run must have been
issued before the data were retrieved. Also lists single-run fetch failures of the provenance audit."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rows = []
for loc in ("perth", "clarkson", "ocean_reef", "fremantle"):
    long = pd.read_parquet(ROOT / "data" / loc / "joined" / "paired_long.parquet")
    e = long[long.training_eligible]
    issue = e.forecast_issue_time_utc.where(e.forecast_issue_time_utc.notna(), e.issue_time_inferred_max_utc)
    avail_margin_h = (e.window_start_utc - (issue + pd.Timedelta(hours=6))) / pd.Timedelta(hours=1)
    retrieved = pd.to_datetime(e.retrieved_at_utc, utc=True)
    for (src, lead), g in e.assign(avail_margin_h=avail_margin_h, issue=issue, retrieved=retrieved).groupby(
            ["source_api", "lead_group"]):
        rows.append({"location": loc, "source_api": src, "lead": lead, "eligible_rows": len(g),
                     "min_hours_published_before_window_start": round(g.avail_margin_h.min(), 1),
                     "rows_failing_as_of_rule": int((g.avail_margin_h < 0).sum()),
                     "rows_run_after_retrieval": int((g.issue >= g.retrieved).sum()),
                     "first_window": str(g.label_date_local.min()), "last_window": str(g.label_date_local.max())})
t = pd.DataFrame(rows)
t.to_csv(ROOT / "reports" / "historical_timing_audit.csv", index=False)
pd.set_option("display.width", 220)
print(t.to_string(index=False))
print("\ntotal eligible rows failing the as-of rule:", int(t.rows_failing_as_of_rule.sum()),
      "| with run issued after retrieval:", int(t.rows_run_after_retrieval.sum()))
print("\nsingle-run fetch failures during the provenance audit:")
for line in (ROOT / "data" / "raw" / "failures.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    if str(r.get("label", "")).startswith("audit "):
        print(" ", r["label"], "->", str(r.get("error"))[:110])
