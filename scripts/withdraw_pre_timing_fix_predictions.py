"""Move the predictions served before the as-of timing fix into a labelled archive (never overwrite), copy the
cached live responses they were computed from, and record a per-ROW status:

* WITHDRAWN_TIMING_FAIL   - at least one input's offset rule names a run issued (or, +6 h, published) after the
                            prediction time; the value cannot be the documented "predicted N days before valid time".
* TIMING_PASS_REISSUED    - all inputs pass the timing rule; the row is re-issued by the corrected run (its value
                            provenance is audited separately; rows with unarchived inputs stay "unverified").
"""
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
ARCH = ROOT / "archive" / f"WITHDRAWN_{STAMP}_pre_timing_fix_predictions"
if ARCH.exists():
    raise SystemExit(f"{ARCH} exists - refusing to overwrite")

audit = pd.read_csv(ROOT / "reports" / "live_lead_audit_rows.csv")
row_ok = audit.groupby(["location", "gauge_day_label", "lead"]).latest_run_issued_and_published_before_prediction.all()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


recs, rows = [], []
for loc_dir in sorted((ROOT / "data").glob("*/predictions")):
    loc = loc_dir.parent.name
    for f in sorted(loc_dir.iterdir()):
        dst = ARCH / "predictions" / loc / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        h = sha(f)
        shutil.move(str(f), str(dst))
        assert sha(dst) == h and not f.exists()
        recs.append({"action": "moved", "original": f.relative_to(ROOT).as_posix(),
                     "archived": dst.relative_to(ROOT).as_posix(), "sha256": h})
        if f.suffix == ".csv" and f.name.startswith("forecast_"):
            d = pd.read_csv(dst)
            for _, r in d.iterrows():
                if r.status != "ok":
                    st = "NOT_SERVED"
                else:
                    st = "TIMING_PASS_REISSUED" if row_ok.get((loc, r.label_date_local, r.lead_group), False) \
                        else "WITHDRAWN_TIMING_FAIL"
                rows.append({"location": loc, "file": f.name, "gauge_day_label": r.label_date_local,
                             "lead": r.lead_group, "model": r.feature_set_used, "route": r.route, "row_status": st})
live = ROOT / "data" / "raw" / "open_meteo" / "live"
dst = ARCH / "inputs" / "open_meteo_live_cache"
shutil.copytree(live, dst)
for f in sorted(dst.rglob("*")):
    if f.is_file():
        recs.append({"action": "copied", "original": (live / f.relative_to(dst)).relative_to(ROOT).as_posix(),
                     "archived": f.relative_to(ROOT).as_posix(), "sha256": sha(f)})
with open(ARCH / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["action", "original", "archived", "sha256"])
    w.writeheader()
    w.writerows(recs)
pd.DataFrame(rows).to_csv(ARCH / "row_status.csv", index=False)
counts = pd.DataFrame(rows).groupby("row_status").size().to_dict()
(ARCH / "README.md").write_text(
    "# Predictions withdrawn pending correction: served before the as-of timing fix\n\n"
    f"Moved here {STAMP} (never overwritten; hashes in manifest.csv). `row_status.csv` gives each row's status.\n\n"
    "* `WITHDRAWN_TIMING_FAIL`: an input's offset rule names a model run issued after the prediction time (or less "
    "than 6 h before it), so the value cannot be the documented forecast 'predicted N days before valid time' that "
    "the day-N calibrator was trained on. Withdrawn.\n"
    "* `TIMING_PASS_REISSUED`: all inputs pass the timing rule; re-issued by the corrected code. Value provenance is "
    "audited separately (reports/live_provenance_audit.md); rows with inputs whose named run is not archived remain "
    "'unverified'.\n* `NOT_SERVED`: already not served.\n\n"
    "`inputs/open_meteo_live_cache/` preserves the cached live API responses these predictions were computed from.\n\n"
    f"Row counts: {json.dumps(counts)}\n", encoding="utf-8")
print(ARCH)
print(counts)
