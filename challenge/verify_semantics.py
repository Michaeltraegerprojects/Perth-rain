"""Verify, for each challenger, that Previous Runs `precipitation_previous_dayN` values equal the run named by the
offset rule  run = floor_6h(ceil_to_native_step(valid_end)) - 24*N h  (compared on NON-ZERO hours only, against
the Single Runs archive). Writes reports/challenge/semantics_verification.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from challenge.sources import NATIVE_STEP, fetch_previous_runs, fetch_single_run, rule_run_time  # noqa: E402

LAT, LON = -31.9192, 115.8728
TOL = 0.051


def main(models=("icon_global", "ecmwf_aifs025_single"), start="2026-05-01", end="2026-09-28"):
    rows = []
    for model in models:
        for n in (1, 2, 3):
            pr = fetch_previous_runs(model, LAT, LON, pd.Timestamp(start), pd.Timestamp(end), [n])
            s = pr[(pr.lead_day == n) & (pr.precip_mm > 0.05)].sort_values("valid_end_utc")
            if len(s) > 60:
                s = s.iloc[np.linspace(0, len(s) - 1, 60).astype(int)]
            tested = matched = unarchived = 0
            for _, r in s.iterrows():
                run = rule_run_time(r.valid_end_utc, NATIVE_STEP[model], n)
                sr = fetch_single_run(model, LAT, LON, run, 5)
                if sr is None:
                    unarchived += 1
                    continue
                v = sr.set_index("valid_end_utc").precip_mm.get(r.valid_end_utc)
                if v is None or pd.isna(v):
                    unarchived += 1
                    continue
                tested += 1
                matched += bool(np.isclose(v, r.precip_mm, atol=TOL))
            rows.append({"model": model, "lead_day": n, "nonzero_hours_sampled": len(s), "compared": tested,
                         "matched": matched, "run_not_archived": unarchived,
                         "match_rate": round(matched / tested, 4) if tested else None})
            print(rows[-1], flush=True)
    out = ROOT / "reports" / "challenge"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "semantics_verification.csv", index=False)


if __name__ == "__main__":
    main()
