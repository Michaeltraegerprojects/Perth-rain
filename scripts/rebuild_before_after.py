"""Frozen-input before/after comparison of the withdrawn (pre-audit) predictions and the rebuilt no-ENSO models.

This measures a SOFTWARE CHANGE on identical inputs. It is NOT evidence of forecast skill: neither set of numbers
is compared with what the gauge later measured. Skill is assessed separately, on held-out days, in
reports/<location>/calibration_report.md.

Procedure, per location with withdrawn predictions:
1. Re-run the OLD artifacts (loaded explicitly from the invalidated archive, for this forensic comparison only -
   never through the serving loader, which refuses them) on the cached live inputs at the original issue time,
   using the pre-audit climate join. The result must reproduce the archived prediction file exactly; this proves
   the cached inputs are the ones the withdrawn predictions used.
2. Run the NEW active artifacts through the serving loader on the same cached inputs and time.
3. Compare row by row.

Materiality rule, fixed before any comparison was run:
    a row's difference is MATERIAL if any of |dP(>=0.2 mm)|, |dP(>=1 mm)|, |dP(>=5 mm)|, |dP(>=10 mm)| >= 0.05,
    or |d median total| >= 0.5 mm, or |d expected total| >= 0.5 mm, or the served/not-served status changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perthrain import climate                     # noqa: E402
from perthrain import predict as P                # noqa: E402
from perthrain.config import load_configs         # noqa: E402
from perthrain.pipeline import make_fetcher       # noqa: E402

ARCH = sorted(ROOT.glob("archive/INVALIDATED_*_enso_contaminated"))[-1]
PROB = ["p_rain_ge_0_2mm", "p_ge_1mm", "p_ge_5mm", "p_ge_10mm"]
AMT = ["q50_mm", "mean_estimate_mm"]


def material(r) -> bool:
    if r.status_old != r.status_new:
        return True
    if r.status_old != "ok":
        return False
    return (any(abs(r[f"{c}_new"] - r[f"{c}_old"]) >= 0.05 for c in PROB)
            or any(abs(r[f"{c}_new"] - r[f"{c}_old"]) >= 0.5 for c in AMT))


def main() -> None:
    import os
    os.chdir(ROOT)
    cfgs = {c.data_dir.name: c for c in load_configs("config.toml")}
    out_rows, notes = [], []
    for loc in ("perth", "ocean_reef", "clarkson"):
        cfg = cfgs[loc]
        recs = sorted((ARCH / "live_predictions" / loc).glob("forecast_*.csv"))
        old_file = recs[-1]
        stamp = old_file.stem.split("_")[1]
        now = pd.Timestamp(pd.to_datetime(stamp, format="%Y%m%dT%H%MZ"), tz="UTC")
        archived = pd.read_csv(old_file)
        # ---- 1. reproduce the withdrawn predictions from the quarantined artifacts (forensic, explicit)
        old_b = {p.stem.split("_")[1]: joblib.load(p) for p in sorted((ARCH / "model_artifacts" / loc).glob("hurdle_*.joblib"))}
        st = P.station_info(cfg)
        windows = P.future_windows(now, cfg.timezone, 4)
        sr, pr, probs = P._fetch_live(cfg, make_fetcher(cfg), st["latitude"], st["longitude"], windows, now, refresh=False)
        frozen_misses = [p for p in probs if p.startswith("frozen replay")]
        # The withdrawn predictions were made by code WITHOUT the as-of timing gate (fixed 2026-09-30 ~15:40Z).
        # To reproduce them forensically, the gate is disabled here by an as-of time far in the future. Do not reuse.
        fw = P.live_feature_table(cfg, sr, pr, windows, pd.Timestamp("2262-01-01", tz="UTC"))
        indices = climate.load_indices(make_fetcher(cfg), cfg.raw_dir)
        clim = P.live_climate(indices, windows)
        old = P.predict_windows(old_b, fw, windows, cfg.timezone, clim, experimental=True)
        a = archived.set_index(["label_date_local", "lead_group"])
        o = old.assign(label_date_local=old.label_date_local.astype(str)).set_index(["label_date_local", "lead_group"])
        ok = a.status == "ok"
        repro = (a.index.equals(o.index) and (a.status == o.status).all()
                 and np.allclose(a.loc[ok, PROB + AMT].to_numpy(float), o.loc[ok, PROB + AMT].to_numpy(float), atol=1e-9))
        notes.append(f"{loc}: withdrawn file {old_file.name} (issued {now:%Y-%m-%d %H:%M}Z); re-running the "
                     f"quarantined artifacts on the cached inputs reproduces it exactly: **{repro}**"
                     + (f"; cache misses: {frozen_misses}" if frozen_misses else ""))
        # ---- 2. the new active artifacts on the same inputs
        new = P.predict_windows(P.load_bundles(cfg), fw, windows, cfg.timezone)
        n = new.assign(label_date_local=new.label_date_local.astype(str))
        m = archived.merge(n, on=["label_date_local", "lead_group"], suffixes=("_old", "_new"))
        for _, r in m.iterrows():
            row = {"location": loc, "gauge_day": r.label_date_local, "lead": r.lead_group,
                   "old_model": r.feature_set_used_old, "old_climate": r.get("climate_used", r.get("climate_used_old", "")),
                   "new_model": r.feature_set_used_new,
                   "new_route": r["route_new"] if "route_new" in r else r.get("route"),   # old files had no route
                   "status_old": r.status_old, "status_new": r.status_new}
            for c in PROB + AMT:
                row[f"{c}_old"], row[f"{c}_new"] = r.get(f"{c}_old", np.nan), r.get(f"{c}_new", np.nan)
                row[f"d_{c}"] = row[f"{c}_new"] - row[f"{c}_old"]
            out_rows.append(row)
    df = pd.DataFrame(out_rows)
    df["material"] = df.apply(material, axis=1)
    df.to_csv(ROOT / "reports" / "rebuild_before_after.csv", index=False)
    show = df[["location", "gauge_day", "lead", "old_model", "old_climate", "new_model", "new_route",
               "p_rain_ge_0_2mm_old", "p_rain_ge_0_2mm_new", "q50_mm_old", "q50_mm_new",
               "mean_estimate_mm_old", "mean_estimate_mm_new", "material"]].copy()
    for c in show.columns:
        if show[c].dtype == float:
            show[c] = show[c].round(3)
    txt = ["# Before/after on frozen inputs - software change only, NOT a skill comparison\n",
           __doc__.split("Materiality rule")[0].strip().replace("\n", " ") + "\n",
           "**Materiality rule (fixed before running):** any of |dP(>=0.2)|, |dP(>=1)|, |dP(>=5)|, |dP(>=10)| >= 0.05, "
           "or |d median| >= 0.5 mm, or |d expected total| >= 0.5 mm, or a change in served status.\n",
           "## Input check\n", *[f"* {x}" for x in notes],
           "\n## Row-by-row (old = withdrawn pending verification; new = rebuilt no-ENSO)\n",
           "| " + " | ".join(show.columns) + " |", "|" + "---|" * len(show.columns)]
    for _, r in show.iterrows():
        txt.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    mat = df[df.material]
    txt.append(f"\n**Material differences: {len(mat)} of {len(df)} rows.**" +
               ("" if mat.empty else " Rows: " + "; ".join(f"{r.location} {r.gauge_day} {r.lead}" for _, r in mat.iterrows())))
    (ROOT / "reports" / "rebuild_before_after.md").write_text("\n".join(txt) + "\n", encoding="utf-8")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
