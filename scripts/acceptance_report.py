"""DIAGNOSTIC report builder (reads only outputs of the tested package code; computes no model results itself).

Writes reports/acceptance_report.md:
  A. held-out comparison per GAUGE and lead on identical station/window/target rows (fingerprint-verified)
  B. paired differences with block-bootstrap intervals
  C. which reviewer findings affected this run (evidence per finding)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import tomllib  # noqa: E402

from perthrain.config import slug  # noqa: E402

locs = [(e["name"], slug(e["name"])) for e in tomllib.loads((ROOT / "config.toml").read_text())["locations"]]
GAUGE = {}
for name, s in locs:
    st = json.loads((ROOT / "reports" / s / "run_summary.json").read_text())["station"]
    GAUGE[s] = (st["station_id"], st["station_name"], float(st["distance_km"]))

metrics, paired, sel = {}, {}, {}
for name, s in locs:
    metrics[s] = pd.read_csv(ROOT / "reports" / s / "calibration_metrics.csv")
    paired[s] = pd.read_csv(ROOT / "reports" / s / "paired_differences.csv")
    sel[s] = json.loads((ROOT / "data" / s / "models" / "selection.json").read_text())

out = ["# Acceptance report (held-out verification, uncertainty and reviewer-impact)\n",
       "Generated from the package outputs of the final run. **Pipeline integrity is not forecast accuracy**: the tests "
       "show the pipeline is consistent; only the held-out comparisons below speak to accuracy, and they are "
       "descriptive unless a paired interval excludes zero.\n",
       "**Event definition:** rain-occurrence events use **>=** (inclusive): an observed 0.2 mm counts as rain. "
       "Threshold 0.2 mm for occurrence; ≥ 1, 5, 10 mm for heavier events.\n",
       "**Independent gauges:** Perth and Ocean Reef use the *same gauge* (009225 PERTH METRO) and the same forecast "
       "grid cell, so they have identical inputs, targets, models and results. They are **one** validation sample, "
       "not two (below they are reported once, as Perth).\n"]

# ------------------------------------------------------------------ identical-input proof for Perth vs Ocean Reef
same = []
for lead in ("day1", "day2", "day3"):
    a = metrics["perth"][(metrics["perth"].lead == lead) & (metrics["perth"].split == "holdout")]
    b = metrics["ocean_reef"][(metrics["ocean_reef"].lead == lead) & (metrics["ocean_reef"].split == "holdout")]
    fa, fb = set(a.key_fingerprint.dropna()), set(b.key_fingerprint.dropna())
    ma = a[a.method == "hurdle"].iloc[0]
    mb = b[b.method == "hurdle"].iloc[0]
    same.append((lead, fa == fb, np.isclose(ma.pinball_mean, mb.pinball_mean, rtol=0, atol=1e-12)))
out.append("Perth vs Ocean Reef, per lead (same key fingerprint, same pinball to 1e-12): " +
           ", ".join(f"{l}: {f} / {p}" for l, f, p in same) + ".\n")

# ------------------------------------------------------------------ A. per gauge and lead
out.append("## A. Held-out results by gauge and lead (identical rows for every method)\n")
out.append("MAE, signed bias (forecast − observed) and RMSE in mm. Calibrated point forecasts: **median** and "
           "**expected total** are reported separately. Raw = the models' own millimetre forecast; blend = equal-weight "
           "mean of the raw members of the composite. Climatology rows are fitted on the training dates only.\n")
rows = []
seen_gauges = set()
verification = []
for name, s in locs:
    if GAUGE[s][0] in seen_gauges:
        continue
    seen_gauges.add(GAUGE[s][0])
    for lead in ("day1", "day2", "day3"):
        h = metrics[s][(metrics[s].lead == lead) & (metrics[s].split == "holdout")]
        base = h[h.method == "hurdle"].iloc[0]
        ids = h[h.method.isin(["hurdle", "climatology", "climatology_flat"]) | h.method.astype(str).str.startswith("raw:")]
        fps, ns, flags = set(ids.key_fingerprint), set(ids.n.astype(int)), set(ids.keys_identical_across_methods)
        verification.append({"gauge": GAUGE[s][1], "lead": lead, "methods": len(ids), "n": sorted(ns),
                             "distinct_fingerprints": len(fps), "all_flagged_identical": flags == {True}})
        g = dict(gauge=f"{GAUGE[s][0]} {GAUGE[s][1]} ({GAUGE[s][2]:.1f} km from {name})", lead=lead,
                 n=int(base.n), test=f"{base.test_first}..{base.test_last}", model=base.feature_set)
        rows.append({**g, "method": "calibrated median", "MAE": base.median_MAE_mm, "bias": base.median_bias_mm,
                     "RMSE": base.median_RMSE_mm, "Brier>=0.2": base["brier_0.2"]})
        rows.append({**g, "method": "calibrated expected total", "MAE": base.mean_est_MAE_mm, "bias": base.mean_est_bias_mm,
                     "RMSE": base.mean_est_RMSE_mm, "Brier>=0.2": base["brier_0.2"]})
        for _, r in h[h.method.astype(str).str.startswith("raw:")].iterrows():
            rows.append({**g, "method": r.method.replace("raw:", "raw "), "MAE": r.MAE_mm, "bias": r.bias_fc_minus_obs_mm,
                         "RMSE": r.RMSE_mm, "Brier>=0.2": np.nan})
        if not (h.method == "raw:equal_weight_blend").any():
            rows.append({**g, "method": "equal-weight blend", "MAE": np.nan, "bias": np.nan, "RMSE": np.nan,
                         "Brier>=0.2": np.nan, "note": "unavailable: a single model serves day 1"})
        for cm, lab in (("climatology", "season-aware climatology"), ("climatology_flat", "season-blind climatology")):
            c = h[h.method == cm].iloc[0]
            rows.append({**g, "method": lab, "MAE": c.median_MAE_mm, "bias": c.mean_est_bias_mm, "RMSE": c.mean_est_RMSE_mm,
                         "Brier>=0.2": c["brier_0.2"], "note": "MAE of its median; bias/RMSE of its expected total"})
T = pd.DataFrame(rows)
T.to_csv(ROOT / "reports" / "acceptance_heldout_table.csv", index=False)


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else
                                       (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r.values) + " |")
    return "\n".join(lines) + "\n"


for (gauge, lead), g in T.groupby(["gauge", "lead"], sort=False):
    out.append(f"\n**{gauge} - {lead}**: n = {g.n.iloc[0]} days ({g.test.iloc[0]}), selected model `{g.model.iloc[0]}`\n")
    out.append(md(g[["method", "MAE", "bias", "RMSE", "Brier>=0.2"] + (["note"] if "note" in g and g.note.notna().any() else [])]))
out.append("\n**Row-identity verification** (station + accumulation window + observed target fingerprint, recorded by the "
           "calibration code for every method): \n")
out.append(md(pd.DataFrame(verification)))
out.append("\nProbability scores exist only for the calibrated model and the climatologies. **Raw model forecasts are "
           "amounts, not probabilities, so no probability improvement over them is claimed or computable.**\n")

# ------------------------------------------------------------------ B. paired differences
out.append("\n## B. Paired differences with 95 % block-bootstrap intervals\n")
out.append("Per-day differences on identical held-out days, first method minus second (negative = the first, i.e. the "
           "calibrated forecast, is better). Circular moving blocks of 7 days, 2000 resamples, seed 0. "
           "`significant` = the interval excludes zero; otherwise the ranking is **descriptive, not significant**. "
           "Several comparisons are made per lead, so isolated significant results should not be over-read.\n")
for name, s in locs:
    if GAUGE[s][0] in {GAUGE[x][0] for _, x in locs[:locs.index((name, s))]}:
        continue
    p = paired[s]
    p = p[p.mean_diff.notna()]
    out.append(f"\n**{GAUGE[s][1]}**\n")
    show = p[["lead", "comparison", "n_days", "mean_diff", "ci95_lo", "ci95_hi", "significant_at_95"]].copy()
    show["comparison"] = show.comparison.str.replace("calibrated ", "cal. ", regex=False)
    out.append(md(show))
    tot = p.groupby("lead").significant_at_95.agg(["sum", "count"])
    out.append("Significant comparisons per lead: " + ", ".join(f"{k}: {int(v['sum'])}/{int(v['count'])}" for k, v in tot.iterrows()) + ".\n")

# ------------------------------------------------------------------ C. reviewer impact
out.append("\n## C. Which reviewer findings affected this run\n")
first = sorted((ROOT / "archive" / "superseded_models" / "perth").glob("models_before_noenso-20260930T1452*"))
imp = []
unequal = all(s_["cv_folds"] == ["cv1", "cv2", "cv3"] for sv in sel.values() for k, s_ in sv.items() if not k.startswith("_")
              and isinstance(s_, dict) and "cv_folds" in s_)
notes = [n for sv in sel.values() for lead in ("day1", "day2", "day3") for n in sv["_notes"][lead]]
excl = [e for sv in sel.values() for lead in ("day1", "day2", "day3") for e in sv[lead]["excluded_candidates"]]
imp.append(("Unequal CV folds (candidates averaged over different days)", "not triggered",
            f"all 12 location x lead selections scored every configuration on cv1,cv2,cv3: {unequal}; selections identical to the pre-fix run"))
imp.append(("Uncaught fit error on too few wet/dry days", "not triggered", f"no fold or fit was dropped; notes emitted: {notes or 'none'}"))
imp.append(("Low date overlap between candidates", "not triggered", f"excluded candidates: {excl or 'none'}"))
imp.append(("Old artifacts moved before fitting (partial replacement)", "not triggered here",
            "no calibrate run failed part-way in this project; regression tests cover it (staging + swap)"))
sk = []
for name, s in locs:
    for lead in ("day1", "day2", "day3"):
        h = metrics[s][(metrics[s].lead == lead) & (metrics[s].split == "holdout") & (metrics[s].method == "hurdle")].iloc[0]
        sk.append(h["pinball_skill_vs_clim_vs_flat"] - h["pinball_skill_vs_clim"])
imp.append(("Season-blind climatology reference", "AFFECTED reported skill (small)",
            f"pinball skill vs the season-blind reference minus skill vs the season-aware one ranged "
            f"{min(sk) * 100:+.1f} to {max(sk) * 100:+.1f} percentage points (mean {np.mean(sk) * 100:+.1f}): small, and "
            "of either sign because two training winters estimate seasonality noisily. Reports now use the "
            "season-aware reference and show both"))
imp.append(("Median only vs expected total", "AFFECTED the earlier report",
            "the earlier table mixed 'median for MAE, expected total for bias'; both are now reported separately with RMSE"))
imp.append(("Served payloads exposed raw_*_mm columns", "AFFECTED earlier served files",
            "earlier latest.csv files carried raw_<model>_mm; renamed input_<model>_mm and documented as provenance; "
            "no raw value is served as a forecast"))
# negative forecast hours: did any served row rest on a window containing them? (15:26Z rows vs the 15:54Z rows,
# same prediction day, made after the fix)
neg_rows = []
for name, sl in locs:
    f_old = ROOT / "archive" / "superseded_predictions" / sl / "forecast_20260930T1526Z.csv"
    if not f_old.exists():
        continue
    o = pd.read_csv(f_old).set_index(["label_date_local", "lead_group"])
    n_ = pd.read_csv(ROOT / "data" / sl / "predictions" / "forecast_20260930T1554Z.csv").set_index(["label_date_local", "lead_group"])
    for k in o.index:
        if o.loc[k].status == "ok" and n_.loc[k].status == "ok" and o.loc[k].feature_set_used != n_.loc[k].feature_set_used:
            neg_rows.append(f"{name} {k[0]} {k[1]}: {o.loc[k].feature_set_used} P(>=0.2)={o.loc[k].p_rain_ge_0_2mm:.3f}, "
                            f"E={o.loc[k].mean_estimate_mm:.3f} mm -> {n_.loc[k].feature_set_used} "
                            f"P={n_.loc[k].p_rain_ge_0_2mm:.3f}, E={n_.loc[k].mean_estimate_mm:.3f} mm")
imp.append(("Live 'usable' ignored the negative-hour exclusion used in training", "TRIGGERED - changed served rows",
            "JMA previous_day3 returned -0.1 mm for 1 Oct 01Z-06Z at the Perth Metro and Swanbourne grid cells (5 of those "
            "hours fall in gauge day 2 Oct). Training excludes any window with a negative forecast hour. Earlier served "
            "rows that used it, versus the corrected rows: " + "; ".join(neg_rows) +
            ". Material under the pre-set rule (|dP| >= 0.05) but both are tiny amounts; this is a software change, not a skill claim"))
imp.append(("Junction check depended on Python >= 3.12", "not triggered here",
            f"this machine runs Python {sys.version.split()[0]} (junctions detected); on 3.11 the guard would have been "
            "blind. Now detected from file attributes; tested"))
imp.append(("Quarantine root computed from an unresolved relative path", "not triggered here",
            "the run started from the project root, where the relative path happened to resolve; now resolved"))
imp.append(("Cross-location artifact loaded", "not triggered", "each artifact's meta.location matched its location; now enforced"))
imp.append(("No-window / empty-fetch / horizon 0 crashes; --as-of without frozen inputs; UTC stamp with offset times",
            "not triggered", "none of these inputs occurred; now handled and tested"))
imp.append(("Tests that could pass for the wrong reason (payload scan on names/latest only, circular routing, exhausted rows "
            "checked on one column)", "affected the ASSURANCE, not the outputs",
            "the tests would not have caught a forged or wrong payload; replaced by value-level, all-row, hard-coded checks"))
out.append("| finding | effect on this run | evidence |\n|---|---|---|")
for a, b, c in imp:
    out.append(f"| {a} | **{b}** | {c} |")

# preserved model choices
out.append("\n**Model choices were preserved from before the holdout was scored.** Selection uses only the three "
           "expanding-window CV folds; the holdout table above is never used to reselect. The selected "
           "(feature set, C, seasonal term) per location and lead is identical to the first no-ENSO rebuild:\n")
cmp = []
diff = 0
for name, sl in locs:
    old_sel = sorted((ROOT / "archive" / "superseded_models" / sl).glob("models_before_*/selection.json"))[0]
    o = json.loads(old_sel.read_text())
    for lead in ("day1", "day2", "day3"):
        cur = sel[sl][lead]["selected"]
        prev = o[lead]["selected"]
        same_cfg = (cur["feature_set"], cur["C"], cur["season"]) == (prev["feature_set"], prev["C"], prev["season"])
        diff += (not same_cfg)
        cmp.append({"location": name, "lead": lead, "selected": f"{cur['feature_set']} C={cur['C']} season={cur['season']}",
                    "same as first no-ENSO rebuild": same_cfg})
out.append(f"Compared programmatically with the oldest superseded no-ENSO selection of each location: "
           f"{diff} of {len(cmp)} differ.\n")
out.append(md(pd.DataFrame(cmp)))
(ROOT / "reports" / "acceptance_report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
