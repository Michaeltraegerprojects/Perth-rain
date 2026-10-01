"""Markdown report for the calibration experiment. Verdicts are computed, not asserted."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calibrate import GAP_DAYS, HOLDOUT_FRAC
from .report import _md

CV_FRACTIONS = (0.35, 0.5, 0.65)


def _pick(df: pd.DataFrame, lead: str, method: str) -> pd.Series | None:
    r = df[(df.lead == lead) & (df.method == method) & (df.split == "holdout")]
    return r.iloc[0] if len(r) else None


def verdict(hold: pd.DataFrame, lead: str, paired: pd.DataFrame | None = None) -> list[str]:
    h = _pick(hold, lead, "hurdle")
    c = _pick(hold, lead, "climatology")
    cf = _pick(hold, lead, "climatology_flat")
    if h is None or c is None:
        return [f"* {lead}: no held-out score available."]
    lines = []
    n = int(h.n)
    lines.append(f"* **{lead}** (n={n} held-out days {h.test_first} .. {h.test_last}; model `{h.feature_set}`, "
                 f"fitted on dates up to {h.train_last}; identical station/window/target rows for every method: "
                 f"{h.get('keys_identical_across_methods')}):")
    lines.append(f"  * distribution (pinball loss, lower is better): calibrated {h.pinball_mean:.3f}; "
                 f"season-aware climatology {c.pinball_mean:.3f}" +
                 (f"; season-blind climatology {cf.pinball_mean:.3f}" if cf is not None else "") +
                 f". Skill vs season-aware climatology **{h.get('pinball_skill_vs_clim', np.nan):+.1%}**" +
                 (f" (vs season-blind {h.get('pinball_skill_vs_clim_vs_flat', np.nan):+.1%})" if cf is not None else "") + ".")
    lines.append(f"  * calibrated **median** as the point forecast: MAE {h.median_MAE_mm:.2f} mm, signed bias "
                 f"{h.median_bias_mm:+.2f} mm, RMSE {h.median_RMSE_mm:.2f} mm. Calibrated **expected total**: MAE "
                 f"{h.mean_est_MAE_mm:.2f} mm, bias {h.mean_est_bias_mm:+.2f} mm, RMSE {h.mean_est_RMSE_mm:.2f} mm "
                 f"(observed mean {h.mean_obs_mm:.2f} mm; bias = forecast minus observed).")
    raws = hold[(hold.lead == lead) & hold.method.astype(str).str.startswith("raw:") & (hold.split == "holdout")]
    for _, r in raws.iterrows():
        lines.append(f"  * `{r.method}` raw forecast: MAE {r.MAE_mm:.2f} mm, bias {r.bias_fc_minus_obs_mm:+.2f} mm, "
                     f"RMSE {r.RMSE_mm:.2f} mm.")
    bs = []
    for T in (0.2, 1, 5, 10):
        k = f"brier_{T:g}"
        if k in h and pd.notna(h[k]):
            bss = h.get(f"BSS_{T:g}", np.nan)
            bs.append(f">= {T:g} mm: Brier {h[k]:.3f} (seasonal-climatology {c.get(k, np.nan):.3f}, "
                      f"BSS {bss:+.2f}; event freq {h[f'freq_{T:g}']:.1%}, mean forecast prob {h[f'mean_p_{T:g}']:.1%})")
    if bs:
        lines.append("  * rain-occurrence probabilities (event = rainfall >= the threshold, inclusive; the "
                     "occurrence threshold is 0.2 mm): " + "; ".join(bs) + ". Raw models issue amounts, not "
                     "probabilities, so **no probability comparison with them exists**.")
    lines.append(f"  * quantile calibration - observation exceeded the predicted 50th / 75th / 90th percentile on "
                 f"{h.exceed_q50:.0%} / {h.exceed_q75:.0%} / {h.exceed_q90:.0%} of days "
                 f"(calibrated = at most 50 % / 25 % / 10 %); on rainy days the model's 20-80 % range contained the "
                 f"observed total {h.wetday_cover_60:.0%} of the time (target 60 %).")
    if paired is not None and len(paired):
        pl = paired[(paired.lead == lead) & paired.mean_diff.notna()]
        sig = pl[pl.significant_at_95 == True]  # noqa: E712
        lines.append(f"  * paired differences with 95 % block-bootstrap intervals (7-day blocks): "
                     f"{len(sig)} of {len(pl)} differ from zero at the 95 % level; the rest are descriptive only "
                     f"(see the paired-differences table).")
    return lines


def climate_section(hold: pd.DataFrame, sel: dict) -> list[str]:
    """El Nino / IOD: latest readings, descriptive seasonal relationship, and the held-out with/without test."""
    lines = ["\n## El Nino / Indian Ocean Dipole\n"]
    if sel.get("_enso_status", "").startswith("not used"):
        lines.append("**Not used in these models.** The climate-index feature is marked UNVERIFIED: the definition of "
                     "BoM's index file could not be confirmed from BoM's own documentation (automated access is "
                     "blocked), and its historical values may have been revised after first publication, so a "
                     "historical join could use information that was not available at forecast time. These are the "
                     "no-ENSO baseline models. See reports/enso_source_verification.md and, for the experimental "
                     "evaluation, reports/<location>/enso_audit.md.\n")
        return lines
    info = sel.get("_climate_latest")
    if info:
        r = info["readings"]
        lines.append(f"Latest BoM index readings: Nino 3.4 **{r['nino34']['value']:+.2f} C** (week to "
                     f"{r['nino34']['period_end']}; record maximum since {r['nino34']['record_start']}: "
                     f"{r['nino34']['max_in_record']:+.2f}), IOD **{r['iod']['value']:+.2f} C**, SOI "
                     f"**{r['soi']['value']:+.1f}** - {info['phase']}. Index values are the weekly/30-day numbers "
                     f"in BoM's files; this is a plain-language reading, not BoM's official ENSO status.\n")
    rel = sel.get("_enso_relationship")
    if rel:
        lines.append("**Descriptive check on the gauge's own record** (Spearman rank correlation between a season's "
                     "total rainfall and the index averaged over the preceding season / the same season; one point "
                     "per year, so low statistical power - p < 0.05 would be needed to call it real):\n")
        lines.append(_md(pd.DataFrame(rel)))
    grp, yrs = sel.get("_enso_groups"), sel.get("_enso_years")
    if grp:
        lines.append("\n**Spring (Sep-Nov) rainfall at this gauge grouped by the winter (Jun-Aug) Nino 3.4 state** "
                     "(the index is known before spring starts; +/-0.5 C thresholds; few years per group, so read as "
                     "context, not as a forecast):\n")
        lines.append(_md(pd.DataFrame(grp)))
        if yrs:
            lines.append("\n" + _md(pd.DataFrame(yrs)))
    if len(hold) and "method" in hold.columns:
        h = hold[hold.method.isin(["hurdle:no_climate", "hurdle:climate", "hurdle"])].copy()
        h = h[h.get("climate").notna()] if "climate" in h.columns else h
        rows = []
        for lead, g in h.groupby("lead"):
            nc = g[g.method == "hurdle:no_climate"]
            wc = g[g.method == "hurdle:climate"]
            if nc.empty or wc.empty:
                continue
            a, b = nc.iloc[0], wc.iloc[0]
            rows.append({"lead": lead, "feature_set": a.feature_set, "n_heldout_days": int(a.n),
                         "test_first": a.test_first, "test_last": a.test_last,
                         "pinball_no_climate": round(a.pinball_mean, 4),
                         "pinball_with_enso_iod": round(b.pinball_mean, 4),
                         "relative_change": f"{(b.pinball_mean / a.pinball_mean - 1):+.1%}",
                         "with_variant": b.climate,
                         "BSS1_no_climate": round(a.get("BSS_1", np.nan), 3),
                         "BSS1_with": round(b.get("BSS_1", np.nan), 3)})
        if rows:
            lines.append("\n**Held-out test - identical days, same model family, with vs without the ENSO/IOD "
                         "regressors** (negative relative_change = lower pinball loss = better with El Nino). "
                         "The `jma_gsm long record` rows use 2016-2026, where the index actually varied "
                         "(El Nino 2023 and 2026, La Nina 2020-22); the other rows cover 2024-2026 only:\n")
            lines.append(_md(pd.DataFrame(rows)))
            better = [r for r in rows if r["relative_change"].startswith("-")]
            lines.append(f"\nENSO/IOD lowered held-out loss in {len(better)} of {len(rows)} tests; changes of a "
                         f"percent or two are within sampling noise for a few hundred days.\n")
    lines.append("These are associations and held-out scores only. They do not establish that El Nino causes any "
                 "change in local rainfall, and the climate-index feature itself is unverified.\n")
    return lines


def drift_table(wide: pd.DataFrame, lead: str) -> pd.DataFrame:
    """Total forecast / total observed rain per calendar quarter, per model, over eligible days.
    A ratio that moves over time means a model's bias is not stationary (model upgrades, regime
    change), which limits how well a calibration fitted on the past transfers to the future."""
    d = wide[wide.lead_group == lead].copy()
    d["quarter"] = pd.to_datetime(d.label_date_local).dt.to_period("Q").astype(str)
    rows = []
    for c in [c[len("fc_"):-len("_mm")] for c in d.columns if c.startswith("fc_")]:
        e = d[d[f"eligible_{c}"].astype(bool) & d[f"fc_{c}_mm"].notna() & d.observed_precip_mm.notna()]
        for q, g in e.groupby("quarter"):
            if len(g) >= 40 and g.observed_precip_mm.sum() > 5:
                rows.append({"model": c, "quarter": q, "days": len(g),
                             "obs_total_mm": round(g.observed_precip_mm.sum(), 0),
                             "fc_over_obs": round(g[f"fc_{c}_mm"].sum() / g.observed_precip_mm.sum(), 2)})
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows)
    return t.pivot(index="quarter", columns="model", values="fc_over_obs").reset_index()


def _station_label(cfg) -> str:
    import json
    p = cfg.reports_dir / "run_summary.json"
    if not p.exists():
        return "unknown"
    st = json.loads(p.read_text()).get("station") or {}
    return f"{st.get('station_id')} {st.get('station_name')}, {float(st.get('distance_km', 'nan')):.1f} km from {cfg.location_name}"


def write_calibration_report(cfg, metrics: pd.DataFrame, sel: dict, wide: pd.DataFrame | None = None,
                             suffix: str = "", paired: pd.DataFrame | None = None) -> None:
    hold = metrics[metrics.split == "holdout"] if len(metrics) else metrics
    cols_h = ["lead", "method", "feature_set", "n", "test_first", "test_last", "pinball_mean",
              "pinball_skill_vs_clim", "median_MAE_mm", "median_bias_mm", "mean_est_MAE_mm", "mean_est_bias_mm",
              "mean_est_RMSE_mm", "brier_0.2", "BSS_0.2", "exceed_q50", "exceed_q75", "exceed_q90", "auc_0.2"]
    run = sel.get("_run", {})
    cols_r = ["lead", "method", "n", "MAE_mm", "bias_fc_minus_obs_mm", "RMSE_mm", "brier_1", "auc_1"]
    txt = [f"# Calibration report - {cfg.location_name}{' (EXPERIMENTAL, unverified ENSO feature)' if suffix else ''}\n",
           f"Run `{run.get('run_id')}` created {run.get('created_utc')} - **enso_status = {run.get('enso_status')}** - "
           f"code sha256 `{str(run.get('code_sha256'))[:12]}` (full provenance in data/<location>/models/manifest.json).\n",
           f"Gauge: **{_station_label(cfg)}** (see data_quality_report.md). Held-out scores below use "
           f"chronologically **held-out** days (last {int(HOLDOUT_FRAC * 100)} % of the common dates); models were "
           f"selected on earlier expanding-window folds ({len(CV_FRACTIONS)} folds, {GAP_DAYS}-day gap; cross-validation "
           f"scores are NOT held-out) and the held-out block was scored once.\n",
           "## Verdicts (computed from the held-out scores)\n"]
    for lead in ("day1", "day2", "day3"):
        if lead in sel:
            txt += verdict(hold, lead, paired)
    txt += climate_section(hold, sel)
    txt.append("\n## Selection (cross-validation on pre-holdout dates)\n")
    for lead, s in ((k, v) for k, v in sel.items() if not k.startswith("_")):
        txt.append(f"**{lead}** - eval dates {s['eval_first']} .. {s['eval_last']} ({s['n_eval_dates']} common days); "
                   f"final model trained on {s['n_train_final']} days {s['train_first']} .. {s['train_last']}")
        txt.append(_md(pd.DataFrame(s["ranking"])))
    if len(hold):
        h = hold[hold.method.isin(["hurdle", "climatology", "climatology_flat", "hurdle(alt)"])]
        txt.append("\n## Held-out probabilistic scores\n")
        txt.append(_md(h[[c for c in cols_h if c in h.columns]]))
        r = hold[hold.method.astype(str).str.startswith("raw:")]
        txt.append("\n## Held-out raw model forecasts (same days)\n")
        txt.append(_md(r[[c for c in cols_r if c in r.columns]]))
        if paired is not None and len(paired):
            txt.append("\n## Paired differences with uncertainty (first method minus second; negative = first is better)\n")
            txt.append("Per-day differences on identical held-out days, circular moving-block bootstrap (7-day blocks, "
                       "2000 resamples). `significant_at_95` = the 95 % interval excludes zero. Where it does not, "
                       "the ranking is **descriptive, not significant**.\n")
            cols_p = [c for c in ["lead", "comparison", "n_days", "mean_diff", "ci95_lo", "ci95_hi",
                                  "share_first_better", "significant_at_95", "note"] if c in paired.columns]
            txt.append(_md(paired[cols_p]))
    if wide is not None:
        txt.append("\n## Is the forecast bias stable? (forecast total / observed total, per quarter)\n")
        for lead in ("day2",):
            dt = drift_table(wide, lead)
            if len(dt):
                txt.append(f"{lead} - a value of 1.00 means the model's rain total matched the gauge; "
                           f"values that move over time mean past calibration transfers less well:\n")
                txt.append(_md(dt))
    txt.append("""
## How to read this / limits

* `pinball_skill_vs_clim`, `BSS_*` are skill scores against a **season-aware climatology** (same calendar month
  +/- 1 month) fitted on the training dates only; `*_vs_flat` columns use a season-blind climatology, which is an
  easier reference for a March-September block. 0 = no better than climatology, 1 = perfect, negative = worse.
* Raw model rows are deterministic millimetre forecasts. They issue no probabilities, so no probability score can
  be compared with them; only MAE, signed bias and RMSE are.
* Event definition: rainfall **>= 0.2 mm** (inclusive) counts as wet (gauge resolution); every threshold event
  is `>=`. Observations below 0.2 mm are dry.
* "Median" and "expected total" are two different point forecasts of the same predictive distribution; each is
  scored separately.
* Exceedance probabilities above the modelled 95th conditional percentile use half of the unresolved 5 % tail.
* Held-out blocks are a few hundred days: differences of a few percent are within noise. Skill for
  `>= 10 mm` rests on very few events.
* Predictions describe the **gauge**, not the suburb: distance to the gauge is in the data-quality report.
""")
    (cfg.reports_dir / f"calibration_report{suffix}.md").write_text("\n".join(txt), encoding="utf-8")
