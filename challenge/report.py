"""Assemble challenge reports (historical and prospective) as JSON for the dashboard and Markdown for reading."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tomllib

from challenge import scoring

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "challenge"

MULTIPLE_COMPARISONS = ("Every paired comparison in a report (all gauges, leads and competitors) is one family. "
                        "Bootstrap p-values are Holm-adjusted across that whole family before any verdict, so one "
                        "isolated significant result among many is not treated as a win.")
VERDICT_RULE = (f"A verdict needs at least {scoring.MIN_DAYS} common days and {scoring.MIN_WET_DAYS} observed wet days "
                f"(>= 0.2 mm) and a Holm-adjusted p < {scoring.ALPHA}; otherwise 'insufficient evidence' or "
                "'inconclusive'. Wins are never awarded for missing forecasts.")


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso")) if len(df) else []


def _distances(gauge_id: str) -> list[dict]:
    out = []
    for e in tomllib.loads((ROOT / "config.toml").read_text())["locations"]:
        s = e["name"].lower().replace(" ", "_")
        st = json.loads((ROOT / "reports" / s / "run_summary.json").read_text())["station"]
        if st["station_id"] == gauge_id:
            out.append({"location": e["name"], "distance_km": round(float(st["distance_km"]), 1)})
    return out


def build(tab: pd.DataFrame, track: str, key_cols=("window_start_utc",)) -> dict:
    groups, all_paired = [], []
    if tab.empty:
        return {"track": track, "groups": [], "note": "no scored windows yet"}
    for (gid, lead), g in tab.groupby(["gauge_id", "lead"], sort=True):
        g = scoring.add_blend(g[g.observed_precip_mm.notna()])
        if g.empty:
            continue
        cols = scoring.amount_columns(g)
        cov = scoring.coverage(g, cols)
        amounts, common = scoring.amount_scores(g, cols)
        probs = scoring.probability_scores(common, cols)
        rel = pd.concat([scoring.reliability(common, T) for T in (0.2, 1.0)], ignore_index=True)
        paired = scoring.paired(common, cols, key_cols) if len(common) >= 2 else pd.DataFrame()
        if len(paired):
            paired["gauge_id"], paired["lead"] = gid, lead
            all_paired.append(paired)
        dates = pd.to_datetime(common.window_end_utc, utc=True) if len(common) else pd.Series(dtype="datetime64[ns, UTC]")
        groups.append({
            "gauge_id": gid, "gauge_name": g.gauge_name.iloc[0], "lead": lead, "distances": _distances(gid),
            "champion_model": g.champion_model.iloc[0] if "champion_model" in g else None,
            "champion_run_id": g.champion_run_id.iloc[0] if "champion_run_id" in g else None,
            "eligible_windows": len(g), "common_days": len(common),
            "first_date": str(dates.min().date()) if len(dates) else None,
            "last_date": str(dates.max().date()) if len(dates) else None,
            "coverage": _records(cov), "amounts": _records(amounts), "probabilities": _records(probs),
            "reliability": _records(rel), "interval": scoring.interval_coverage(common)})
    paired = scoring.holm(pd.concat(all_paired, ignore_index=True)) if all_paired else pd.DataFrame()
    for grp in groups:
        p = paired[(paired.gauge_id == grp["gauge_id"]) & (paired.lead == grp["lead"])] if len(paired) else paired
        grp["paired"] = _records(p.drop(columns=["gauge_id", "lead"])) if len(p) else []
    return {"track": track, "groups": groups, "multiple_comparisons": MULTIPLE_COMPARISONS, "verdict_rule": VERDICT_RULE,
            "n_comparisons": int(len(paired))}


def to_markdown(rep: dict) -> str:
    lines = [f"# Forecast Challenge: {rep['track']}\n", rep.get("note", ""), rep.get("multiple_comparisons", ""), "",
             rep.get("verdict_rule", ""), ""]
    for g in rep["groups"]:
        lines.append(f"\n## {g['gauge_name']} ({g['gauge_id']}), {g['lead']}: {g['common_days']} common days "
                     f"{g['first_date']} to {g['last_date']}\n")
        lines.append("| competitor | n | MAE | RMSE | bias | wet n | wet MAE | CRPS |\n|---|---|---|---|---|---|---|---|")
        for a in g["amounts"]:
            lines.append(f"| {a['competitor']} | {a['n_days']} | {a.get('MAE')} | {a.get('RMSE')} | {a.get('bias')} | "
                         f"{a['n_wet_days']} | {a.get('wet_MAE')} | {a.get('CRPS')} |")
        lines.append("\nCoverage: " + "; ".join(f"{c['competitor']} {c['with_forecast']}/{c['eligible_windows']}"
                                                  for c in g["coverage"]))
        lines.append("\n| comparison (abs-error diff, negative = first better) | mean | 95% CI | p (Holm) | verdict |\n|---|---|---|---|---|")
        for p in g["paired"]:
            lines.append(f"| {p['first']} vs {p['second']} | {p['mean_abs_error_diff']} | {p['ci95']} | {p['p_holm']} | {p['verdict']} |")
    return "\n".join(lines) + "\n"


def write(rep: dict, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    (OUT / f"{name}.md").write_text(to_markdown(rep), encoding="utf-8")
