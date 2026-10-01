"""Baseline verification on training-eligible pairs only."""
from __future__ import annotations

import numpy as np
import pandas as pd


def contingency(fc: pd.Series, ob: pd.Series, thr: float) -> dict:
    f, o = fc >= thr, ob >= thr
    hits, misses = int((f & o).sum()), int((~f & o).sum())
    fa, cn = int((f & ~o).sum()), int((~f & ~o).sum())
    div = lambda a, b: a / b if b else np.nan
    return {f"hits_{thr}": hits, f"misses_{thr}": misses, f"false_alarms_{thr}": fa,
            f"correct_negatives_{thr}": cn, f"POD_{thr}": div(hits, hits + misses),
            f"FAR_{thr}": div(fa, hits + fa), f"CSI_{thr}": div(hits, hits + misses + fa),
            f"freq_bias_{thr}": div(hits + fa, hits + misses),
            f"accuracy_{thr}": div(hits + cn, hits + misses + fa + cn)}


def scores(fc: pd.Series, ob: pd.Series, thresholds) -> dict:
    err = fc - ob
    d = {"n": int(len(fc)), "MAE_mm": float(err.abs().mean()), "mean_error_fc_minus_obs_mm": float(err.mean()),
         "RMSE_mm": float(np.sqrt((err ** 2).mean())), "mean_fc_mm": float(fc.mean()),
         "mean_obs_mm": float(ob.mean())}
    for t in thresholds:
        d.update(contingency(fc, ob, t))
    return d


def per_model(long: pd.DataFrame, thresholds) -> pd.DataFrame:
    e = long[long.training_eligible]
    rows = []
    for (src, model, lg), g in e.groupby(["source_api", "model", "lead_group"]):
        rows.append({"comparison": "all_eligible_rows_per_model", "source_api": src, "model": model,
                     "lead_group": lg, "first_date": g.label_date_local.min(),
                     "last_date": g.label_date_local.max(),
                     **scores(g.forecast_precip_mm, g.observed_precip_mm, thresholds)})
    return pd.DataFrame(rows)


def complete_case(wide: pd.DataFrame, cols: list[str], label: str, thresholds) -> pd.DataFrame:
    """Identical rows for every model in ``cols`` (all eligible) + an equal-weight blend."""
    rows = []
    for lg, g in wide.groupby("lead_group"):
        mask = np.ones(len(g), bool)
        for c in cols:
            mask &= g[f"eligible_{c}"].to_numpy()
        g = g[mask]
        if len(g) < 30:
            rows.append({"comparison": label, "lead_group": lg, "model": "INSUFFICIENT", "n": len(g)})
            continue
        members = {c: g[f"fc_{c}_mm"] for c in cols}
        members["equal_weight_blend(" + "+".join(cols) + ")"] = g[[f"fc_{c}_mm" for c in cols]].mean(axis=1)
        for name, fc in members.items():
            rows.append({"comparison": label, "lead_group": lg, "model": name,
                         "first_date": g.label_date_local.min(), "last_date": g.label_date_local.max(),
                         **scores(fc, g.observed_precip_mm, thresholds)})
    return pd.DataFrame(rows)


def chronological_scaling(long: pd.DataFrame, source_api: str, model: str, thresholds,
                          train_frac: float = 0.7) -> pd.DataFrame:
    """Multiplicative bias correction fitted on EARLIER dates, evaluated on LATER dates.

    factor = sum(obs_train) / sum(fc_train); nothing from the test period is used.
    """
    rows = []
    e = long[long.training_eligible & (long.source_api == source_api) & (long.model == model)]
    for lg, g in e.groupby("lead_group"):
        g = g.sort_values("window_start_utc")
        dates = g.label_date_local.drop_duplicates().sort_values()
        if len(dates) < 100:
            continue
        cut = dates.iloc[int(len(dates) * train_frac)]
        tr, te = g[g.label_date_local < cut], g[g.label_date_local >= cut]
        factor = tr.observed_precip_mm.sum() / tr.forecast_precip_mm.sum()
        base = {"comparison": "chronological_holdout_scaling", "source_api": source_api, "model": model,
                "lead_group": lg, "train_first": tr.label_date_local.min(), "train_last": tr.label_date_local.max(),
                "test_first": te.label_date_local.min(), "test_last": te.label_date_local.max(),
                "n_train": len(tr), "scale_factor_fitted_on_train": factor}
        rows.append({**base, "variant": "raw", **scores(te.forecast_precip_mm, te.observed_precip_mm, thresholds)})
        rows.append({**base, "variant": "scaled", **scores(te.forecast_precip_mm * factor, te.observed_precip_mm,
                                                           thresholds)})
    return pd.DataFrame(rows)
