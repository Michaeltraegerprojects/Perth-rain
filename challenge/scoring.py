"""Scores for the Forecast Challenge. Amounts and probabilities are scored separately.

Fairness rules applied here:
* every leaderboard uses the IDENTICAL set of keys (gauge, 9am-9am window, lead) on which every listed competitor
  has a forecast; coverage (how often each competitor had a forecast at all) is reported separately;
* missing forecasts are never scored and never counted as wins;
* deterministic models have no probabilities: their probability scores are reported as unavailable. For a point
  forecast the CRPS equals the absolute error, which is reported as such;
* paired differences use a circular moving-block bootstrap over days (7-day blocks); p-values are Holm-adjusted
  across every comparison in the same report; a verdict needs enough days and wet days, otherwise
  "insufficient evidence"; a non-significant difference is "inconclusive".
"""
from __future__ import annotations

import tomllib
from math import comb, erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLDS = (0.2, 1.0, 5.0, 10.0)
WET = 0.2
SETTINGS = Path(__file__).resolve().parent / "settings.toml"


def load_rules(path: Path | None = None) -> dict:
    """The verdict rules, read and validated from settings.toml [verdicts] (the single source of truth)."""
    raw = tomllib.loads(Path(path or SETTINGS).read_text(encoding="utf-8")).get("verdicts", {})
    rules = {"min_days": raw.get("min_days"), "min_wet_days": raw.get("min_wet_days"), "alpha": raw.get("alpha")}
    for k in ("min_days", "min_wet_days"):
        if not isinstance(rules[k], int) or isinstance(rules[k], bool) or rules[k] < 1:
            raise ValueError(f"[verdicts] {k} must be a positive integer, got {rules[k]!r}")
    if not isinstance(rules["alpha"], (int, float)) or not 0 < rules["alpha"] < 1:
        raise ValueError(f"[verdicts] alpha must be between 0 and 1, got {rules['alpha']!r}")
    return rules
LABELS = {
    "champion_median": "Our forecast (median)", "champion_expected": "Our forecast (expected total)",
    "sr_ecmwf_ifs": "ECMWF IFS (raw)", "pr_ecmwf_ifs025": "ECMWF IFS 0.25° (raw)", "pr_jma_gsm": "JMA GSM (raw)",
    "pr_ncep_gfs_global": "NOAA GFS (raw)", "blend": "Equal-weight blend of existing raw models",
    "icon_global": "DWD ICON Global (challenger, raw)", "ecmwf_aifs025_single": "ECMWF AIFS (challenger, raw)",
    "icon_global_12z": "DWD ICON Global, 12 UTC run (challenger, raw)",
    "ecmwf_aifs025_single_12z": "ECMWF AIFS, 12 UTC run (challenger, raw)",
    "clim": "Season-aware climatology", "champion_distribution": "Our forecast (full distribution, CRPS)",
}
EXISTING_RAW = ["sr_ecmwf_ifs", "pr_ecmwf_ifs025", "pr_jma_gsm", "pr_ncep_gfs_global"]
CHALLENGERS = ["icon_global", "ecmwf_aifs025_single", "icon_global_12z", "ecmwf_aifs025_single_12z"]


def block_bootstrap(diff: np.ndarray, block: int = 7, B: int = 2000, seed: int = 0) -> dict:
    """Circular moving-block bootstrap of a chronologically ordered per-day difference series (same method as the
    audited release). The p-value uses the bootstrap standard error with a normal approximation, so it is not
    floored at 1/B (a floored p-value could never survive a Holm adjustment over many comparisons)."""
    diff = np.asarray(diff, float)
    n = len(diff)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, nb))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(B, nb * block)[:, :n]
    means = diff[idx].mean(axis=1)
    se = float(means.std(ddof=1))
    mean = float(diff.mean())
    if se > 1e-12 * max(1.0, float(np.abs(diff).max())):
        z = abs(mean) / se
        p = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
    else:
        # no resampling variability: identical errors, the same difference every day, or fewer days than one
        # block (every resample is then the whole series). The normal approximation is undefined; use the exact
        # two-sided sign test, which gives a tie p = 1 and one day p = 1.
        p = sign_test_p(diff)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(diff.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi), "se": se, "p": float(p)}


def amount_columns(tab: pd.DataFrame) -> dict[str, str]:
    cols = {"champion_median": "champion_median_mm", "champion_expected": "champion_expected_mm"}
    for c in EXISTING_RAW + CHALLENGERS:
        if f"fc_{c}_mm" in tab and tab[f"fc_{c}_mm"].notna().any():
            cols[c] = f"fc_{c}_mm"
    if "fc_blend_mm" in tab and tab.fc_blend_mm.notna().any():
        cols["blend"] = "fc_blend_mm"
    return cols


def add_blend(tab: pd.DataFrame) -> pd.DataFrame:
    raw = [f"fc_{c}_mm" for c in EXISTING_RAW if f"fc_{c}_mm" in tab and tab[f"fc_{c}_mm"].notna().any()]
    tab = tab.copy()
    if len(raw) > 1:
        tab["fc_blend_mm"] = tab[raw].mean(axis=1, skipna=False)      # only when every member is present
    return tab


def coverage(tab: pd.DataFrame, cols: dict) -> pd.DataFrame:
    rows = []
    for k, c in cols.items():
        have = tab[c].notna()
        rows.append({"competitor": LABELS.get(k, k), "key": k, "eligible_windows": len(tab),
                     "with_forecast": int(have.sum()), "missing": int((~have).sum()),
                     "coverage": round(float(have.mean()), 4) if len(tab) else None})
    return pd.DataFrame(rows)


def amount_scores(tab: pd.DataFrame, cols: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = tab.dropna(subset=list(cols.values()) + ["observed_precip_mm"])
    y = common.observed_precip_mm.to_numpy(float)
    wet = y >= WET
    rows = []
    for k, c in cols.items():
        f = common[c].to_numpy(float)
        e = f - y
        rows.append({"competitor": LABELS.get(k, k), "key": k, "n_days": len(y), "n_wet_days": int(wet.sum()),
                     "MAE": _r(np.abs(e).mean()), "RMSE": _r(np.sqrt((e ** 2).mean())), "bias": _r(e.mean()),
                     "wet_MAE": _r(np.abs(e[wet]).mean()) if wet.any() else None,
                     "wet_bias": _r(e[wet].mean()) if wet.any() else None,
                     "CRPS": _r(np.abs(e).mean()) if k not in ("champion_median", "champion_expected") else None})
    if "champion_crps" in common:
        rows.append({"competitor": "Our forecast (full distribution)", "key": "champion_distribution",
                     "n_days": len(y), "n_wet_days": int(wet.sum()), "CRPS": _r(common.champion_crps.mean())})
        rows.append({"competitor": LABELS["clim"], "key": "clim", "n_days": len(y), "n_wet_days": int(wet.sum()),
                     "CRPS": _r(common.clim_crps.mean())})
    return pd.DataFrame(rows), common


def probability_scores(common: pd.DataFrame, cols: dict) -> pd.DataFrame:
    y = common.observed_precip_mm.to_numpy(float)
    rows = []
    for T in THRESHOLDS:
        ev = (y >= T).astype(float)
        for who, col in (("champion", f"champion_p_ge_{T:g}"), ("clim", f"clim_p_ge_{T:g}")):
            if col in common:
                rows.append({"threshold_mm": T, "competitor": "Our forecast" if who == "champion" else LABELS["clim"],
                             "key": who, "n_days": len(y), "n_events": int(ev.sum()),
                             "Brier": _r(np.mean((common[col].to_numpy(float) - ev) ** 2), 4)})
        for k in cols:
            if k.startswith("champion"):
                continue
            rows.append({"threshold_mm": T, "competitor": LABELS.get(k, k), "key": k, "n_days": len(y),
                         "n_events": int(ev.sum()), "Brier": None,
                         "note": "unavailable: deterministic rain amount, no probability issued"})
    return pd.DataFrame(rows)


def reliability(common: pd.DataFrame, T: float, min_bin=10) -> pd.DataFrame:
    col = f"champion_p_ge_{T:g}"
    if col not in common:
        return pd.DataFrame()
    p = common[col].to_numpy(float)
    ev = (common.observed_precip_mm.to_numpy(float) >= T).astype(float)
    edges = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])
    idx = np.digitize(p, edges) - 1
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        rows.append({"threshold_mm": T, "bin": f"{edges[b]:.2f}-{min(edges[b + 1], 1):.2f}", "n": int(m.sum()),
                     "mean_forecast": _r(p[m].mean(), 3), "observed_frequency": _r(ev[m].mean(), 3),
                     "enough_data": bool(m.sum() >= min_bin)})
    return pd.DataFrame(rows)


def interval_coverage(common: pd.DataFrame) -> dict:
    if "champion_p10_mm" not in common:
        return {}
    y = common.observed_precip_mm.to_numpy(float)
    lo, hi = common.champion_p10_mm.to_numpy(float), common.champion_p90_mm.to_numpy(float)
    return {"n_days": len(y), "inside_10_90": _r(np.mean((y >= lo - 1e-9) & (y <= hi + 1e-9)), 3),
            "nominal": 0.8, "above_90th": _r(np.mean(y > hi + 1e-9), 3), "nominal_above": 0.1,
            "note": "Dry days sit exactly on a 0 mm lower bound, so 'inside' is inflated; 'above 90th' is the cleaner check."}


def paired(common: pd.DataFrame, cols: dict, keys=("window_start_utc",)) -> pd.DataFrame:
    """Per-day absolute-error differences, each ours-minus-other (negative = ours better), plus each challenger
    minus raw ECMWF IFS. Chronological order is required for the block bootstrap."""
    c = common.sort_values(list(keys))
    y = c.observed_precip_mm.to_numpy(float)
    rows = []
    pairs = [(a, b) for a in ("champion_median", "champion_expected") for b in cols if not b.startswith("champion")]
    pairs += [(ch, "sr_ecmwf_ifs") for ch in CHALLENGERS if ch in cols and "sr_ecmwf_ifs" in cols]
    if "champion_crps" in c:   # whole-distribution score; for a point forecast the CRPS is its absolute error
        pairs += [("champion_distribution", b) for b in cols if not b.startswith("champion")]
    for a, b in pairs:
        fa = c.champion_crps.to_numpy(float) if a == "champion_distribution" else np.abs(c[cols[a]].to_numpy(float) - y)
        d = fa - np.abs(c[cols[b]].to_numpy(float) - y)
        r = block_bootstrap(d)
        p = r["p"]
        rows.append({"first": LABELS.get(a, a), "second": LABELS.get(b, b), "first_key": a, "second_key": b,
                     "mean_diff_exact": float(r["mean_diff"]),     # unrounded: verdicts use this, never the display value
                     "score": "CRPS" if a == "champion_distribution" else "absolute error",
                     "n_days": len(y), "n_wet_days": int((y >= WET).sum()), "mean_abs_error_diff": _r(r["mean_diff"], 4),
                     "ci95": [_r(r["ci95_lo"], 4), _r(r["ci95_hi"], 4)], "p_bootstrap": float(p)})
    return pd.DataFrame(rows)


def sign_test_p(diff: np.ndarray) -> float:
    """Exact two-sided sign test on the non-zero daily differences (zero differences are ties and dropped)."""
    diff = np.asarray(diff, float)
    nz = diff[np.abs(diff) > 1e-12 * max(1.0, float(np.abs(diff).max()) if len(diff) else 1.0)]
    m = len(nz)
    if m == 0:
        return 1.0
    k = int((nz > 0).sum())
    tail = sum(comb(m, i) for i in range(min(k, m - k) + 1)) / 2 ** m
    return float(min(1.0, 2 * tail))


def holm(df: pd.DataFrame, rules: dict | None = None) -> pd.DataFrame:
    """Holm step-down adjustment across all rows; adds verdicts using the rules from settings.toml."""
    rules = rules or load_rules()
    if df.empty:
        return df
    df = df.copy()
    order = np.argsort(df.p_bootstrap.to_numpy())
    m = len(df)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * df.p_bootstrap.iloc[i]))
        adj[i] = running
    df["p_holm"] = adj

    def verdict(r):
        if r.n_days < rules["min_days"] or r.n_wet_days < rules["min_wet_days"]:
            return "insufficient evidence"
        d = r.mean_diff_exact if "mean_diff_exact" in r and pd.notna(r.mean_diff_exact) else r.mean_abs_error_diff
        if r.p_holm < rules["alpha"] and d != 0:
            return "first better" if d < 0 else "second better"
        return "inconclusive"
    df["verdict"] = df.apply(verdict, axis=1)
    return df


def _r(x, nd=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)
