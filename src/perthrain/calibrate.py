"""Local rainfall calibration: occurrence + amount (hurdle) model with predictive quantiles.

Model (per gauge and lead group):

* occurrence  P(rain >= 0.2 mm)  - logistic regression on log(1+forecast mm) of the chosen
  model(s) plus a seasonal cycle;
* amount | rain - linear quantile regression of log(mm) on the same features at levels
  0.05 ... 0.95 (19 levels), rearranged to be monotone;
* combined into exceedance probabilities P(rain >= T), an unconditional predictive
  distribution (dry outcomes are exactly 0), a median and a mean estimate.

Validation is strictly chronological: the last ``HOLDOUT_FRAC`` of the common evaluation
dates is never used for fitting or selection; feature sets / regularisation are chosen on
expanding-window folds inside the earlier dates; every fold keeps a ``GAP_DAYS`` gap between
training and test dates. The final model is refitted on all data only after the held-out
score has been recorded.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("perthrain.calibrate")

WET_MM = 0.2
GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)          # 19 conditional quantile levels
THRESHOLDS = (0.2, 1.0, 5.0, 10.0)
LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)
GAP_DAYS = 3
HOLDOUT_FRAC = 0.20
MIN_TRAIN = 150
LOG_WET = float(np.log(WET_MM))

# feature-set name -> model columns (as in training_wide: fc_<col>_mm), how they are combined and the
# leads they can serve. Correlated models are NOT fed in as separate regressors: with r ~ 0.7-0.93 the
# quantile regression overfits the amounts (measured: wet-day MAE 3.6-3.9 mm vs 2.5-2.6 mm for one model),
# so multi-model sets are collapsed to ONE feature, the mean of log(1+mm).
FEATURE_SETS: dict[str, dict] = {
    "ecmwf_ifs_sr": {"cols": ["sr_ecmwf_ifs"], "leads": ["day1", "day2", "day3"]},
    "ecmwf_ifs025_pr": {"cols": ["pr_ecmwf_ifs025"], "leads": ["day2", "day3"]},
    "jma_gsm_pr": {"cols": ["pr_jma_gsm"], "leads": ["day2", "day3"]},
    "ecmwf_composite": {"cols": ["sr_ecmwf_ifs", "pr_ecmwf_ifs025"], "combine": "mean_log",
                        "leads": ["day2", "day3"]},
    "all_composite": {"cols": ["sr_ecmwf_ifs", "pr_jma_gsm", "pr_ncep_gfs_global", "pr_ecmwf_ifs025"],
                      "combine": "mean_log", "leads": ["day2", "day3"]},
}
C_GRID = (0.1, 1.0)
SEASON_GRID = (True, False)
# climate-driver regressors (BoM weekly Nino 3.4 / IOD, lagged 14 days - see climate.py). Whether they
# help is decided by cross-validation; None = not used.
CLIMATE_GRID = (None, "nino34", "nino34_iod")
CLIMATE_COLS = {"nino34": ["clim_nino34"], "nino34_iod": ["clim_nino34", "clim_iod"]}


# ------------------------------------------------------------------------ features
def make_X(df: pd.DataFrame, cols: list[str], combine: str | None = None, season: bool = True,
           climate: str | None = None) -> np.ndarray:
    feats = [np.log1p(df[f"fc_{c}_mm"].to_numpy(float)) for c in cols]
    if combine == "mean_log":
        feats = [np.mean(feats, axis=0)]
    if season:
        doy = pd.to_datetime(df["label_date_local"]).dt.dayofyear.to_numpy(float)
        ang = 2 * np.pi * doy / 365.25
        feats += [np.sin(ang), np.cos(ang)]
    for c in CLIMATE_COLS.get(climate, []):
        feats.append(df[c].to_numpy(float))
    return np.column_stack(feats)


def feature_names(cols: list[str], combine: str | None = None, season: bool = True,
                  climate: str | None = None) -> list[str]:
    """Names of the columns make_X builds, in the same order (the artifact's declared feature schema)."""
    names = [f"log1p(fc_{c}_mm)" for c in cols]
    if combine == "mean_log":
        names = ["mean(" + ", ".join(names) + ")"]
    if season:
        names += ["season_sin(day_of_year)", "season_cos(day_of_year)"]
    return names + list(CLIMATE_COLS.get(climate, []))


def fit_X(df: pd.DataFrame, fit: dict) -> np.ndarray:
    X = make_X(df, fit["cols"], fit.get("combine"), fit.get("season", True), fit.get("climate"))
    schema = fit.get("feature_schema")
    if schema is not None and X.shape[1] != len(schema):
        raise ValueError(f"feature matrix has {X.shape[1]} columns but the artifact schema declares {len(schema)}")
    return X


# ------------------------------------------------------------------------- model
class HurdleModel:
    def __init__(self, C: float = 1.0, alpha: float = 1e-3):
        self.C, self.alpha = C, alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HurdleModel":
        y = np.asarray(y, float)
        wet = y >= WET_MM
        if wet.sum() < 40 or (~wet).sum() < 40:
            raise ValueError(f"too few wet/dry rows to fit ({wet.sum()} wet, {(~wet).sum()} dry)")
        self.clf_ = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=2000)).fit(X, wet)
        self.scaler_ = StandardScaler().fit(X[wet])
        Xw, lw = self.scaler_.transform(X[wet]), np.log(y[wet])
        self.q_ = [QuantileRegressor(quantile=float(q), alpha=self.alpha, solver="highs").fit(Xw, lw)
                   for q in GRID]
        # the 19 quantile levels leave out the upper 5 % tail; a ratio estimator fitted on the
        # training rows restores the mean (validated on held-out data, never on the fit rows)
        _, lq = self.parts(X[wet])
        grid_mean = np.exp(lq).mean(axis=1)
        self.tail_scale_ = float(y[wet].mean() / grid_mean.mean())
        return self

    def parts(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.clf_.predict_proba(X)[:, 1]
        Xw = self.scaler_.transform(X)
        lq = np.column_stack([m.predict(Xw) for m in self.q_])
        lq = np.maximum(np.sort(lq, axis=1), LOG_WET)
        return p, lq


def climatology_flat(y_train: np.ndarray, n: int):
    """Season-blind reference: one wet frequency and one wet-amount distribution for every day."""
    y_train = np.asarray(y_train, float)
    wet = y_train >= WET_MM
    lq_row = np.log(np.quantile(y_train[wet], GRID))
    scale = float(y_train[wet].mean() / np.exp(lq_row).mean())
    return np.full(n, wet.mean()), np.tile(lq_row, (n, 1)), scale


def _month_window(m: int) -> list[int]:
    return [(m - 2) % 12 + 1, m, m % 12 + 1]


def climatology_seasonal(y_train: np.ndarray, dates_train, dates_test, min_days: int = 60, min_wet: int = 20):
    """Season-aware reference: for each test day use the training days of the same calendar month +/- 1 month
    (falls back to the flat climatology when the pooled window has too few days). Fitted on training dates only."""
    y = np.asarray(y_train, float)
    mt = pd.DatetimeIndex(dates_train).month.to_numpy()
    fp, flq, fscale = climatology_flat(y, 1)
    cache = {}
    for m in range(1, 13):
        yy = y[np.isin(mt, _month_window(m))]
        wet = yy >= WET_MM
        # wet FREQUENCY comes from the seasonal window whenever it holds enough days (a genuinely dry season has
        # zero wet days and must not fall back to the flat frequency); only the wet-AMOUNT distribution falls back
        # to the flat one when the window has too few wet days to describe it
        p_m = float(wet.mean()) if len(yy) >= min_days else float(fp[0])
        if wet.sum() >= min_wet:
            row = np.log(np.quantile(yy[wet], GRID))
            cache[m] = (p_m, row, float(yy[wet].mean() / np.exp(row).mean()))
        else:
            cache[m] = (p_m, flq[0], fscale)
    md = pd.DatetimeIndex(dates_test).month.to_numpy()
    return (np.array([cache[m][0] for m in md]), np.vstack([cache[m][1] for m in md]),
            np.array([cache[m][2] for m in md]))


# ------------------------------------------------------- distribution functions
def exceed_prob(p: np.ndarray, lq: np.ndarray, T: float) -> np.ndarray:
    """P(rain >= T mm). Above the modelled 95th conditional percentile the unresolved
    tail (5 %) is split evenly: exceedance there is reported as 0.025 * p."""
    if T <= WET_MM:
        return p.copy()
    lt = np.log(T)
    qs = np.concatenate([[0.0], GRID])
    out = np.empty(len(p))
    for i in range(len(p)):
        xs = np.concatenate([[LOG_WET], lq[i]])
        out[i] = np.interp(lt, xs, qs, left=0.0, right=0.975)
    return p * (1.0 - out)


def mixture_quantile(p: np.ndarray, lq: np.ndarray, tau: float) -> np.ndarray:
    """tau-quantile of the unconditional amount; dry outcomes are exactly 0."""
    qs = np.concatenate([[0.0], GRID])
    out = np.zeros(len(p))
    for i in range(len(p)):
        dry = 1.0 - p[i]
        if tau <= dry:
            continue
        u = min((tau - dry) / p[i], GRID[-1])
        xs = np.concatenate([[LOG_WET], lq[i]])
        out[i] = float(np.exp(np.interp(u, qs, xs)))
    return out


def mean_estimate(model_or_scale, p: np.ndarray, lq: np.ndarray) -> np.ndarray:
    scale = model_or_scale.tail_scale_ if hasattr(model_or_scale, "tail_scale_") else np.asarray(model_or_scale, float)
    return p * np.exp(lq).mean(axis=1) * scale


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1.0) * d)))


def per_day_scores(p: np.ndarray, lq: np.ndarray, y: np.ndarray, scale) -> dict:
    """Per-day losses (arrays) used for paired comparisons. Event definition: rain >= threshold (>=, not >)."""
    quants = {t: mixture_quantile(p, lq, t) for t in LEVELS}
    pin = np.mean([np.maximum(t * (y - quants[t]), (t - 1.0) * (y - quants[t])) for t in LEVELS], axis=0)
    out = {"pinball": pin, "abs_median": np.abs(quants[0.5] - y),
           "abs_mean": np.abs(mean_estimate(scale, p, lq) - y)}
    for T in THRESHOLDS:
        out[f"brier_{T:g}"] = (exceed_prob(p, lq, T) - (y >= T).astype(float)) ** 2
    return out


def paired_bootstrap(diff: np.ndarray, block: int = 7, B: int = 2000, seed: int = 0) -> dict:
    """Circular moving-block bootstrap of the mean of a chronologically ordered per-day difference series
    (blocks keep short-range weather persistence). Negative mean = the first method is better."""
    diff = np.asarray(diff, float)
    n = len(diff)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, nb))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(B, nb * block)[:, :n]
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"n_days": n, "mean_diff": float(diff.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi),
            "share_first_better": float((means < 0).mean()), "block_days": block, "resamples": B,
            "significant_at_95": bool(lo > 0 or hi < 0)}


# --------------------------------------------------------------------------- metrics
def prob_metrics(p: np.ndarray, lq: np.ndarray, y: np.ndarray, scale, clim: dict | None = None,
                 clim_flat: dict | None = None) -> dict:
    out: dict = {"n": int(len(y)), "mean_obs_mm": float(y.mean())}
    q_loss = []
    quants = {t: mixture_quantile(p, lq, t) for t in LEVELS}
    for t in LEVELS:
        q_loss.append(pinball(y, quants[t], t))
    out["pinball_mean"] = float(np.mean(q_loss))
    # One-sided calibration. Plain interval "coverage" is inflated by the point mass at 0 (a dry day
    # gives the interval [0, 0], which trivially contains an observed 0), so instead report how often
    # the observation EXCEEDS each predicted quantile: <= 1 - level when calibrated (equal to it for
    # a continuous distribution; lower where dry days sit exactly on the quantile).
    for t in (0.5, 0.75, 0.9):
        out[f"exceed_q{int(t * 100)}"] = float(np.mean(y > quants[t] + 1e-9))
    wetday = y >= WET_MM
    if wetday.sum() >= 10:   # spread check on rainy days only, where the distribution is continuous
        lo, hi = np.exp(lq[wetday, 3]), np.exp(lq[wetday, 15])      # cond. 20th / 80th percentile
        out["wetday_cover_60"] = float(np.mean((y[wetday] >= lo) & (y[wetday] <= hi)))
    # calibrated MEDIAN as the point forecast
    out["median_MAE_mm"] = float(np.mean(np.abs(quants[0.5] - y)))
    out["median_bias_mm"] = float(np.mean(quants[0.5] - y))
    out["median_RMSE_mm"] = float(np.sqrt(np.mean((quants[0.5] - y) ** 2)))
    # calibrated EXPECTED TOTAL as the point forecast
    mean_p = mean_estimate(scale, p, lq)
    out["mean_est_MAE_mm"] = float(np.mean(np.abs(mean_p - y)))
    out["mean_est_bias_mm"] = float(np.mean(mean_p - y))
    out["mean_est_RMSE_mm"] = float(np.sqrt(np.mean((mean_p - y) ** 2)))
    for T in THRESHOLDS:
        pe = exceed_prob(p, lq, T)
        ev = (y >= T).astype(float)              # event: rainfall >= T (inclusive)
        out[f"brier_{T:g}"] = float(np.mean((pe - ev) ** 2))
        out[f"mean_p_{T:g}"] = float(pe.mean())
        out[f"freq_{T:g}"] = float(ev.mean())
        if 0 < ev.sum() < len(ev):
            out[f"auc_{T:g}"] = float(roc_auc_score(ev, pe))
    for ref, tag in ((clim, ""), (clim_flat, "_vs_flat")):
        if not ref:
            continue
        if ref["pinball_mean"] > 0:
            out["pinball_skill_vs_clim" + tag] = 1.0 - out["pinball_mean"] / ref["pinball_mean"]
        for T in THRESHOLDS:
            if ref.get(f"brier_{T:g}", 0) > 0:
                out[f"BSS_{T:g}{tag}"] = 1.0 - out[f"brier_{T:g}"] / ref[f"brier_{T:g}"]
    return out


def raw_metrics(fc: np.ndarray, y: np.ndarray) -> dict:
    err = fc - y
    out = {"n": int(len(y)), "mean_obs_mm": float(y.mean()), "MAE_mm": float(np.abs(err).mean()),
           "bias_fc_minus_obs_mm": float(err.mean()), "RMSE_mm": float(np.sqrt((err ** 2).mean()))}
    for T in THRESHOLDS:
        ev, hit = (y >= T), (fc >= T)
        out[f"brier_{T:g}"] = float(np.mean((hit.astype(float) - ev.astype(float)) ** 2))  # 0/1 forecast
        if 0 < ev.sum() < len(ev) and len(np.unique(fc)) > 1:
            out[f"auc_{T:g}"] = float(roc_auc_score(ev, fc))
    return out


# ----------------------------------------------------------------------- data prep
MIN_COMMON = 200          # minimum number of dates shared by all compared candidates


class CalibrationError(RuntimeError):
    pass


def _set_frame(wide: pd.DataFrame, lead: str, cols: list[str]) -> pd.DataFrame:
    d = wide[wide.lead_group == lead]
    ok = d.observed_precip_mm.notna()
    if "clim_nino34" in d.columns:            # every row must have its climate regressors available
        ok &= d["clim_nino34"].notna() & d["clim_iod"].notna()
    for c in cols:
        ok &= d[f"eligible_{c}"].astype(bool) & d[f"fc_{c}_mm"].notna()
    d = d[ok].sort_values("label_date_local").reset_index(drop=True)
    d["label_date_local"] = pd.to_datetime(d["label_date_local"])
    return d


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], C: float,
                 combine: str | None = None, season: bool = True, climate: str | None = None):
    m = HurdleModel(C=C).fit(make_X(train, cols, combine, season, climate),
                             train.observed_precip_mm.to_numpy(float))
    p, lq = m.parts(make_X(test, cols, combine, season, climate))
    return m, p, lq


def frame_fingerprint(df: pd.DataFrame) -> tuple[int, str]:
    """(rows, sha256) over the exact station, accumulation window and observed target of every row."""
    import hashlib
    key = (df.station_id.astype(str) + "|" + pd.to_datetime(df.window_start_utc).astype(str) + "|" +
           pd.to_datetime(df.window_end_utc).astype(str) + "|" + df.observed_precip_mm.round(6).astype(str))
    return len(df), hashlib.sha256("\n".join(sorted(key)).encode()).hexdigest()


def _row_keys(df: pd.DataFrame) -> pd.Series:
    return (df.station_id.astype(str) + "|" + pd.to_datetime(df.window_start_utc).astype(str) + "|" +
            pd.to_datetime(df.window_end_utc).astype(str) + "|" + df.observed_precip_mm.round(6).astype(str))


@dataclass
class LeadResult:
    lead: str
    eval_first: str = ""
    eval_last: str = ""
    n_eval: int = 0
    cv: list[dict] = field(default_factory=list)
    holdout: list[dict] = field(default_factory=list)
    paired: list[dict] = field(default_factory=list)
    selected: dict | None = None
    ranking: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    excluded_candidates: list[str] = field(default_factory=list)
    cv_folds: list[str] = field(default_factory=list)


def _candidates(wide: pd.DataFrame, lead: str, only: tuple | None = None) -> dict[str, dict]:
    out = {}
    for name, spec in FEATURE_SETS.items():
        if only and name not in only:
            continue
        if lead in spec["leads"] and all(f"fc_{c}_mm" in wide.columns for c in spec["cols"]):
            df = _set_frame(wide, lead, spec["cols"])
            if len(df) >= MIN_TRAIN + 60:
                out[name] = {"cols": spec["cols"], "combine": spec.get("combine"), "df": df}
    return out


def _pick_common(cands: dict) -> tuple[dict, set, list[str]]:
    """Keep the largest group of candidates that share >= MIN_COMMON dates; report the rest as excluded
    (with disjoint or barely overlapping periods there is no fair common evaluation set)."""
    names, excluded = list(cands), []

    def inter(ns):
        return set.intersection(*[set(cands[n]["df"].label_date_local) for n in ns])

    common = inter(names)
    while len(common) < MIN_COMMON and len(names) > 1:
        drop = max(names, key=lambda d: len(inter([n for n in names if n != d])))
        names.remove(drop)
        excluded.append(drop)
        common = inter(names)
    return {n: cands[n] for n in names}, common, excluded


def evaluate_lead(wide: pd.DataFrame, lead: str, only: tuple | None = None,
                  use_climate: bool | None = None) -> tuple[LeadResult, dict]:
    res = LeadResult(lead=lead)
    if use_climate is None:
        use_climate = "clim_nino34" in wide.columns
    elif not use_climate:
        wide = wide.drop(columns=[c for c in wide.columns if c.startswith("clim_")])
    cands = _candidates(wide, lead, only)
    if not cands:
        res.notes.append("no feature set has enough eligible rows")
        return res, {}
    cands, common, res.excluded_candidates = _pick_common(cands)
    if res.excluded_candidates:
        res.notes.append(f"excluded from selection (too little date overlap with the others): {res.excluded_candidates}")
    if len(common) < MIN_COMMON:
        res.notes.append(f"only {len(common)} common dates; need {MIN_COMMON}")
        return res, {}
    dates = np.array(sorted(common))
    n = len(dates)
    hold_start = dates[int(n * (1 - HOLDOUT_FRAC))]
    cv_cuts = [dates[int(n * f)] for f in (0.35, 0.5, 0.65)]
    cv_ends = cv_cuts[1:] + [hold_start]
    res.eval_first, res.eval_last, res.n_eval = str(dates[0].date()), str(dates[-1].date()), n
    gap = pd.Timedelta(days=GAP_DAYS)

    def split(df, lo, hi):
        train = df[df.label_date_local < lo - gap]
        test = df[(df.label_date_local >= lo) & (df.label_date_local < hi) & df.label_date_local.isin(common)]
        return train, test

    def score(name, C, season, climate, lo, hi, tag, clim_override=None):
        """One fit + evaluation; returns None when the fold is unusable (too little data, or too few wet/dry days)."""
        v = cands[name]
        train, test = split(v["df"], lo, hi)
        if len(train) < MIN_TRAIN or len(test) < 20:
            return None
        try:
            m, p, lq = _fit_predict(train, test, v["cols"], C, v["combine"], season, climate)
            ytr = train.observed_precip_mm.to_numpy(float)
            pf, lqf, sf = climatology_flat(ytr, len(test))
            ps, lqs, ss = climatology_seasonal(ytr, train.label_date_local, test.label_date_local)
        except ValueError:
            return None
        y = test.observed_precip_mm.to_numpy(float)
        cflat = prob_metrics(pf, lqf, y, sf)
        cseas = clim_override or prob_metrics(ps, lqs, y, ss)
        pm = prob_metrics(p, lq, y, m.tail_scale_, cseas, cflat)
        pm.update(method="hurdle", feature_set=name, C=C, season=season, climate=climate or "none", split=tag,
                  lead=lead, n_train=len(train), train_first=str(train.label_date_local.min().date()),
                  train_last=str(train.label_date_local.max().date()),
                  test_first=str(test.label_date_local.min().date()),
                  test_last=str(test.label_date_local.max().date()))
        return {"pm": pm, "cseas": cseas, "cflat": cflat, "test": test, "model": m, "p": p, "lq": lq, "y": y,
                "clim_parts": {"seasonal": (ps, lqs, ss), "flat": (pf, lqf, sf)}}

    # ---- selection on expanding-window folds inside the pre-holdout period. Every configuration is scored on
    # the SAME folds: a fold that any configuration cannot score is dropped for all of them.
    configs = [(name, C, season, climate) for name in cands
               for climate in (CLIMATE_GRID if use_climate else (None,)) for C in C_GRID for season in SEASON_GRID]
    fold_res = {cfg_: [score(*cfg_, lo, hi, f"cv{i + 1}") for i, (lo, hi) in enumerate(zip(cv_cuts, cv_ends))]
                for cfg_ in configs}
    while configs:
        ok_folds = [i for i in range(len(cv_cuts)) if all(fold_res[c][i] is not None for c in configs)]
        if ok_folds:
            break
        worst = min(configs, key=lambda c: sum(r is not None for r in fold_res[c]))
        configs.remove(worst)
        res.notes.append(f"configuration {worst[:3]} dropped: it could not be scored on folds the others could")
    if not configs:
        res.notes.append("cross-validation produced no usable fold")
        return res, {}
    res.cv_folds = [f"cv{i + 1}" for i in ok_folds]
    table = []
    for cfg_ in configs:
        rows = [fold_res[cfg_][i]["pm"] for i in ok_folds]
        res.cv.extend(rows)
        w = np.array([r["n"] for r in rows], float)
        table.append({"feature_set": cfg_[0], "C": cfg_[1], "season": cfg_[2], "climate": cfg_[3],
                      "cv_pinball": float(np.average([r["pinball_mean"] for r in rows], weights=w)),
                      "cv_brier_1": float(np.average([r["brier_1"] for r in rows], weights=w)),
                      "cv_n": int(w.sum()), "cv_folds": ",".join(res.cv_folds)})
    table.sort(key=lambda r: r["cv_pinball"])
    res.ranking = table
    best = table[0]
    res.selected = best

    # ---- one-shot evaluation on the untouched holdout, for the winner AND every reference
    end = pd.Timestamp.max
    win = score(best["feature_set"], best["C"], best["season"], best["climate"], hold_start, end, "holdout")
    if win is None:
        res.notes.append("the selected configuration could not be scored on the holdout")
        return res, {}
    test, p, lq, y = win["test"], win["p"], win["lq"], win["y"]
    # identical-rows verification: every compared method must be scored on the same station / accumulation
    # windows / observed targets. Recompute on the common intersection if the row sets ever differ.
    frames = {"hurdle": test}
    raw_frames = {}
    for name, v in cands.items():
        t2 = v["df"][v["df"].label_date_local.isin(test.label_date_local)]
        for c in v["cols"]:
            raw_frames[f"raw:{c}"] = (t2, c, name)
    big = max(cands.values(), key=lambda v: len(v["cols"]))
    if len(big["cols"]) > 1:
        raw_frames["raw:equal_weight_blend"] = (big["df"][big["df"].label_date_local.isin(test.label_date_local)],
                                                None, "consensus")
    key_sets = {"hurdle": set(_row_keys(test))}
    key_sets.update({k: set(_row_keys(f[0])) for k, f in raw_frames.items()})
    common_keys = set.intersection(*key_sets.values())
    identical = all(ks == common_keys for ks in key_sets.values())
    if not identical:
        res.notes.append(f"holdout rows differed between methods; recomputed on the {len(common_keys)}-row intersection")
    mask = _row_keys(test).isin(common_keys).to_numpy()
    test, p, lq, y = test[mask].reset_index(drop=True), p[mask], lq[mask], y[mask]
    n_rows, fp = frame_fingerprint(test)
    ps, lqs, ss = (a[mask] if isinstance(a, np.ndarray) else a for a in win["clim_parts"]["seasonal"])
    pf, lqf, sf = win["clim_parts"]["flat"]
    pf, lqf = pf[mask], lqf[mask]
    cflat = prob_metrics(pf, lqf, y, sf)
    cseas = prob_metrics(ps, lqs, y, ss)
    pm = prob_metrics(p, lq, y, win["model"].tail_scale_, cseas, cflat)
    pm.update(win["pm"] | {k: v for k, v in pm.items()})
    pm.update(n=int(len(y)), keys_identical_across_methods=bool(identical), key_fingerprint=fp)
    res.holdout.append(pm)
    for cm, name in ((cseas, "climatology"), (cflat, "climatology_flat")):
        res.holdout.append(dict(cm, method=name, split="holdout", lead=lead, feature_set="-",
                                n_train=pm["n_train"], keys_identical_across_methods=bool(identical),
                                key_fingerprint=fp, test_first=pm["test_first"], test_last=pm["test_last"]))
    raw_series = {}
    for label, (f, c, name) in raw_frames.items():
        f = f[_row_keys(f).isin(common_keys)].sort_values("label_date_local").reset_index(drop=True)
        fc = f[f"fc_{c}_mm"].to_numpy(float) if c else np.mean([f[f"fc_{k}_mm"].to_numpy(float) for k in big["cols"]], axis=0)
        assert (f.observed_precip_mm.to_numpy(float) == y).all(), "observed targets differ between methods"
        rm = raw_metrics(fc, y)
        rm.update(method=label, split="holdout", lead=lead, feature_set=name, keys_identical_across_methods=bool(identical),
                  key_fingerprint=fp, test_first=pm["test_first"], test_last=pm["test_last"])
        if not any(r.get("method") == label for r in res.holdout):
            res.holdout.append(rm)
        raw_series[label] = fc
    # alternatives (transparency only; not used for selection), scored against the WINNER's seasonal climatology
    for r in table[1:4]:
        o2 = score(r["feature_set"], r["C"], r["season"], r["climate"], hold_start, end, "holdout", clim_override=cseas)
        if o2 is not None and len(o2["y"]) == len(win["y"]):
            o2["pm"]["method"] = "hurdle(alt)"
            res.holdout.append(o2["pm"])
    if use_climate:
        for tag, want in (("hurdle:no_climate", lambda r: r["climate"] is None),
                          ("hurdle:climate", lambda r: r["climate"] is not None)):
            cand_rows = [r for r in table if r["feature_set"] == best["feature_set"] and want(r)]
            if cand_rows:
                r = cand_rows[0]
                o3 = score(r["feature_set"], r["C"], r["season"], r["climate"], hold_start, end, "holdout", clim_override=cseas)
                if o3 is not None:
                    o3["pm"]["method"] = tag
                    res.holdout.append(o3["pm"])
    # ---- paired differences with block-bootstrap uncertainty (blocks of 7 consecutive days)
    dc = per_day_scores(p, lq, y, win["model"].tail_scale_)
    dsea = per_day_scores(ps, lqs, y, ss)
    dfl = per_day_scores(pf, lqf, y, sf)

    def add(comp, a, b, unit):
        d = paired_bootstrap(a - b)
        res.paired.append(dict(lead=lead, comparison=comp, unit=unit, first_minus_second=True, **d))

    for label, fc in raw_series.items():
        ae = np.abs(fc - y)
        add(f"MAE: calibrated median  minus  {label}", dc["abs_median"], ae, "mm")
        add(f"MAE: calibrated expected total  minus  {label}", dc["abs_mean"], ae, "mm")
    add("Brier (event rain >= 0.2 mm): calibrated minus seasonal climatology", dc["brier_0.2"], dsea["brier_0.2"], "Brier")
    add("Brier (event rain >= 0.2 mm): calibrated minus flat climatology", dc["brier_0.2"], dfl["brier_0.2"], "Brier")
    add("pinball loss: calibrated minus seasonal climatology", dc["pinball"], dsea["pinball"], "mm")
    res.paired.append(dict(lead=lead, comparison="Brier: calibrated vs raw model", unit="Brier", n_days=len(y),
                           note="UNAVAILABLE: raw model forecasts are amounts in mm, not probabilities, so no "
                                "comparable probability score exists"))

    # ---- refit on ALL eligible rows for deployment (best configuration per feature set); the ranking order is
    # the fallback routing order. A set whose final fit fails is left out of the routing.
    def refit(r):
        v = cands[r["feature_set"]]
        X = make_X(v["df"], v["cols"], v["combine"], r["season"], r["climate"])
        m = HurdleModel(C=r["C"]).fit(X, v["df"].observed_precip_mm.to_numpy(float))
        crange = {c: (float(v["df"][c].min()), float(v["df"][c].max())) for c in CLIMATE_COLS.get(r["climate"], [])}
        return {"model": m, "cols": v["cols"], "combine": v["combine"], "season": r["season"],
                "climate": r["climate"], "clim_range": crange, "C": r["C"], "n_train": len(v["df"]),
                "feature_schema": feature_names(v["cols"], v["combine"], r["season"], r["climate"]),
                "train_first": str(v["df"].label_date_local.min().date()),
                "train_last": str(v["df"].label_date_local.max().date())}

    fits, seen = {}, set()
    for r in table:
        name = r["feature_set"]
        if name in seen:
            continue
        seen.add(name)
        try:
            fits[name] = refit(r)
        except ValueError as exc:
            res.notes.append(f"final fit of {name} failed ({exc}); left out of the routing")
    if best["feature_set"] not in fits:
        res.notes.append("the selected configuration could not be refitted on all data")
        return res, {}
    top = fits[best["feature_set"]]
    bundle = {"model": top["model"], "cols": top["cols"], "feature_set": best["feature_set"], "C": best["C"],
              "season": best["season"], "combine": top["combine"],
              "lead": lead, "n_train": top["n_train"], "train_first": top["train_first"],
              "train_last": top["train_last"], "fits": fits,
              "routing_order": [n for n in dict.fromkeys(r["feature_set"] for r in table) if n in fits],
              "climate": best["climate"],
              "holdout": [r for r in res.holdout if r.get("method") == "hurdle"][:1],
              "ranking": table}
    return res, bundle


# ----------------------------------------------------------------------------- run
def run_calibration(cfg, wide_path: Path | None = None, use_climate: bool = False) -> dict:
    import os
    import shutil
    src = Path(wide_path or (cfg.joined_dir / "training_wide.parquet"))
    wide = pd.read_parquet(src)
    indices = None
    if use_climate:
        from . import climate
        from .pipeline import make_fetcher
        indices = climate.load_indices(make_fetcher(cfg), cfg.raw_dir)
        wide = climate.attach_climate(wide, indices)
    else:                                       # the default run can never see climate columns
        wide = wide.drop(columns=[c for c in wide.columns if c.startswith("clim_")])
    from . import enso_guard, provenance
    enso_status = enso_guard.EXPERIMENTAL if use_climate else enso_guard.NO_ENSO
    final_dir = Path(cfg.data_dir) / ("models_enso_experimental" if use_climate else "models")
    meta = provenance.make_run_meta(cfg, enso_status, training_wide_path=src)
    # Build EVERYTHING in a staging directory; the live artifacts are touched only after all leads have been
    # fitted, written and validated. Any failure leaves the existing artifacts exactly as they were.
    stage = Path(cfg.data_dir) / f"staging_{meta['run_id']}"
    stage.mkdir(parents=True)
    rows, paired, sel, manifest = [], [], {}, []
    try:
        for lead in ("day1", "day2", "day3"):
            res, bundle = evaluate_lead(wide, lead, use_climate=use_climate)
            sel.setdefault("_notes", {})[lead] = res.notes
            for note in res.notes:
                log.warning("%s %s: %s", cfg.location_name, lead, note)
            if not bundle:
                raise CalibrationError(f"{cfg.location_name} {lead}: no model could be built ({res.notes})")
            bundle["enso_status"] = enso_status
            bundle["meta"] = dict(meta, lead=lead)
            if enso_status == enso_guard.NO_ENSO:
                enso_guard.assert_no_enso(bundle, f"{cfg.location_name} {lead}", expected_lead=lead)
            path = stage / f"hurdle_{lead}.joblib"
            joblib.dump(bundle, path)
            manifest.append({"lead": lead, "file": path.name, "sha256": provenance.sha256_file(path),
                             "selected_feature_set": bundle["feature_set"], "routing_order": bundle["routing_order"],
                             "fits": {k: {"feature_schema": f["feature_schema"], "cols": f["cols"], "C": f["C"],
                                          "season": f["season"], "climate": f["climate"] or "none",
                                          "n_train": f["n_train"], "train_first": f["train_first"],
                                          "train_last": f["train_last"]} for k, f in bundle["fits"].items()}})
            rows += res.cv + res.holdout
            paired += res.paired
            sel[lead] = {"selected": res.selected, "ranking": res.ranking, "eval_first": res.eval_first,
                         "eval_last": res.eval_last, "n_eval_dates": res.n_eval, "notes": res.notes,
                         "cv_folds": res.cv_folds, "excluded_candidates": res.excluded_candidates,
                         "n_train_final": bundle["n_train"], "train_first": bundle["train_first"],
                         "train_last": bundle["train_last"]}
            log.info("%s %s: selected %s C=%s season=%s climate=%s (cv pinball %.4f on %s), holdout n=%s", cfg.location_name,
                     lead, res.selected["feature_set"], res.selected["C"], res.selected["season"],
                     res.selected["climate"], res.selected["cv_pinball"], ",".join(res.cv_folds),
                     next((r["n"] for r in res.holdout if r.get("method") == "hurdle"), None))
        # ---- ENSO/IOD test on the LONG record (experimental mode only)
        if indices is not None:
            for lead in ("day2", "day3"):
                r2, _ = evaluate_lead(wide, lead, only=("jma_gsm_pr",), use_climate=True)
                for r in r2.holdout:
                    if str(r.get("method")).startswith(("hurdle", "climatology")):
                        rows.append(dict(r, lead=f"{lead}[jma_gsm long record]"))
            sel["_climate_latest"] = {"readings": climate.latest_reading(indices),
                                      "phase": climate.describe_phase(indices["nino34"].value.iloc[-1],
                                                                      indices["iod"].value.iloc[-1])}
            obs_path = Path(cfg.data_dir) / "clean" / "observations.parquet"
            if obs_path.exists():
                obs_all = pd.read_parquet(obs_path)
                sel["_enso_relationship"] = climate.enso_rain_relationship(obs_all, indices).to_dict("records")
                yt, yg = climate.enso_year_table(obs_all, indices)
                sel["_enso_years"], sel["_enso_groups"] = yt.to_dict("records"), yg.to_dict("records")
        sel["_enso_status"] = ("experimental: UNVERIFIED feature included in the search" if indices is not None
                               else "not used: feature UNVERIFIED (see reports/enso_source_verification.md)")
        sel["_run"] = {k: meta[k] for k in ("run_id", "created_utc", "enso_status", "code_sha256")}
        (stage / "selection.json").write_text(json.dumps(sel, indent=1, default=str))
        (stage / "manifest.json").write_text(json.dumps({"meta": meta, "artifacts": manifest}, indent=1, default=str))
        # validate what was written (reload from disk) before anything live is touched
        if enso_status == enso_guard.NO_ENSO:
            for lead in ("day1", "day2", "day3"):
                enso_guard.assert_no_enso(joblib.load(stage / f"hurdle_{lead}.joblib"), f"staged {lead}", expected_lead=lead)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    # ---- swap: old artifacts are MOVED aside (never overwritten), then the validated ones move in
    prev = provenance.move_aside(final_dir, provenance.archive_root(cfg) / "superseded_models" / Path(cfg.data_dir).name,
                                 f"{final_dir.name}_before_{meta['run_id']}")
    if prev:
        log.info("%s: previous artifacts moved to %s", cfg.location_name, prev)
    final_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(stage.iterdir()):
        os.replace(f, final_dir / f.name)
    stage.rmdir()
    from .predict import load_bundles, routing_table
    suffix = "_enso_experimental" if use_climate else ""
    rt = routing_table(load_bundles(cfg, experimental=use_climate))     # reloads + re-validates the final files
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    rt.to_csv(cfg.reports_dir / f"routing_table{suffix}.csv", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(cfg.reports_dir / f"calibration_metrics{suffix}.csv", index=False)
    pd.DataFrame(paired).to_csv(cfg.reports_dir / f"paired_differences{suffix}.csv", index=False)
    from .calibrate_report import write_calibration_report
    write_calibration_report(cfg, metrics, sel, wide, suffix=suffix, paired=pd.DataFrame(paired))
    return sel
