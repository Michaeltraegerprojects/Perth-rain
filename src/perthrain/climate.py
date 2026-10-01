"""Climate-driver indices (El Nino / IOD / SOI) from BoM's anonymous FTP.

Source: ``ftp://ftp.bom.gov.au/anon/gen/clim_data/IDCK000081.zip`` (public, updated daily), files
``nino_3.4.csv`` and ``iod.csv`` (weekly SST anomaly indices, columns start_date,end_date,value) and
``soi.csv`` (rolling 30-day Southern Oscillation Index). Units: degrees C anomaly (Nino 3.4, IOD),
SOI index units.

Leakage rule: a row dated D only sees the latest index value whose averaging period ENDED at least
``LAG_DAYS`` before the start of the gauge window. Weekly indices are published with a delay, so a
14-day lag is conservative and avoids using values that were not yet available.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "ftp://ftp.bom.gov.au/anon/gen/clim_data/IDCK000081.zip"
FILES = {"nino34": "nino_3.4.csv", "iod": "iod.csv", "soi": "soi.csv"}
LAG_DAYS = 14
UNITS = {"nino34": "degC anomaly (weekly)", "iod": "degC anomaly (weekly)", "soi": "index (30-day)"}


def parse_indices(blob: bytes) -> dict[str, pd.DataFrame]:
    out = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for key, fn in FILES.items():
            df = pd.read_csv(io.BytesIO(zf.read(fn)))
            val = [c for c in df.columns if c not in ("start_date", "end_date")][0]
            out[key] = pd.DataFrame({
                "start": pd.to_datetime(df.start_date.astype(str), format="%Y%m%d"),
                "end": pd.to_datetime(df.end_date.astype(str), format="%Y%m%d"),
                "value": pd.to_numeric(df[val], errors="coerce")}).dropna().sort_values("end").reset_index(drop=True)
    return out


def load_indices(fetcher, raw_dir: Path) -> dict[str, pd.DataFrame]:
    res = fetcher.get_file(URL, Path(raw_dir) / "bom_ftp" / "climate" / "IDCK000081.zip", refresh=True,
                           label="BoM climate drivers IDCK000081")
    if not res.ok:
        raise RuntimeError(f"cannot download climate indices: {res.error}")
    return parse_indices(res.body)


def lagged_values(asof: pd.Series, series: pd.DataFrame, lag_days: int = LAG_DAYS) -> pd.DataFrame:
    """Latest index value whose period ended on/before ``asof - lag_days``."""
    cutoff = (pd.to_datetime(asof) - pd.Timedelta(days=lag_days)).to_numpy()
    idx = np.searchsorted(series.end.to_numpy(), cutoff, side="right") - 1
    ok = idx >= 0
    safe = np.where(ok, idx, 0)
    value = np.where(ok, series.value.to_numpy()[safe], np.nan)
    end = pd.to_datetime(np.where(ok, series.end.to_numpy()[safe], np.datetime64("NaT")))
    age = (pd.to_datetime(asof).to_numpy() - end.to_numpy()) / np.timedelta64(1, "D")
    return pd.DataFrame({"value": value, "end": end, "age_days": age}, index=getattr(asof, "index", None))


def attach_climate(df: pd.DataFrame, indices: dict[str, pd.DataFrame], date_col: str = "label_date_local",
                   lag_days: int = LAG_DAYS) -> pd.DataFrame:
    """Add ``clim_<name>`` (+ ``clim_<name>_age_days``) columns. The window of label date D starts on D-1."""
    out = df.copy()
    asof = pd.to_datetime(out[date_col]) - pd.Timedelta(days=1)
    for name, series in indices.items():
        lv = lagged_values(asof, series, lag_days)
        out[f"clim_{name}"] = lv["value"].to_numpy()
        out[f"clim_{name}_age_days"] = lv["age_days"].to_numpy()
    return out


def latest_reading(indices: dict[str, pd.DataFrame]) -> dict:
    r = {}
    for name, s in indices.items():
        last = s.iloc[-1]
        r[name] = {"value": float(last.value), "period_end": str(last.end.date()), "units": UNITS[name],
                   "max_in_record": float(s.value.max()), "record_start": str(s.start.iloc[0].date())}
    return r


def describe_phase(nino34: float, iod: float | None = None) -> str:
    """Describes the NUMBERS only (+/-0.5 C Nino3.4 and +/-0.4 C IOD thresholds). This is not BoM's official
    ENSO/IOD status, which also weighs atmospheric indicators and (since Sep 2025) uses the relative index."""
    if nino34 >= 0.5:
        enso = f"Nino3.4 anomaly {nino34:+.2f} C, above the +0.5 C warm threshold"
    elif nino34 <= -0.5:
        enso = f"Nino3.4 anomaly {nino34:+.2f} C, below the -0.5 C cool threshold"
    else:
        enso = f"Nino3.4 anomaly {nino34:+.2f} C, within +/-0.5 C"
    if iod is None:
        return enso
    ind = (f"IOD {iod:+.2f} C, above +0.4 C" if iod >= 0.4 else
           f"IOD {iod:+.2f} C, below -0.4 C" if iod <= -0.4 else f"IOD {iod:+.2f} C, within +/-0.4 C")
    return f"{enso}, {ind}"


def enso_rain_relationship(obs: pd.DataFrame, indices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Descriptive check on the long gauge record: seasonal rainfall vs the index averaged over the
    PRECEDING season (so the index is known in advance) and over the same season."""
    from scipy.stats import spearmanr
    d = obs.dropna(subset=["observed_precip_mm"]).copy()
    d["date"] = pd.to_datetime(d.label_date_local)
    rows = []
    seasons = {"JJA": (6, 8), "SON": (9, 11), "MJJASO (wet season)": (5, 10)}
    for sname, (m0, m1) in seasons.items():
        for name in ("nino34", "iod"):
            s = indices[name]
            xs, ys, xc = [], [], []
            for y in sorted(d.date.dt.year.unique()):
                seg = d[(d.date.dt.year == y) & (d.date.dt.month >= m0) & (d.date.dt.month <= m1)]
                expected = (pd.Timestamp(y, m1, 1) + pd.offsets.MonthEnd(0) - pd.Timestamp(y, m0, 1)).days + 1
                if len(seg) < 0.9 * expected:
                    continue
                start = pd.Timestamp(y, m0, 1)
                prev = s[(s.end >= start - pd.Timedelta(days=90)) & (s.end < start - pd.Timedelta(days=LAG_DAYS))]
                same = s[(s.end >= start) & (s.end <= pd.Timestamp(y, m1, 28))]
                if len(prev) < 8 or len(same) < 8:
                    continue
                ys.append(seg.observed_precip_mm.sum()); xs.append(prev.value.mean()); xc.append(same.value.mean())
            if len(ys) >= 8:
                r1, p1 = spearmanr(xs, ys)
                r2, p2 = spearmanr(xc, ys)
                rows.append({"season": sname, "index": name, "n_years": len(ys),
                             "rho_preceding_season_index": round(float(r1), 2), "p_preceding": round(float(p1), 2),
                             "rho_same_season_index": round(float(r2), 2), "p_same": round(float(p2), 2)})
    return pd.DataFrame(rows)


def enso_year_table(obs: pd.DataFrame, indices: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-year spring (Sep-Nov) rainfall vs the winter (Jun-Aug) Nino 3.4 / IOD average, i.e. an index that
    is known before spring starts, plus a grouped summary by ENSO state (+/-0.5 degC thresholds)."""
    d = obs.dropna(subset=["observed_precip_mm"]).copy()
    d["date"] = pd.to_datetime(d.label_date_local)
    rows = []
    for y in sorted(d.date.dt.year.unique()):
        seg = d[(d.date.dt.year == y) & (d.date.dt.month.between(9, 11))]
        if len(seg) < 0.9 * 91:
            continue
        w0, w1 = pd.Timestamp(y, 6, 1), pd.Timestamp(y, 8, 31)
        row = {"year": y, "SON_rain_mm": round(float(seg.observed_precip_mm.sum()), 0),
               "SON_rain_days_ge_1mm": int((seg.observed_precip_mm >= 1).sum())}
        for name in ("nino34", "iod"):
            s = indices[name]
            v = s[(s.end >= w0) & (s.end <= w1)].value
            row[f"JJA_{name}"] = round(float(v.mean()), 2) if len(v) >= 8 else np.nan
        rows.append(row)
    t = pd.DataFrame(rows).dropna(subset=["JJA_nino34"])
    if t.empty:
        return t, pd.DataFrame()
    t["JJA_ENSO_state"] = np.where(t.JJA_nino34 >= 0.5, "index >= +0.5",
                                   np.where(t.JJA_nino34 <= -0.5, "index <= -0.5", "within +/-0.5"))
    g = t.groupby("JJA_ENSO_state").agg(years=("year", "count"), mean_SON_rain_mm=("SON_rain_mm", "mean"),
                                        min_mm=("SON_rain_mm", "min"), max_mm=("SON_rain_mm", "max"),
                                        mean_rain_days_ge_1mm=("SON_rain_days_ge_1mm", "mean")).round(1).reset_index()
    return t, g
