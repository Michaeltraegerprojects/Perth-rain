import numpy as np
import pandas as pd
import pytest

from perthrain import calibrate as C
from perthrain import climate, predict


# --------------------------------------------------------------------- distribution maths
def _parts(p, q_mm):
    """constant conditional quantile grid from a list of 19 mm values"""
    lq = np.tile(np.log(np.asarray(q_mm, float)), (len(p), 1))
    return np.asarray(p, float), lq


GRID_MM = np.linspace(0.4, 20.0, 19)


def test_exceedance_is_monotone_and_bounded_by_p():
    p, lq = _parts([0.6], GRID_MM)
    vals = [C.exceed_prob(p, lq, T)[0] for T in (0.2, 1, 5, 10, 25)]
    assert vals[0] == pytest.approx(0.6)                      # P(>=0.2) is the occurrence probability
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))
    assert vals[-1] == pytest.approx(0.6 * 0.025)             # beyond the modelled 95th percentile
    assert vals[0] <= 1.0 and vals[-1] >= 0.0


def test_mixture_quantile_puts_dry_mass_at_zero():
    p, lq = _parts([0.3], GRID_MM)
    assert C.mixture_quantile(p, lq, 0.5)[0] == 0.0           # 70 % dry -> median is 0
    assert C.mixture_quantile(p, lq, 0.9)[0] > 0.2            # upper quantile is in the wet part
    p2, lq2 = _parts([0.95], GRID_MM)
    q = [C.mixture_quantile(p2, lq2, t)[0] for t in (0.25, 0.5, 0.75, 0.9)]
    assert q == sorted(q) and q[0] > 0


def test_quantile_fit_is_monotone_after_rearrangement():
    rng = np.random.default_rng(0)
    n = 600
    fc = rng.gamma(0.7, 3.0, n)
    y = np.where(rng.random(n) < 1 - np.exp(-fc), rng.gamma(2.0, np.maximum(fc, 0.3)), 0.0)
    X = np.column_stack([np.log1p(fc)])
    m = C.HurdleModel().fit(X, y)
    p, lq = m.parts(X)
    assert (np.diff(lq, axis=1) >= -1e-12).all() and (lq >= np.log(C.WET_MM) - 1e-12).all()
    assert 0 < p.min() and p.max() < 1
    assert p[fc > 5].mean() > p[fc < 0.5].mean()              # more forecast rain -> higher chance


def test_pinball_and_climatology_baseline():
    y = np.array([0.0, 0.0, 1.0, 4.0])
    assert C.pinball(y, np.zeros(4), 0.5) == pytest.approx(0.5 * y.mean())
    p, lq, scale = C.climatology_flat(np.array([0.0] * 6 + [0.4, 1.0, 2.0, 8.0]), 3)
    assert p[0] == pytest.approx(0.4) and lq.shape == (3, len(C.GRID)) and scale > 0


def test_make_X_composite_season_and_climate_columns():
    df = pd.DataFrame({"label_date_local": pd.to_datetime(["2026-06-01", "2026-06-02"]),
                       "fc_a_mm": [0.0, 3.0], "fc_b_mm": [1.0, 3.0],
                       "clim_nino34": [1.0, 1.2], "clim_iod": [0.1, 0.2]})
    assert C.make_X(df, ["a", "b"], None, False).shape == (2, 2)
    X = C.make_X(df, ["a", "b"], "mean_log", True, "nino34_iod")
    assert X.shape == (2, 1 + 2 + 2)
    assert X[1, 0] == pytest.approx(np.log1p(3.0))            # mean of two equal log1p values
    assert list(X[:, -2]) == [1.0, 1.2] and list(X[:, -1]) == [0.1, 0.2]


# ------------------------------------------------------------------------ climate indices
def _series():
    ends = pd.to_datetime(["2026-08-02", "2026-08-09", "2026-08-16", "2026-08-23", "2026-08-30"])
    return pd.DataFrame({"start": ends - pd.Timedelta(days=6), "end": ends, "value": [1.0, 1.5, 2.0, 2.5, 3.0]})


def test_lagged_values_never_use_a_period_ending_inside_the_lag():
    asof = pd.Series(pd.to_datetime(["2026-08-24", "2026-08-30", "2026-09-06", "2026-07-01"]))
    lv = climate.lagged_values(asof, _series(), lag_days=14)
    # 08-24 -> cutoff 08-10 -> week ending 08-09 (1.5); 08-30 -> cutoff 08-16 -> ending 08-16 (2.0);
    # 09-06 -> cutoff 08-23 -> 2.5; 07-01 -> nothing published yet
    assert lv.value.tolist()[:3] == [1.5, 2.0, 2.5] and np.isnan(lv.value.iloc[3])
    assert (lv.age_days.dropna() >= 14).all()


def test_attach_climate_uses_window_start_day():
    df = pd.DataFrame({"label_date_local": pd.to_datetime(["2026-08-31"])})
    out = climate.attach_climate(df, {"nino34": _series()})
    # window of label 08-31 starts 08-30; cutoff = 08-16 -> value 2.0
    assert out.clim_nino34.iloc[0] == 2.0


def test_parse_indices_and_phase_wording():
    import io, zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("nino_3.4.csv", "start_date,end_date,nino_3.4\n20260921,20260927,3.16\n")
        z.writestr("iod.csv", "start_date,end_date,iod\n20260921,20260927,0.76\n")
        z.writestr("soi.csv", "start_date,end_date,soi\n20260830,20260928,-20.6\n")
    ind = climate.parse_indices(buf.getvalue())
    assert ind["nino34"].value.iloc[0] == 3.16 and ind["soi"].end.iloc[0] == pd.Timestamp("2026-09-28")
    # the wording describes the numbers only - no El Nino / La Nina status claim
    txt = climate.describe_phase(3.16, 0.76)
    assert txt == "Nino3.4 anomaly +3.16 C, above the +0.5 C warm threshold, IOD +0.76 C, above +0.4 C"
    assert "El Nino" not in txt and "La Nina" not in txt
    assert climate.describe_phase(0.0, 0.0) == "Nino3.4 anomaly +0.00 C, within +/-0.5 C, IOD +0.00 C, within +/-0.4 C"
    assert climate.describe_phase(-0.8) == "Nino3.4 anomaly -0.80 C, below the -0.5 C cool threshold"


# ------------------------------------------------------------------------- live windows
def test_future_windows_only_include_windows_not_yet_started():
    now = pd.Timestamp("2026-09-30 13:55", tz="UTC")           # 21:55 AWST on 30 Sep
    w = predict.future_windows(now, "Australia/Perth", 3)
    assert w.label_date_local.iloc[0].isoformat() == "2026-10-02"   # label 1 Oct began at 09:00 on 30 Sep
    assert (w.window_start_utc > now).all()
    assert (w.window_end_utc - w.window_start_utc == pd.Timedelta(hours=24)).all()
