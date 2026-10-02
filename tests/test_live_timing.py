"""Live lead-time integrity: a previous-runs input may only be used when every model run the offset rule names was
issued and (by the documented latency) published BEFORE the prediction time. The Previous Runs API returns values
for future valid times anyway (filled from the latest run), so the returned data alone cannot be trusted."""
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from perthrain import forecasts
from perthrain import predict as P
from perthrain.config import Config

NOW = pd.Timestamp("2026-09-30T14:56Z")          # 22:56 AWST on 30 Sep


def _pr_body(model: str, leads=(2, 3)):
    times = pd.date_range("2026-09-30T00:00", "2026-10-06T23:00", freq="h")
    return {"latitude": -31.9, "longitude": 115.9, "elevation": 20.0,
            "hourly_units": {"time": "iso8601", **{f"precipitation_previous_day{n}": "mm" for n in leads}},
            "hourly": {"time": times.strftime("%Y-%m-%dT%H:%M").tolist(),
                       **{f"precipitation_previous_day{n}": [0.1] * len(times) for n in leads}}}


@pytest.mark.parametrize("model", ["jma_gsm", "ncep_gfs_global", "ecmwf_ifs025"])
def test_previous_runs_inputs_needing_unissued_runs_are_not_usable(model):
    cfg = Config()
    windows = P.future_windows(NOW, cfg.timezone, 4)                     # labels 2..5 Oct
    pr = forecasts.parse_previous_runs(_pr_body(model), model, [2, 3], "u", {}, "2026-09-30T14:57:00Z")
    fw = P.live_feature_table(cfg, pd.DataFrame(), pr, windows, NOW)
    col = "pr_" + model
    u = fw[fw.col == col].set_index(["label_date_local", "lead_day"])
    # the (synthetic) API response supplied complete hours for every day-2 and day-3 row ...
    assert u[u.index.get_level_values("lead_day").isin([2, 3])].forecast_complete.all()
    # ... but only rows whose named runs existed (+6 h publication) before 14:56Z may be used
    assert u.loc[(date(2026, 10, 2), 2), "usable"] and u.loc[(date(2026, 10, 2), 3), "usable"]
    assert u.loc[(date(2026, 10, 3), 3), "usable"]
    for key in [(date(2026, 10, 3), 2), (date(2026, 10, 4), 2), (date(2026, 10, 4), 3),
                (date(2026, 10, 5), 2), (date(2026, 10, 5), 3)]:
        assert not u.loc[key, "usable"], key
        assert u.loc[key, "timing_reason"].startswith("required run not issued/published"), key


def test_blocked_rows_are_reported_not_served(clean_bundles):
    cfg = Config()
    windows = P.future_windows(NOW, cfg.timezone, 4)
    pr = forecasts.parse_previous_runs(_pr_body("jma_gsm"), "jma_gsm", [2, 3], "u", {}, "2026-09-30T14:57:00Z")
    fw = P.live_feature_table(cfg, pd.DataFrame(), pr, windows, NOW)
    pred = P.predict_windows(clean_bundles, fw, windows, cfg.timezone)
    row = pred[(pred.label_date_local == date(2026, 10, 5)) & (pred.lead_group == "day2")].iloc[0]
    assert row.status == "no_complete_forecast" and row.route == "exhausted"
    assert "pr_jma_gsm" in row.inputs_blocked_by_timing
    ok = pred[pred.status == "ok"]
    assert set(zip(ok.label_date_local, ok.lead_group)) <= {(date(2026, 10, 2), "day2"), (date(2026, 10, 2), "day3"),
                                                          (date(2026, 10, 3), "day3")}


from test_routing import LOCS as _LOCS

ROOT = Path(__file__).resolve().parents[1]
LOCS = [loc for _name, loc in _LOCS]


@pytest.mark.local_data
@pytest.mark.parametrize("loc", LOCS)
def test_every_served_row_used_only_runs_issued_before_the_prediction(loc):
    """Real entry point on the cached inputs of the newest served run: every input of every served row must come
    from runs issued and (est.) published before the prediction time."""
    from test_routing import _cfg
    cfg = _cfg(loc)
    recs = sorted((cfg.data_dir / "predictions").glob("forecast_*.csv"))
    assert recs, f"no served predictions for {loc}"
    now = pd.Timestamp(pd.to_datetime(recs[-1].stem.split("_")[1], format="%Y%m%dT%H%MZ"), tz="UTC")
    r = P.compute_predictions(cfg, now=now, refresh_live=False)
    fw, pred = r["features"], r["pred"]
    from perthrain.calibrate import FEATURE_SETS
    for _, s in pred[pred.status == "ok"].iterrows():
        n = int(s.lead_group[-1])
        for col in FEATURE_SETS[s.feature_set_used]["cols"]:
            f = fw[(fw.label_date_local == s.label_date_local) & (fw.lead_day == n) & (fw.col == col)].iloc[0]
            issue = f.forecast_issue_time_utc if pd.notna(f.forecast_issue_time_utc) else f.issue_time_inferred_max_utc
            assert issue + pd.Timedelta(hours=cfg.publication_latency_hours) <= now, (s.label_date_local, s.lead_group, col)
