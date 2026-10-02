"""Two defects found while refreshing data on 2026-10-02 (beyond the external review of 0de31ee).

N-1: the HTTP cache reused an HTTP 400 "run not available" answer forever, so a run requested before it was
     published stayed "not archived" even after it was published.
N-2: the live timing gate assumed every model publishes 6 h after its run starts. JMA took ~9.6 h; at 13:54 UTC on
     2 Oct the gate accepted inputs credited to JMA's 06 UTC run, which did not exist yet (the API silently filled
     those hours from an older run). Availability must come from the model's own published run list when known.
Written to fail on the code before the fix."""
import json
from datetime import date

import pandas as pd
import pytest

from perthrain import forecasts
from perthrain import predict as P
from perthrain.config import Config
from perthrain.http import Fetcher


class _Resp:
    def __init__(self, status, text):
        self.status_code, self.text, self.headers = status, text, {}


class _Session:
    def __init__(self, answers):
        self.answers, self.calls, self.headers = list(answers), 0, {}

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return self.answers.pop(0)


def _age_cached_failure(tmp_path, hours):
    meta = next(tmp_path.rglob("*.meta.json"))
    m = json.loads(meta.read_text())
    m["retrieved_at_utc"] = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta.write_text(json.dumps(m))


def test_stale_run_not_available_answer_is_asked_again(tmp_path):
    f = Fetcher(tmp_path, rate_per_second=0)
    f.session = _Session([_Resp(400, '{"error":true,"reason":"The requested model run is not available."}'),
                          _Resp(200, '{"hourly": {"time": [], "precipitation": []}}')])
    url, params = "https://example.invalid/single", {"run": "2026-09-30T12:00"}
    assert f.get_json(url, params, "sub").status == 400
    _age_cached_failure(tmp_path, hours=30)
    r = f.get_json(url, params, "sub")
    assert r.status == 200 and f.session.calls == 2, "a day-old 'not available' answer must not be trusted"


def test_fresh_failure_is_still_cached(tmp_path):
    f = Fetcher(tmp_path, rate_per_second=0)
    f.session = _Session([_Resp(400, '{"error":true,"reason":"The requested model run is not available."}')])
    url, params = "https://example.invalid/single", {"run": "2026-09-30T12:00"}
    f.get_json(url, params, "sub")
    assert f.get_json(url, params, "sub").status == 400 and f.session.calls == 1


NOW = pd.Timestamp("2026-10-02T13:54Z")


def _jma_body():
    times = pd.date_range("2026-10-01T00:00", "2026-10-07T23:00", freq="h")
    return {"latitude": -31.9, "longitude": 115.9, "elevation": 20.0,
            "hourly_units": {"time": "iso8601", "precipitation_previous_day2": "mm", "precipitation_previous_day3": "mm"},
            "hourly": {"time": times.strftime("%Y-%m-%dT%H:%M").tolist(),
                       "precipitation_previous_day2": [0.1] * len(times),
                       "precipitation_previous_day3": [0.1] * len(times)}}


def _row(fw, label, n):
    return fw[(fw.col == "pr_jma_gsm") & (fw.lead_day == n) & (fw.label_date_local == label)].iloc[0]


def test_input_credited_to_an_unpublished_run_is_not_usable():
    cfg = Config()
    windows = P.future_windows(NOW, cfg.timezone, 4)
    pr = forecasts.parse_previous_runs(_jma_body(), "jma_gsm", [2, 3], "u", {}, "2026-10-02T13:54:10Z")
    meta = {"jma_gsm": {"last_run_init_utc": pd.Timestamp("2026-10-02T00:00Z"), "checked_utc": "2026-10-02T13:54:05Z"}}
    fw = P.live_feature_table(cfg, pd.DataFrame(), pr, windows, NOW, model_meta=meta)
    r = _row(fw, date(2026, 10, 4), 2)               # named run 2 Oct 06Z: 6 h rule says yes, JMA says not yet
    assert not r.usable and "not yet published" in r.timing_reason
    assert r.availability_basis == "observed"
    ok = _row(fw, date(2026, 10, 4), 3)              # named run 1 Oct 06Z: published
    assert ok.usable and ok.availability_basis == "observed"


def test_without_model_metadata_the_estimate_is_labelled_as_such():
    cfg = Config()
    windows = P.future_windows(NOW, cfg.timezone, 4)
    pr = forecasts.parse_previous_runs(_jma_body(), "jma_gsm", [2, 3], "u", {}, "2026-10-02T13:54:10Z")
    fw = P.live_feature_table(cfg, pd.DataFrame(), pr, windows, NOW, model_meta={})
    r = _row(fw, date(2026, 10, 4), 3)
    assert r.usable and r.availability_basis == "estimated (initialisation + 6 h)"


def test_served_rows_report_their_availability_basis(clean_bundles):
    cfg = Config()
    windows = P.future_windows(NOW, cfg.timezone, 4)
    pr = forecasts.parse_previous_runs(_jma_body(), "jma_gsm", [2, 3], "u", {}, "2026-10-02T13:54:10Z")
    meta = {"jma_gsm": {"last_run_init_utc": pd.Timestamp("2026-10-02T00:00Z"), "checked_utc": "2026-10-02T13:54:05Z"}}
    fw = P.live_feature_table(cfg, pd.DataFrame(), pr, windows, NOW, model_meta=meta)
    pred = P.predict_windows(clean_bundles, fw, windows, cfg.timezone)
    ok = pred[pred.status == "ok"]
    assert len(ok) and set(ok.availability_basis) <= {"observed", "estimated (initialisation + 6 h)"}
    blocked = pred[(pred.label_date_local == date(2026, 10, 4)) & (pred.lead_group == "day2")].iloc[0]
    assert "pr_jma_gsm" in str(blocked.inputs_blocked_by_timing)
