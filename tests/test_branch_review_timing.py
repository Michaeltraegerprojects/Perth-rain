"""Second review of branch fix/audit-0de31ee (2026-10-04), timing and cache defects in the audit fixes themselves.

R-3  the live gate fetched model metadata AFTER the forecast data, so a run published in between was credited, and
     "published by the time metadata was read" was taken as "published by the forecast time".
R-4  in an offline (cache-only) replay a cached HTTP 400 older than 6 h was reported as "not in the cache", so a
     replay depended on the wall clock.
R-5  every cached HTTP 400 was re-requested after 6 h, including definitive answers about long-archived runs.
Written to fail on 58d0730 before the fix."""
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from perthrain import forecasts
from perthrain import predict as P
from perthrain.config import Config
from perthrain.http import Fetcher, FetchResult


class _Resp:
    def __init__(self, status, text):
        self.status_code, self.text, self.headers = status, text, {}


class _Session:
    def __init__(self, answers):
        self.answers, self.calls, self.headers = list(answers), 0, {}

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return self.answers.pop(0)


NOT_AVAILABLE = '{"error":true,"reason":"The requested model run is not available."}'


def _cache_400(tmp_path, params, retrieved):
    f = Fetcher(tmp_path, rate_per_second=0)
    f.session = _Session([_Resp(400, NOT_AVAILABLE)])
    f.get_json("https://example.invalid/single", params, "sub")
    meta = next(tmp_path.rglob("*.meta.json"))
    m = json.loads(meta.read_text())
    m["retrieved_at_utc"] = retrieved.strftime("%Y-%m-%dT%H:%M:%SZ")
    meta.write_text(json.dumps(m))
    return Fetcher(tmp_path, rate_per_second=0)


# ------------------------------------------------------------------------- R-4: offline replay
def test_offline_replay_returns_a_cached_400_whatever_its_age(tmp_path):
    params = {"run": "2026-09-30T12:00"}
    f = _cache_400(tmp_path, params, pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3))
    f.session, f.offline = _Session([]), True
    r = f.get_json("https://example.invalid/single", params, "sub")
    assert r.status == 400 and r.from_cache and f.session.calls == 0


# ------------------------------------------------------------------------- R-5: which 400s expire
def test_400_for_a_long_archived_run_is_definitive(tmp_path):
    params = {"run": "2025-01-01T12:00"}
    f = _cache_400(tmp_path, params, pd.Timestamp("2025-06-01T00:00Z"))   # asked five months after the run
    f.session = _Session([])
    assert f.get_json("https://example.invalid/single", params, "sub").status == 400
    assert f.session.calls == 0, "a 'not available' answer given long after the run is final"


def test_400_for_a_recent_previous_runs_request_still_expires(tmp_path):
    now = pd.Timestamp.now(tz="UTC")
    params = {"start_date": f"{now - pd.Timedelta(days=2):%Y-%m-%d}", "end_date": f"{now + pd.Timedelta(days=3):%Y-%m-%d}"}
    f = _cache_400(tmp_path, params, now - pd.Timedelta(hours=7))
    f.session = _Session([_Resp(200, '{"hourly": {"time": []}}')])
    assert f.get_json("https://example.invalid/single", params, "sub").status == 200 and f.session.calls == 1


# ------------------------------------------------------------------------- R-3: gate order and evidence time
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


def _fw(meta):
    cfg = Config()
    windows = P.future_windows(NOW, cfg.timezone, 4)
    pr = forecasts.parse_previous_runs(_jma_body(), "jma_gsm", [2, 3], "u", {}, "2026-10-02T13:54:10Z")
    return P.live_feature_table(cfg, pd.DataFrame(), pr, windows, NOW, model_meta=meta)


def test_latest_run_published_after_the_forecast_time_is_not_usable():
    # metadata read a minute after the forecast time lists the 06Z run, but it became available 30 s after NOW
    meta = {"jma_gsm": {"last_run_init_utc": pd.Timestamp("2026-10-02T06:00Z"),
                        "last_run_available_utc": NOW + pd.Timedelta(seconds=30), "checked_utc": "2026-10-02T13:55:00Z"}}
    r = _row(_fw(meta), date(2026, 10, 4), 2)          # named run 2 Oct 06Z
    assert not r.usable and "after the forecast time" in r.timing_reason


def test_latest_run_published_before_the_forecast_time_is_usable():
    meta = {"jma_gsm": {"last_run_init_utc": pd.Timestamp("2026-10-02T06:00Z"),
                        "last_run_available_utc": NOW - pd.Timedelta(minutes=5), "checked_utc": "2026-10-02T13:54:02Z"}}
    r = _row(_fw(meta), date(2026, 10, 4), 2)
    assert r.usable and r.availability_basis == "observed"


class _RecordingFetcher:
    def __init__(self):
        self.offline, self.calls = False, []

    def get_json(self, url, params, subdir, **kw):
        self.calls.append((url, self.offline))
        return FetchResult(url, params, None, None, Path("none"), False, None, "offline: not in the cache")


@pytest.mark.parametrize("refresh", [True, False])
def test_model_metadata_is_read_before_the_forecast_data(clean_bundles, monkeypatch, refresh):
    rec = _RecordingFetcher()
    monkeypatch.setattr(P, "make_fetcher", lambda cfg: rec)
    monkeypatch.setattr(P, "station_info", lambda cfg: {"latitude": -31.9, "longitude": 115.9})
    P.compute_predictions(Config(), now=NOW, refresh_live=refresh, bundles=clean_bundles)
    first_data = next(i for i, (u, _) in enumerate(rec.calls) if "/static/meta.json" not in u)
    assert all("/static/meta.json" in u for u, _ in rec.calls[:first_data]) and first_data > 0, \
        "metadata must be read first: a run published while the data is fetched must not be credited"
    assert all(off == (not refresh) for _, off in rec.calls), "a replay must never touch the network"


def test_without_an_availability_time_the_latest_run_is_not_credited():
    class F:
        def get_json(self, url, params, subdir, **kw):
            body = {"last_run_initialisation_time": int(pd.Timestamp("2026-10-02T06:00Z").timestamp())}
            return FetchResult(url, params, 200, body, Path("x"), True, "2026-10-02T13:54:00Z")
    meta = P.fetch_model_meta(F(), refresh=False)
    assert meta["jma_gsm"]["last_run_available_utc"] is None
    r = _row(_fw(meta), date(2026, 10, 4), 2)          # named run 2 Oct 06Z = the latest, publication time unknown
    assert not r.usable and "unknown" in r.timing_reason
