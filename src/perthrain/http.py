"""Polite, cached, resumable HTTP/FTP fetching with provenance logging.

Every network retrieval is written to ``<raw>/manifest.jsonl`` (URL, parameters,
retrieval time, HTTP status, cache file, sha256). Failures are also appended to
``<raw>/failures.jsonl``. Responses are cached on disk, so re-running the
pipeline never re-downloads a completed request.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cache_key(url: str, params: dict | None) -> str:
    blob = url + "?" + json.dumps(params or {}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


class RateLimiter:
    """Thread-safe minimum interval between requests (global across workers)."""

    def __init__(self, rate_per_second: float):
        self.interval = 1.0 / rate_per_second if rate_per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            time.sleep(delay)


@dataclass
class FetchResult:
    url: str
    params: dict
    status: int | None
    body: Any
    cache_path: Path
    from_cache: bool
    retrieved_at_utc: str | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None


class Fetcher:
    #: a cached HTTP 400 ("run not available") for a RECENT run is trusted for this long, then asked again: the run
    #: may have been published since (a request made before publication must not stay "not archived" forever)
    FAILURE_TTL_HOURS = 6.0
    #: a 400 given more than this long after the latest model time the request covers is final, never re-asked
    RECENT_RUN_HOURS = 48.0

    def __init__(self, raw_dir: Path, *, rate_per_second: float = 1.0, connect_timeout: float = 20,
                 read_timeout: float = 120, max_retries: int = 6, user_agent: str = "perthrain"):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = RateLimiter(rate_per_second)
        self.timeout = (connect_timeout, read_timeout)
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.user_agent = user_agent
        self._log_lock = threading.Lock()
        self.stats = {"network": 0, "cache_hits": 0, "retries": 0, "failures": 0}
        self.offline = False          # True: cache only; a miss returns an error result and never hits the network

    # ------------------------------------------------------------------ logging
    def _append(self, name: str, record: dict) -> None:
        with self._log_lock:
            with open(self.raw_dir / name, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")

    # --------------------------------------------------------------------- JSON
    def get_json(self, url: str, params: dict, subdir: str, *, retry_failed: bool = False,
                 label: str = "", refresh: bool = False) -> FetchResult:
        """GET JSON with an on-disk cache. ``refresh=True`` ignores and overwrites the cache
        (used for live forecasts, whose request parameters repeat while the answer changes)."""
        key = cache_key(url, params)
        folder = self.raw_dir / subdir
        folder.mkdir(parents=True, exist_ok=True)
        body_path = folder / f"{key}.json"
        meta_path = folder / f"{key}.meta.json"

        if meta_path.exists() and not refresh:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cached_ok = meta.get("status") == 200 and body_path.exists()
            # a cached 400 is reused offline (a replay must not depend on the clock), and otherwise unless the caller
            # asks again or it may have been answered before the run was published
            reuse_400 = meta.get("status") == 400 and (
                self.offline or (not retry_failed and not self._maybe_premature(meta, params)))
            if cached_ok or reuse_400:
                self.stats["cache_hits"] += 1
                body = json.loads(body_path.read_text(encoding="utf-8")) if body_path.exists() else None
                return FetchResult(url, params, meta.get("status"), body, body_path, True,
                                   meta.get("retrieved_at_utc"), meta.get("error"))

        if self.offline:
            return FetchResult(url, params, None, None, body_path, False, None, "offline: not in the cache")
        status, text, error, attempts = None, None, None, 0
        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            self.limiter.wait()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                status, text = resp.status_code, resp.text
                if status in RETRYABLE_STATUS:
                    error = f"HTTP {status}"
                    retry_after = resp.headers.get("Retry-After")
                    self._backoff(attempt, label, error, float(retry_after) if retry_after and
                                  retry_after.isdigit() else None)
                    continue
                error = None if status == 200 else f"HTTP {status}: {text[:300]}"
                break
            except (requests.ConnectionError, requests.Timeout) as exc:
                status, error = None, f"{type(exc).__name__}: {str(exc)[:200]}"
                self._backoff(attempt, label, error)
        retrieved = utcnow_iso()
        self.stats["network"] += 1

        meta = {"url": url, "params": params, "status": status, "retrieved_at_utc": retrieved,
                "attempts": attempts, "error": error, "label": label,
                "user_agent": self.user_agent}
        body = None
        if status == 200 and text is not None:
            body_path.write_text(text, encoding="utf-8")
            body = json.loads(text)
            meta["sha256"] = hashlib.sha256(text.encode()).hexdigest()
        elif status == 400 and text is not None:
            # Open-Meteo returns 400 with a JSON reason for unavailable runs; keep it.
            body_path.write_text(text, encoding="utf-8")
        meta["cache_file"] = str(body_path.relative_to(self.raw_dir))
        if status is not None:  # do not persist pure network failures: allow resume
            meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        self._append("manifest.jsonl", meta)
        if error:
            self.stats["failures"] += 1
            self._append("failures.jsonl", meta)
            log.warning("FAILED %s %s -> %s", label, url, error)
        else:
            log.info("downloaded %s (%s attempt%s)", label or url, attempts, "s" if attempts > 1 else "")
        return FetchResult(url, params, status, body, body_path, False, retrieved, error)

    def _maybe_premature(self, meta: dict, params: dict) -> bool:
        """True if a cached 400 is older than FAILURE_TTL_HOURS and was answered within RECENT_RUN_HOURS of the
        latest model time the request covers (an explicit ``run``, or the end of the ``end_date`` day): the run
        may not have been published yet then. A 400 about a long-archived run, or a request without either
        parameter, is final."""
        try:
            got = datetime.fromisoformat(meta["retrieved_at_utc"].replace("Z", "+00:00"))
            if params.get("run"):
                asked = datetime.fromisoformat(str(params["run"])).replace(tzinfo=timezone.utc)
            elif params.get("end_date"):
                asked = datetime.fromisoformat(str(params["end_date"])).replace(tzinfo=timezone.utc) + timedelta(days=1)
            else:
                return False
        except (KeyError, TypeError, ValueError):
            return False
        recent = (got - asked).total_seconds() / 3600 < self.RECENT_RUN_HOURS
        stale = (datetime.now(timezone.utc) - got).total_seconds() / 3600 >= self.FAILURE_TTL_HOURS
        return recent and stale

    def _backoff(self, attempt: int, label: str, error: str, retry_after: float | None = None) -> None:
        if attempt >= self.max_retries:
            return
        self.stats["retries"] += 1
        delay = retry_after if retry_after else min(120.0, 2.0 ** (attempt + 1)) + random.uniform(0, 1)
        log.warning("retry %d for %s after %.1fs (%s)", attempt + 1, label, delay, error)
        time.sleep(delay)

    # ---------------------------------------------------------------------- FTP
    def get_file(self, url: str, dest: Path, *, refresh: bool = False, label: str = "") -> FetchResult:
        """Fetch a (usually FTP) file to ``dest``; cached unless ``refresh``."""
        dest = Path(dest)
        meta_path = dest.with_suffix(dest.suffix + ".meta.json")
        if dest.exists() and meta_path.exists() and not refresh:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.stats["cache_hits"] += 1
            return FetchResult(url, {}, meta.get("status"), dest.read_bytes(), dest, True,
                               meta.get("retrieved_at_utc"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        data, error, attempts = None, None, 0
        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            self.limiter.wait()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
                with urllib.request.urlopen(req, timeout=self.timeout[1]) as resp:
                    data = resp.read()
                error = None
                break
            except Exception as exc:  # urllib raises URLError / socket errors / ftplib errors
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
                if "550" in str(exc):  # FTP "file not found" is not transient
                    break
                self._backoff(attempt, label, error)
        retrieved = utcnow_iso()
        self.stats["network"] += 1
        meta = {"url": url, "params": {}, "status": 200 if data is not None else None,
                "retrieved_at_utc": retrieved, "attempts": attempts, "error": error, "label": label,
                "cache_file": str(dest.relative_to(self.raw_dir)) if dest.is_relative_to(self.raw_dir)
                else str(dest)}
        if data is not None:
            dest.write_bytes(data)
            meta["sha256"] = hashlib.sha256(data).hexdigest()
            meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        self._append("manifest.jsonl", meta)
        if error:
            self.stats["failures"] += 1
            self._append("failures.jsonl", meta)
            log.warning("FAILED %s -> %s", url, error)
        return FetchResult(url, {}, meta["status"], data, dest, False, retrieved, error)

    def list_ftp_dir(self, url: str) -> list[str]:
        """Return names in an FTP directory listing (not cached; small)."""
        self.limiter.wait()
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(req, timeout=self.timeout[1]) as resp:
            text = resp.read().decode("latin-1")
        self.stats["network"] += 1
        self._append("manifest.jsonl", {"url": url, "params": {}, "status": 200,
                                        "retrieved_at_utc": utcnow_iso(), "label": "ftp listing"})
        return [line.split()[-1] for line in text.splitlines() if line.strip()]
