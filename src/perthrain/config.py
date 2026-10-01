"""Configuration loading (TOML file + CLI overrides)."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    location_name: str = "Perth"
    latitude: float = -31.95224
    longitude: float = 115.8614
    timezone: str = "Australia/Perth"
    coordinate_source: str = "GeoNames id 2063523 via Open-Meteo Geocoding API"

    station_id: str = ""
    search_radius_km: float = 30.0
    min_completeness: float = 0.95
    cdo_csv_paths: list[str] = field(default_factory=list)

    start: str = "auto"
    end: str = "auto"
    max_years: int = 11
    archive_floor: str = "2016-01-01"
    observations_start: str = "2009-01-01"

    lead_days: list[int] = field(default_factory=lambda: [1, 2, 3])
    previous_runs_models: list[str] = field(
        default_factory=lambda: ["jma_gsm", "ecmwf_ifs025", "ncep_gfs_global", "bom_access_global"])
    single_runs: dict[str, str] = field(
        default_factory=lambda: {"ecmwf_ifs": "2024-03-14", "ncep_gfs_global": "2026-04-02"})
    single_run_hour_utc: int = 12
    publication_latency_hours: float = 6.0
    require_publication_before_window: bool = True

    # "allow_uniform_disaggregation" keeps 3 h/6 h models (flagged in alignment_status);
    # "strict" restricts training rows to models whose native step divides the window.
    alignment_policy: str = "allow_uniform_disaggregation"

    suspicious_daily_mm: float = 100.0
    rain_thresholds_mm: list[float] = field(default_factory=lambda: [0.2, 1.0, 5.0])

    rate_per_second: float = 1.0
    concurrency: int = 2
    connect_timeout_s: float = 20
    read_timeout_s: float = 120
    max_retries: int = 6
    user_agent: str = "perthrain-dataset/0.1 (non-commercial forecast verification research)"

    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    # raw responses are cached once and shared by every configured location
    raw_root: Path = Path("data")

    @property
    def raw_dir(self) -> Path:
        return Path(self.raw_root) / "raw"

    @property
    def clean_dir(self) -> Path:
        return self.data_dir / "clean"

    @property
    def joined_dir(self) -> Path:
        return self.data_dir / "joined"


_SECTIONS = {
    "location": {"name": "location_name", "latitude": "latitude", "longitude": "longitude",
                 "timezone": "timezone", "coordinate_source": "coordinate_source"},
    "station": {"id": "station_id", "search_radius_km": "search_radius_km",
                "min_completeness": "min_completeness", "cdo_csv_paths": "cdo_csv_paths"},
    "period": {"start": "start", "end": "end", "max_years": "max_years",
               "archive_floor": "archive_floor", "observations_start": "observations_start"},
    "forecast": {"lead_days": "lead_days", "previous_runs_models": "previous_runs_models",
                 "single_runs": "single_runs", "single_run_hour_utc": "single_run_hour_utc",
                 "publication_latency_hours": "publication_latency_hours",
                 "require_publication_before_window": "require_publication_before_window"},
    "qc": {"suspicious_daily_mm": "suspicious_daily_mm", "rain_thresholds_mm": "rain_thresholds_mm",
           "alignment_policy": "alignment_policy"},
    "http": {"rate_per_second": "rate_per_second", "concurrency": "concurrency",
             "connect_timeout_s": "connect_timeout_s", "read_timeout_s": "read_timeout_s",
             "max_retries": "max_retries", "user_agent": "user_agent"},
}


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.strip().lower()).strip("_")


def load_configs(path: str | Path | None, only: str | None = None, **overrides) -> list[Config]:
    """One Config per configured location.

    A TOML with a ``[[locations]]`` array yields one Config per entry, each writing to
    ``<data_dir>/<slug>`` and ``<reports_dir>/<slug>``. Without it, a single Config from
    ``[location]`` is returned. ``only`` selects one location by name (case-insensitive).
    CLI overrides such as --lat/--lon always apply to a single location.
    """
    raw: dict = {}
    if path:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
    entries = raw.get("locations") or []
    if only and not entries:
        raise SystemExit(f"--location {only!r} needs a config file with [[locations]] entries")
    if not entries or any(overrides.get(k) is not None for k in ("latitude", "longitude")):
        if entries and any(overrides.get(k) is not None for k in ("location_name", "search_radius_km", "coordinate_source")):
            pass
        return [load_config(path, **overrides)]
    if only:
        entries = [e for e in entries if slug(e.get("name", "")) == slug(only)]
        if not entries:
            raise SystemExit(f"no configured location matches {only!r}")
    out = []
    for entry in entries:
        cfg = load_config(path, **overrides)
        for key, attr in _SECTIONS["location"].items():
            if key in entry:
                setattr(cfg, attr, entry[key])
        for key in ("id", "search_radius_km", "cdo_csv_paths", "min_completeness"):
            if key in entry:
                setattr(cfg, _SECTIONS["station"][key], entry[key])
        if overrides.get("station_id"):
            cfg.station_id = str(overrides["station_id"]).zfill(6)
        else:
            cfg.station_id = str(entry.get("id", "")).strip().zfill(6) if str(entry.get("id", "")).strip() else ""
        s = slug(cfg.location_name)
        cfg.raw_root = Path(cfg.data_dir)
        cfg.data_dir = Path(cfg.data_dir) / s
        cfg.reports_dir = Path(cfg.reports_dir) / s
        out.append(cfg)
    return out


def load_config(path: str | Path | None, **overrides) -> Config:
    cfg = Config()
    if path:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        for section, mapping in _SECTIONS.items():
            for key, attr in mapping.items():
                if key in raw.get(section, {}):
                    setattr(cfg, attr, raw[section][key])
    for attr, value in overrides.items():
        if value is not None:
            setattr(cfg, attr, value)
    cfg.station_id = str(cfg.station_id).strip().zfill(6) if str(cfg.station_id).strip() else ""
    cfg.data_dir = Path(cfg.data_dir)
    cfg.reports_dir = Path(cfg.reports_dir)
    return cfg
