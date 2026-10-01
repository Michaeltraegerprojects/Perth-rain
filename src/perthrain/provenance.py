"""Run identity and provenance for model artifacts and predictions."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha256() -> str:
    """Content hash of the package source (file name + bytes of every .py file, sorted)."""
    h = hashlib.sha256()
    for p in sorted(PKG_DIR.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def git_commit() -> dict:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PKG_DIR, capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PKG_DIR, capture_output=True, text=True,
                                   timeout=10).stdout.strip()
            return {"commit": out.stdout.strip(), "dirty": bool(dirty)}
        return {"commit": None, "reason": (out.stderr or "not a git repository").strip()[:200]}
    except Exception as exc:  # git not installed
        return {"commit": None, "reason": f"git unavailable: {exc}"[:200]}


def archive_root(cfg) -> Path:
    """<project>/archive as a RESOLVED path (raw_root may be relative or absolute)."""
    return Path(cfg.raw_root).resolve().parent / "archive"


def source_data_version(cfg, training_wide_path: Path | None = None) -> dict:
    """Hashes and extents of the clean inputs the models are trained on (the file actually read)."""
    import pandas as pd
    files = {"training_wide": Path(training_wide_path or Path(cfg.joined_dir) / "training_wide.parquet"),
             "paired_long": Path(cfg.joined_dir) / "paired_long.parquet",
             "observations": Path(cfg.clean_dir) / "observations.parquet"}
    out = {}
    for k, p in files.items():
        if p.exists():
            out[k] = {"path": str(p).replace("\\", "/"), "sha256": sha256_file(p), "bytes": p.stat().st_size}
    wide = pd.read_parquet(files["training_wide"])
    out["training_wide"].update(rows=len(wide), columns=len(wide.columns),
                                label_date_first=str(pd.to_datetime(wide.label_date_local).min().date()),
                                label_date_last=str(pd.to_datetime(wide.label_date_local).max().date()))
    summ = Path(cfg.reports_dir) / "run_summary.json"
    if summ.exists():
        s = json.loads(summ.read_text())
        out["pipeline_run_generated_at_utc"] = s.get("generated_at_utc")
        out["station_id"] = (s.get("station") or {}).get("station_id")
    return out


def dependency_versions() -> dict:
    from importlib import metadata
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "pyarrow", "requests", "tzdata"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def make_run_meta(cfg, enso_status: str, kind: str = "calibration", training_wide_path: Path | None = None) -> dict:
    created = datetime.now(timezone.utc)
    code = code_sha256()
    data = source_data_version(cfg, training_wide_path)
    ident = hashlib.sha256((code + data["training_wide"]["sha256"] + enso_status).encode()).hexdigest()[:8]
    prefix = "noenso" if enso_status == "no_enso" else "ensoexp"
    return {"run_id": f"{prefix}-{created:%Y%m%dT%H%M%SZ}-{ident}", "kind": kind,
            "created_utc": created.strftime("%Y-%m-%dT%H:%M:%SZ"), "location": cfg.location_name, "gauge_data_dir": Path(cfg.data_dir).name,
            "enso_status": enso_status, "code_sha256": code, "git": git_commit(), "source_data": data,
            "versions": dependency_versions()}


def move_aside(target_dir: Path, archive_root: Path, label: str) -> Path | None:
    """If ``target_dir`` holds files, MOVE them to archive_root/<label>/ (never overwrite) and return that path."""
    target_dir = Path(target_dir)
    files = [p for p in target_dir.glob("*") if p.is_file()] if target_dir.exists() else []
    if not files:
        return None
    dest = Path(archive_root) / label
    n = 1
    while dest.exists():
        n += 1
        dest = Path(archive_root) / f"{label}_{n}"
    dest.mkdir(parents=True)
    for p in files:
        shutil.move(str(p), str(dest / p.name))
    return dest
