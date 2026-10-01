"""Guards that keep the UNVERIFIED ENSO / climate-index feature out of the default (no-ENSO) models.

Every artifact written by the default calibration carries ``enso_status = "no_enso"`` and an explicit
``feature_schema`` per fit. Loading, predicting and the test-suite all call :func:`bundle_violations`; a
default-mode artifact with any climate-index feature, an ENSO-derived interaction, a JMA (or any) fallback
trained with a climate index, or an estimator expecting more inputs than its declared schema is rejected.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

NO_ENSO = "no_enso"
EXPERIMENTAL = "experimental_unverified"
# metadata columns whose NAME mentions ENSO but which carry the status label, not a feature value
PROVENANCE_COLUMNS = frozenset({"enso_status"})

# The prohibition is specific to the UNVERIFIED ENSO inputs (BoM Nino3.4 / relative Nino3.4 / SOI files), the IOD
# SST-anomaly index that was used alongside them, and any feature derived from them (clim_* columns, interactions).
# Forecast rainfall from any NWP model - including JMA GSM - is legitimate and is NOT matched.
def _token(word: str) -> str:
    # regex \b treats "_" and digits as word characters, so \bsoi\b would MISS "soi_30d" and \bsst\b "sst34";
    # bound on LETTERS only ("period" must not match iod, "sensor" must not match enso)
    return rf"(?<![a-z]){word}(?![a-z])"


FORBIDDEN_FEATURE = re.compile(
    "|".join([_token("enso"), "nino", "rnino", "roni", _token("oni"), _token("ssts?"), "ssta", _token("iod"), _token("dmi"),
              "dipole", _token("soi"), _token("mei"), r"southern.?oscillation", "clim_", "climate"]),
    re.IGNORECASE)


def _norm(text: str) -> str:
    """NFKD-normalise so 'Niño' -> 'Nino' (the regexes then need no accented variants)."""
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def forbidden(text: str) -> bool:
    return bool(FORBIDDEN_FEATURE.search(_norm(str(text))))
# Text scan patterns (reports, predictions, READMEs). JMA is listed so every JMA reference can be reviewed,
# but a JMA *forecast* reference is legitimate - only JMA + climate-index features are a violation.
TEXT_CLIMATE = re.compile(_token("enso") + r"|nino|rnino|roni|" + _token("oni") + "|" + _token("iod") + "|" + _token("dmi") + "|" +
                          _token("soi") + r"|indian ocean dipole|southern oscillation|sst anomal|ssta|"
                          r"clim_nino|clim_iod|climate[- _]?(index|indices|driver)", re.IGNORECASE)
TEXT_JMA = re.compile(r"(?<![a-z0-9])jma", re.IGNORECASE)


class EnsoGuardError(RuntimeError):
    pass


def approved_feature_names() -> set[str]:
    """The explicitly approved baseline features: NWP rainfall of the configured feature sets (single model or
    mean-of-logs composite) and the seasonal cycle. Anything else - including MJO, SAM or any other climate
    index that the ENSO-specific prohibition does not name - is NOT approved for a no-ENSO artifact."""
    from .calibrate import FEATURE_SETS, feature_names
    out: set[str] = set()
    for spec in FEATURE_SETS.values():
        for season in (True, False):
            out.update(feature_names(spec["cols"], spec.get("combine"), season, None))
    return out


def fit_violations(name: str, fit: dict) -> list[str]:
    out = []
    schema = fit.get("feature_schema")
    if not schema:
        out.append(f"{name}: no feature_schema recorded")
        schema = []
    approved = approved_feature_names()
    for f in schema:
        if forbidden(f):
            out.append(f"{name}: forbidden feature {f!r}")
        elif f not in approved:
            out.append(f"{name}: feature {f!r} is not in the approved baseline feature set")
    for c in fit.get("cols", []):
        if forbidden(c):
            out.append(f"{name}: forbidden input column {c!r}")
    if fit.get("climate") not in (None, "none"):
        out.append(f"{name}: trained with climate configuration {fit.get('climate')!r}")
    if fit.get("clim_range"):
        out.append(f"{name}: carries climate clipping bounds {fit.get('clim_range')!r}")
    from .calibrate import GRID, HurdleModel
    model = fit.get("model")
    if not isinstance(model, HurdleModel):
        out.append(f"{name}: 'model' is missing or not a HurdleModel ({type(model).__name__})")
        return out
    q = getattr(model, "q_", None)
    if not q or len(q) != len(GRID) or getattr(model, "clf_", None) is None or getattr(model, "scaler_", None) is None:
        out.append(f"{name}: model is not fitted (missing occurrence/amount estimators or quantile levels)")
        return out
    ests = [("occurrence", model.clf_), ("amount_scaler", model.scaler_)] + [(f"quantile[{i}]", e) for i, e in enumerate(q)]
    for label, est in ests:
        n = getattr(est, "n_features_in_", None)
        if n is None:
            n = getattr(getattr(est, "steps", [[None, None]])[-1][1], "n_features_in_", None) \
                if hasattr(est, "steps") else None
        if n is None:
            out.append(f"{name}: {label} has no n_features_in_ (cannot verify input count)")
        elif schema and n != len(schema):
            out.append(f"{name}: {label} expects {n} inputs but the schema declares {len(schema)} "
                       f"(hidden or missing feature)")
    return out


REQUIRED_META = ("run_id", "created_utc", "enso_status", "code_sha256", "source_data", "versions", "lead")
KNOWN_MODEL_COLS = {"sr_ecmwf_ifs", "sr_ncep_gfs_global", "pr_ecmwf_ifs025", "pr_jma_gsm", "pr_ncep_gfs_global",
                    "pr_bom_access_global"}


def metadata_violations(bundle: dict, expected_lead: str | None = None, expected_status: str = NO_ENSO) -> list[str]:
    """Missing or incompatible metadata, and an ordered feature schema that does not match what the code would
    build for the fit's own parameters (cols, combine, season, climate)."""
    from .calibrate import FEATURE_SETS, feature_names
    out = []
    meta = bundle.get("meta")
    if not isinstance(meta, dict):
        return ["no 'meta' block (pre-audit or foreign artifact)"]
    for k in REQUIRED_META:
        v = meta.get(k)
        if k in ("source_data", "versions"):
            good = isinstance(v, dict) and len(v) > 0
        else:
            good = isinstance(v, str) and v.strip() != ""
        if not good:
            out.append(f"meta.{k} missing or of the wrong type ({type(v).__name__})")
    if meta.get("enso_status") != bundle.get("enso_status"):
        out.append(f"meta.enso_status {meta.get('enso_status')!r} != bundle enso_status {bundle.get('enso_status')!r}")
    if bundle.get("enso_status") != expected_status:
        out.append(f"enso_status {bundle.get('enso_status')!r}, expected {expected_status!r}")
    if expected_lead and (bundle.get("lead") != expected_lead or meta.get("lead") != expected_lead):
        out.append(f"lead mismatch: file is for {expected_lead}, bundle says {bundle.get('lead')}/{meta.get('lead')}")
    order = bundle.get("routing_order")
    fits = bundle.get("fits") or {}
    if not order:
        out.append("routing_order missing")
    else:
        for n in order:
            if n not in fits:
                out.append(f"routing_order entry {n!r} has no fit")
        for n in fits:
            if n not in order:
                out.append(f"fit {n!r} is not in routing_order (unreachable)")
        if bundle.get("feature_set") != order[0]:
            out.append(f"selected feature set {bundle.get('feature_set')!r} is not first in routing_order")
    for name, fit in fits.items():
        spec = FEATURE_SETS.get(name)
        if spec is None:
            out.append(f"{name}: unknown feature set (incompatible code version)")
        elif list(spec["cols"]) != list(fit.get("cols", [])) or spec.get("combine") != fit.get("combine"):
            out.append(f"{name}: cols/combine {fit.get('cols')}/{fit.get('combine')} differ from the code's "
                       f"definition {spec['cols']}/{spec.get('combine')}")
        unknown = set(fit.get("cols", [])) - KNOWN_MODEL_COLS
        if unknown:
            out.append(f"{name}: unknown input columns {sorted(unknown)}")
        expected = feature_names(fit.get("cols", []), fit.get("combine"), fit.get("season", True), fit.get("climate"))
        if fit.get("feature_schema") != expected:
            out.append(f"{name}: stored feature_schema {fit.get('feature_schema')} != expected ordered schema {expected}")
    return out


def validate_artifact(bundle: dict, expected_lead: str | None = None) -> list[str]:
    """Everything a default-mode (no-ENSO) artifact must satisfy before it may serve a prediction."""
    return metadata_violations(bundle, expected_lead) + bundle_violations(bundle)


def bundle_violations(bundle: dict) -> list[str]:
    out = []
    if bundle.get("enso_status") != NO_ENSO:
        out.append(f"enso_status is {bundle.get('enso_status')!r}, expected {NO_ENSO!r}")
    if bundle.get("climate") not in (None, "none"):
        out.append(f"selected configuration uses climate {bundle.get('climate')!r}")
    if bundle.get("fits_noclim"):
        out.append("legacy 'fits_noclim' section present (pre-audit artifact layout)")
    fits = bundle.get("fits") or {}
    if not fits:
        out.append("no fits")
    for name, fit in fits.items():
        out += fit_violations(name, fit)
    for r in bundle.get("ranking", []):
        if r.get("climate") not in (None, "none"):
            out.append(f"routing/ranking entry for {r.get('feature_set')} references climate {r.get('climate')!r}")
    return out


def assert_no_enso(bundle: dict, where: str = "", expected_lead: str | None = None) -> None:
    v = validate_artifact(bundle, expected_lead)
    if v:
        raise EnsoGuardError(f"{where}: artifact is not a valid no-ENSO artifact:\n  - " + "\n  - ".join(v))


def _is_link(p: Path) -> bool:
    """Symlink, or a Windows reparse point (directory junction, mount point) - detected from the file attributes,
    so it does not depend on Path.is_junction (Python >= 3.12 only)."""
    import os
    import stat
    if p.is_symlink():
        return True
    try:
        st = os.lstat(p)
    except OSError:
        return False
    return bool(getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def assert_servable_path(path: Path, models_dir: Path, quarantine_root: Path) -> Path:
    """An artifact may be served only if it is a regular file whose RESOLVED location is directly inside the
    configured models directory and not inside the project's quarantine root (``<project>/archive``).

    Comparison is on resolved paths, not on folder names, so a project that itself sits under a folder called
    "archive" is fine, while ``..`` traversal, file symlinks and linked (symlink/junction) directories that lead
    into the quarantine root are refused. Returns the resolved path."""
    p = Path(path)
    if _is_link(p):
        raise EnsoGuardError(f"{p}: artifact is a link; only regular files in the models directory are served")
    models_dir = Path(models_dir)
    for d in [models_dir, *models_dir.parents][:3]:          # models/, <location>/, data/
        if d.exists() and _is_link(d):
            raise EnsoGuardError(f"{d}: linked directory in the artifact path (symlink/junction)")
    try:
        rp = p.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise EnsoGuardError(f"{p}: cannot resolve artifact path ({exc})") from exc
    import os
    import stat
    if not stat.S_ISREG(os.lstat(rp).st_mode):
        raise EnsoGuardError(f"{p}: not a regular file")
    if os.lstat(rp).st_nlink > 1:
        raise EnsoGuardError(f"{p}: has {os.lstat(rp).st_nlink} hard links; only unshared files are served")
    q = Path(quarantine_root).resolve()
    if rp == q or q in rp.parents:
        raise EnsoGuardError(f"{p}: resolves to {rp}, inside the quarantine root {q}")
    if rp.parent != models_dir.resolve():
        raise EnsoGuardError(f"{p}: resolves to {rp}, not directly inside the models directory {models_dir.resolve()}")
    return rp


def quarantined_hashes(quarantine_root: Path) -> set[str]:
    """SHA-256 of every file recorded in a quarantine manifest - catches copies and hard links of quarantined
    artifacts that a path check cannot see."""
    import csv
    import hashlib
    out = set()
    root = Path(quarantine_root)
    for m in root.glob("*/manifest.csv"):
        with open(m, newline="", encoding="utf-8-sig") as fh:        # tolerate a BOM
            out |= {r["sha256"].strip().lower() for r in csv.DictReader(fh) if r.get("sha256")}
    # also hash every archived model file directly (covers superseded_models/ and the forensic snapshot, which
    # have no manifest)
    for f in root.rglob("*.joblib"):
        if f.is_file():
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out.add(h.hexdigest())
    return out


# ----------------------------------------------------------------------------------- scanning
def scan_artifacts(root: Path, pattern: str = "data/*/models*/hurdle_*.joblib") -> list[dict]:
    """Inspect every model artifact matching ``pattern`` (default: active + experimental under data/)."""
    import joblib
    rows = []
    for p in sorted(Path(root).glob(pattern)):
        b = joblib.load(p)
        fits = b.get("fits", {})
        loc = p.parent.parent.name if p.parent.name.startswith("models") else p.parent.name
        rows.append({"file": str(p).replace("\\", "/"), "location": loc, "lead": b.get("lead"),
                     "active": p.resolve().parent.name == "models" and p.resolve().parent.parent.parent.name == "data",
                     "run_id": (b.get("meta") or {}).get("run_id"),
                     "enso_status": b.get("enso_status", "MISSING"), "selected": b.get("feature_set"),
                     "fits": {k: {"cols": v.get("cols"), "climate": v.get("climate"),
                                  "feature_schema": v.get("feature_schema")} for k, v in fits.items()},
                     "violations": validate_artifact(b, b.get("lead")) if p.parent.name == "models"
                     else bundle_violations(b)})
    return rows


def _text_files(root: Path):
    root = Path(root)
    pats = ["README.md", "config*.toml", "reports/**/*.md", "reports/**/*.csv", "reports/**/*.json",
            "data/*/predictions*/*.csv", "data/*/predictions*/*.md", "data/*/models*/*.json"]
    seen = set()
    for pat in pats:
        for p in sorted(root.glob(pat)):
            if p.is_file() and p not in seen:
                seen.add(p)
                yield p


def scan_text(root: Path) -> list[dict]:
    rows = []
    for p in _text_files(root):
        text = p.read_text(encoding="utf-8", errors="replace")
        clim = [(i + 1, ln.strip()[:160]) for i, ln in enumerate(text.splitlines()) if TEXT_CLIMATE.search(ln)]
        jma = sum(1 for ln in text.splitlines() if TEXT_JMA.search(ln))
        if clim or jma:
            rows.append({"file": str(p.relative_to(root)).replace("\\", "/"), "climate_index_lines": len(clim),
                         "jma_lines": jma, "examples": clim[:4]})
    return rows


def scan_payload_columns(root: Path) -> list[dict]:
    """Column names of every active data table and prediction payload that feeds or leaves the models."""
    import pandas as pd
    rows = []
    for pat in ("data/*/clean/*.parquet", "data/*/joined/*.parquet", "data/*/predictions/*.csv"):
        for p in sorted(Path(root).glob(pat)):
            cols = list(pd.read_parquet(p).columns) if p.suffix == ".parquet" else list(pd.read_csv(p, nrows=5).columns)
            bad = [c for c in cols if forbidden(c) and c not in PROVENANCE_COLUMNS]
            if "enso_status" in cols:
                vals_st = set(pd.read_csv(p).enso_status.dropna().astype(str)) if p.suffix == ".csv" else set()
                bad += [f"enso_status={v}" for v in sorted(vals_st - {NO_ENSO})]
            vals = []
            if p.suffix == ".csv" and "climate_used" in cols:
                d = pd.read_csv(p)
                vals = sorted(set(d.climate_used.dropna().astype(str)) - {"none"})
            rows.append({"file": str(p.relative_to(root)).replace("\\", "/"), "n_columns": len(cols),
                         "climate_columns": bad, "non_none_climate_used": vals})
    return rows


def write_scan(root: Path, stage: str) -> Path:
    root = Path(root)
    arts, text, cols = scan_artifacts(root), scan_text(root), scan_payload_columns(root)
    out = {"stage": stage, "artifacts": arts, "payload_columns": cols, "text_references": text,
           "dashboards": "none: the project publishes no dashboards"}
    rep = root / "reports"
    rep.mkdir(exist_ok=True)
    (rep / f"enso_reference_scan_{stage}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    lines = [f"# ENSO / climate-index reference scan - {stage}\n",
             "Scope: every model artifact under `data/<location>/models*/`, every data table and prediction payload "
             "that feeds or leaves the models (`data/<location>/clean|joined|predictions`), every report, README and "
             "config file. `archive/` is excluded (it is labelled separately). No dashboards exist.\n",
             "## Model artifacts (feature schemas and fallback routing)\n",
             "| file | active | lead | run_id | enso_status | selected | fits (feature set: climate) | violations |",
             "|---|---|---|---|---|---|---|---|"]
    for a in arts:
        fits = "; ".join(f"{k}: {v['climate'] or 'none'}" for k, v in a["fits"].items())
        lines.append(f"| {a['file']} | {a['active']} | {a['lead']} | {a['run_id']} | {a['enso_status']} | "
                     f"{a['selected']} | {fits} | {len(a['violations'])} |")
    if not arts:
        lines.append("| _(no artifacts)_ | | | | | | | |")
    lines += ["\n## Data tables and prediction payloads\n", "| file | columns | climate-index columns | non-'none' climate_used |",
              "|---|---|---|---|"]
    for c in cols:
        lines.append(f"| {c['file']} | {c['n_columns']} | {', '.join(c['climate_columns']) or '-'} | "
                     f"{', '.join(c['non_none_climate_used']) or '-'} |")
    lines += ["\n## Text references (reports, README, config, prediction text)\n",
              "Lines mentioning ENSO / Niño / IOD / SOI / climate indices, and lines mentioning JMA (a legitimate "
              "forecast model; listed for review).\n", "| file | climate-index lines | JMA lines | first examples |",
              "|---|---|---|---|"]
    for t in text:
        ex = " / ".join(f"L{n}: {s}" for n, s in t["examples"]).replace("|", "\\|")
        lines.append(f"| {t['file']} | {t['climate_index_lines']} | {t['jma_lines']} | {ex} |")
    n_bad = sum(len(a["violations"]) for a in arts if a["active"])
    n_col = sum(len(c["climate_columns"]) + len(c["non_none_climate_used"]) for c in cols)
    lines.append(f"\n**Active-artifact violations: {n_bad}. Climate columns / climate-used values in active tables "
                 f"and payloads: {n_col}.**\n")
    p = rep / f"enso_reference_scan_{stage}.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p
