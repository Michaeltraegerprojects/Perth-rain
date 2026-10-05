"""Champion freeze: record the audited production model and refuse to run any contest if it has changed. The
champion is never retrained, re-selected or replaced from here.

    python -m challenge freeze                    # writes challenge/champion_freeze.json (never overwrites another)
    python -m challenge verify                    # checks the current files against the record
    python -m challenge refreeze --reason TEXT    # new record for unchanged artifacts after a reviewed code change

Record (schema 2) covers:
* every model artifact, manifest and selection file, by SHA-256 of their bytes;
* the inference code: config.toml plus every module the prediction and calibration paths import (computed from the
  import graph), excluding report-only and diagnostic modules, so a diagnostic edit is not a champion change;
* the runtime versions (Python, numpy, pandas, scikit-learn, scipy, joblib);
* routing order and feature schemas, read from the run's manifest.json (no artifact is unpickled).

verify() compares bytes and versions only. It never deserialises an artifact, so a modified or malicious file is
reported as a change before anything could load it. Text files (config and code) are hashed with CRLF line endings
read as LF, so a Windows checkout and a Linux checkout of the same commit give the same record.
"""
from __future__ import annotations

import ast
import hashlib
import json
import platform
import tomllib
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE = Path(__file__).resolve().parent / "champion_freeze.json"
SCHEMA = 2
INFERENCE_ENTRY = ("predict.py", "calibrate.py")
# imported by the inference/calibration paths but only write reports or run diagnostics / experiments
NOT_INFERENCE = {"calibrate_report.py", "report.py", "climate.py", "noaa_cpc.py", "baseline.py", "probe.py",
                 "cli.py", "__main__.py"}
RUNTIME_PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "joblib")


class ChampionChanged(RuntimeError):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _sha_text(p: Path) -> str:
    """Hash of a text file independent of its line endings (git may check it out with CRLF or LF)."""
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _write_record(path: Path, rec: dict) -> None:
    path.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def locations():
    return [(e["name"], slug(e["name"])) for e in tomllib.loads((ROOT / "config.toml").read_text())["locations"]]


def inference_modules() -> list[str]:
    """Package modules reachable from the prediction and calibration entry points (relative imports), minus
    report-only and diagnostic modules."""
    pkg = ROOT / "src" / "perthrain"
    seen, todo = set(), list(INFERENCE_ENTRY)
    while todo:
        name = todo.pop()
        if name in seen or not (pkg / name).exists():
            continue
        seen.add(name)
        tree = ast.parse((pkg / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                if node.module:
                    todo.append(node.module.split(".")[0] + ".py")
                else:
                    todo += [a.name + ".py" for a in node.names]
    return sorted(seen - NOT_INFERENCE)


def inference_files() -> dict:
    files = {"config.toml": _sha_text(ROOT / "config.toml")}
    for m in inference_modules():
        files[f"src/perthrain/{m}"] = _sha_text(ROOT / "src" / "perthrain" / m)
    return files


def runtime_versions() -> dict:
    out = {"python": platform.python_version()}
    for p in RUNTIME_PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def _location_record(name: str, s: str) -> dict:
    mdir = ROOT / "data" / s / "models"
    man = json.loads((mdir / "manifest.json").read_text())
    sel = json.loads((mdir / "selection.json").read_text())
    leads = {}
    for a in man["artifacts"]:
        lead = a["lead"]
        art = mdir / a["file"]
        digest = _sha(art)
        if digest != a["sha256"]:
            raise ChampionChanged(f"{art}: bytes do not match its own manifest.json; refusing to freeze")
        leads[lead] = {"artifact": f"data/{s}/models/{a['file']}", "sha256": digest,
                       "selected": {k: sel[lead]["selected"][k] for k in ("feature_set", "C", "season", "climate")},
                       "routing_order": a["routing_order"],
                       "feature_schemas": {k: f["feature_schema"] for k, f in a["fits"].items()},
                       "eval_first": sel[lead]["eval_first"], "eval_last": sel[lead]["eval_last"]}
    return {"name": name, "run_id": man["meta"]["run_id"], "created_utc": man["meta"]["created_utc"],
            "training_code_sha256": man["meta"]["code_sha256"], "enso_status": man["meta"]["enso_status"],
            "manifest_sha256": _sha(mdir / "manifest.json"), "selection_sha256": _sha(mdir / "selection.json"),
            "leads": leads}


def holdout_references() -> dict:
    ref = ROOT / "challenge" / "reference"
    return {p.name: _sha(p) for p in sorted(ref.glob("*.parquet"))} if ref.exists() else {}


def check_references(rec: dict) -> None:
    """Every per-row reference a record lists must exist with its recorded hash. Run before any reproduction in
    `refreeze`, so a deleted or edited reference is reported instead of being rebuilt from the current code."""
    ref = ROOT / "challenge" / "reference"
    for name, digest in sorted((rec.get("holdout_references") or {}).items()):
        p = ref / name
        if not p.exists():
            raise ChampionChanged(f"challenge/reference/{name} is recorded in the freeze but missing; restore it from "
                                  "a backup (a recorded reference is never recreated)")
        if _sha(p) != digest:
            raise ChampionChanged(f"challenge/reference/{name} differs from the hash recorded in the freeze")


def snapshot() -> dict:
    """Current state, from file bytes and manifests only (nothing is unpickled)."""
    return {"schema_version": SCHEMA,
            "description": "Frozen champion: audited no-ENSO release. Challengers are compared against it; it is "
                           "never retrained, re-selected or replaced by the challenge code.",
            "inference_files": inference_files(), "runtime": runtime_versions(),
            "holdout_references": holdout_references(),
            "locations": {s: _location_record(name, s) for name, s in locations()}}


_COMPARED = ("schema_version", "inference_files", "runtime", "holdout_references", "locations")


def _diff(old: dict, cur: dict) -> list[str]:
    out = []
    for f in sorted(set(old.get("inference_files", {})) | set(cur["inference_files"])):
        if old.get("inference_files", {}).get(f) != cur["inference_files"].get(f):
            out.append(f)
    for k in sorted(set(old.get("runtime", {})) | set(cur["runtime"])):
        if old.get("runtime", {}).get(k) != cur["runtime"].get(k):
            out.append(f"runtime {k}: frozen {old.get('runtime', {}).get(k)} vs installed {cur['runtime'].get(k)}")
    for f in sorted(set(old.get("holdout_references", {})) | set(cur["holdout_references"])):
        if old.get("holdout_references", {}).get(f) != cur["holdout_references"].get(f):
            out.append(f"challenge/reference/{f}")
    for s in sorted(set(old.get("locations", {})) | set(cur["locations"])):
        o, c = old.get("locations", {}).get(s, {}), cur["locations"].get(s, {})
        for k in ("manifest_sha256", "selection_sha256", "run_id"):
            if o.get(k) != c.get(k):
                out.append(f"data/{s}/models {k}")
        for lead in sorted(set(o.get("leads", {})) | set(c.get("leads", {}))):
            if o.get("leads", {}).get(lead, {}).get("sha256") != c.get("leads", {}).get(lead, {}).get("sha256"):
                out.append(f"data/{s}/models/hurdle_{lead}.joblib")
    return out


def _current_state() -> dict:
    """Hash-only state for verification. Artifact bytes are hashed, never loaded."""
    try:
        return snapshot()
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        raise ChampionChanged(f"champion files missing or unreadable: {exc}") from exc


def freeze(reason: str = "initial freeze") -> Path:
    cur = _current_state()
    if FREEZE.exists():
        old = json.loads(FREEZE.read_text())
        if old.get("schema_version") == SCHEMA and {k: old.get(k) for k in _COMPARED} == {k: cur[k] for k in _COMPARED}:
            return FREEZE
        raise ChampionChanged("champion_freeze.json exists and differs from the current champion "
                              f"({_diff(old, cur) or 'record format'}); not overwritten. Investigate, or use "
                              "`refreeze --reason` after a reviewed change that leaves the artifacts untouched.")
    rec = dict(cur, created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), reason=reason, history=[])
    _write_record(FREEZE, rec)
    return FREEZE


def verify() -> dict:
    if not FREEZE.exists():
        raise ChampionChanged("no champion freeze record; run `python -m challenge freeze` first")
    old = json.loads(FREEZE.read_text())
    if old.get("schema_version") != SCHEMA:
        raise ChampionChanged("champion_freeze.json uses the old format (artifacts and 5 rule files only). "
                              "Run `python -m challenge refreeze --reason ...` once to record full coverage.")
    cur = _current_state()
    diffs = _diff(old, cur)
    if diffs:
        raise ChampionChanged(f"champion differs from its freeze record: {diffs}")
    return old


def refreeze(reason: str) -> Path:
    """New record after a reviewed change to code, config or runtime. Refuses if any artifact, manifest or
    selection changed: a different model is a new champion, not a refreeze. The old record is kept in history."""
    if not reason or len(reason) < 10:
        raise ValueError("refreeze needs a reason describing the reviewed change")
    if not FREEZE.exists():
        raise ChampionChanged("nothing to refreeze; run `freeze` first")
    old = json.loads(FREEZE.read_text())
    check_references(old)                # the per-row evidence the old record relied on must be intact
    cur = _current_state()
    for s, L in old["locations"].items():
        c = cur["locations"].get(s)
        if c is None:
            raise ChampionChanged(f"location {s} disappeared")
        if old.get("schema_version") == SCHEMA:
            same = c["manifest_sha256"] == L["manifest_sha256"] and c["selection_sha256"] == L["selection_sha256"]
        else:   # schema 1 recorded the manifest and selection hashes under the same names
            same = c["manifest_sha256"] == L.get("manifest_sha256") and c["selection_sha256"] == L.get("selection_sha256")
        same = same and all(c["leads"][lead]["sha256"] == d["sha256"] for lead, d in L["leads"].items())
        if not same:
            raise ChampionChanged(f"{s}: model artifacts, manifest or selection changed; this is a new champion, "
                                  "not a refreeze")
    hist_dir = FREEZE.parent / "freeze_history"
    hist_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kept = hist_dir / f"champion_freeze_{stamp}.json"
    kept.write_bytes(FREEZE.read_bytes().replace(b"\r\n", b"\n"))   # as committed (the repository stores LF)
    history = list(old.get("history", [])) + [{"replaced_utc": stamp, "record_file": f"freeze_history/{kept.name}",
                                               "record_sha256": _sha(kept), "schema_version": old.get("schema_version", 1),
                                               "changed": _diff(old, cur) if old.get("schema_version") == SCHEMA
                                               else ["schema 1 -> 2: added config, inference modules, runtime"]}]
    rec = dict(cur, created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), reason=reason, history=history)
    _write_record(FREEZE, rec)
    return FREEZE
