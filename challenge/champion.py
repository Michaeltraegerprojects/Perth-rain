"""Champion freeze: record the audited production model (artifacts, selection rules, feature schemas) and refuse to
run any contest if it has changed. The champion is never retrained, re-selected or replaced from here.

    python -m challenge freeze      # writes challenge/champion_freeze.json (refuses to overwrite a different record)
    python -m challenge verify      # checks current files against the record
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import tomllib

ROOT = Path(__file__).resolve().parents[1]
FREEZE = Path(__file__).resolve().parent / "champion_freeze.json"
RULE_FILES = ["src/perthrain/calibrate.py", "src/perthrain/predict.py", "src/perthrain/build.py",
              "src/perthrain/forecasts.py", "src/perthrain/enso_guard.py"]


class ChampionChanged(RuntimeError):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def locations():
    return [(e["name"], slug(e["name"])) for e in tomllib.loads((ROOT / "config.toml").read_text())["locations"]]


def snapshot() -> dict:
    rec = {"description": "Frozen champion: audited no-ENSO release. Challengers are compared against it; it is "
                          "never retrained, re-selected or replaced by the challenge code.",
           "selection_rule_files": {f: _sha(ROOT / f) for f in RULE_FILES}, "locations": {}}
    for name, s in locations():
        mdir = ROOT / "data" / s / "models"
        man = json.loads((mdir / "manifest.json").read_text())
        sel = json.loads((mdir / "selection.json").read_text())
        leads = {}
        for lead in ("day1", "day2", "day3"):
            b = joblib.load(mdir / f"hurdle_{lead}.joblib")
            leads[lead] = {
                "artifact": f"data/{s}/models/hurdle_{lead}.joblib", "sha256": _sha(mdir / f"hurdle_{lead}.joblib"),
                "selected": {k: sel[lead]["selected"][k] for k in ("feature_set", "C", "season", "climate")},
                "routing_order": b["routing_order"],
                "feature_schemas": {k: f["feature_schema"] for k, f in b["fits"].items()},
                "eval_first": sel[lead]["eval_first"], "eval_last": sel[lead]["eval_last"]}
        rec["locations"][s] = {"name": name, "run_id": man["meta"]["run_id"], "created_utc": man["meta"]["created_utc"],
                               "code_sha256": man["meta"]["code_sha256"], "enso_status": man["meta"]["enso_status"],
                               "manifest_sha256": _sha(mdir / "manifest.json"),
                               "selection_sha256": _sha(mdir / "selection.json"), "leads": leads}
    return rec


def freeze() -> Path:
    rec = snapshot()
    if FREEZE.exists():
        old = json.loads(FREEZE.read_text())
        if old != rec:
            raise ChampionChanged("champion_freeze.json exists and differs from the current champion; not overwritten. "
                                  "Investigate before re-freezing.")
        return FREEZE
    FREEZE.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return FREEZE


def verify() -> dict:
    if not FREEZE.exists():
        raise ChampionChanged("no champion freeze record; run `python -m challenge freeze` first")
    old = json.loads(FREEZE.read_text())
    cur = snapshot()
    if old != cur:
        diffs = [k for k in old["selection_rule_files"] if old["selection_rule_files"][k] != cur["selection_rule_files"].get(k)]
        for s, L in old["locations"].items():
            for lead, d in L["leads"].items():
                if cur["locations"].get(s, {}).get("leads", {}).get(lead, {}).get("sha256") != d["sha256"]:
                    diffs.append(d["artifact"])
        raise ChampionChanged(f"champion differs from its freeze record: {diffs or 'metadata'}")
    return old
