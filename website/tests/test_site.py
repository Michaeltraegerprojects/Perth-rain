"""Checks for the public website: published data, privacy, references, and the export guard."""
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE / "tools"))
import export_site_data as X  # noqa: E402

PUBLIC = SITE / "data" / "v1"
PUBLISHABLE = [p for p in SITE.rglob("*") if p.is_file() and "tests" not in p.parts and "tools" not in p.parts
               and "pending" not in p.parts]


def test_published_data_passes_validation():
    assert X.validate(PUBLIC) == []


def test_site_contains_only_web_files():
    allowed = {".html", ".css", ".js", ".json", ".svg", ".ico", ".txt", ".md", ".webmanifest", ".png", ".nojekyll"}
    bad = [p.name for p in PUBLISHABLE if p.suffix.lower() not in allowed and p.name != ".nojekyll"]
    assert not bad, f"non-web files would be published: {bad}"
    assert not [p for p in PUBLISHABLE if p.suffix in (".joblib", ".parquet", ".zip", ".env", ".pkl")]


@pytest.mark.parametrize("path", PUBLISHABLE, ids=lambda p: p.relative_to(SITE).as_posix())
def test_no_local_paths_credentials_or_personal_data(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    assert not re.search(r"[A-Za-z]:\\|[A-Za-z]:/(?!/)|\\Users\\|/Users/|AppData|\.venv|\.joblib", text), path
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text), f"{path}: e-mail address"
    assert not re.search(r"(api[_-]?key|secret|password|token)\s*[:=]\s*\S", text, re.I), path


def test_html_references_exist():
    for page in SITE.glob("*.html"):
        for ref in re.findall(r'(?:href|src)="([^"#:]+)"', page.read_text(encoding="utf-8")):
            assert (SITE / ref).exists(), f"{page.name} -> {ref}"


def test_scripts_load_only_the_versioned_public_export():
    for js in (SITE / "assets").glob("*.js"):
        for url in re.findall(r'fetch\(\s*([^,)]+)', js.read_text(encoding="utf-8")):
            assert "data/v1" in url or url.strip() == "DATA + \"forecast.json\"", (js.name, url)
    assert 'const DATA = "data/v1/"' in (SITE / "assets" / "forecast.js").read_text(encoding="utf-8")


def test_unavailable_days_carry_no_numbers_and_probabilities_are_ordered():
    fc = json.loads((PUBLIC / "forecast.json").read_text(encoding="utf-8"))
    assert fc["schema_version"] == 1 and fc["exported_utc"]
    for loc in fc["locations"]:
        assert loc["gauge"]["distance_km"] > 0 and loc["forecast_made_utc"]
        for d in loc["days"]:
            if d["availability"] == "unavailable":
                assert d["forecasts"] == []
            for f in d["forecasts"]:
                p = f["probabilities"]
                assert p["ge_0_2mm"] >= p["ge_1mm"] >= p["ge_5mm"] >= p["ge_10mm"]
                assert f["verification"] in {"verified", "unverified", "failed"}
                assert f["lead_hours_to_window_start"] > 0


def _copy_public(tmp_path):
    d = tmp_path / "d"
    shutil.copytree(PUBLIC, d)
    return d


def test_validator_rejects_local_paths(tmp_path):
    d = _copy_public(tmp_path)
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    m["note"] = "built from C:\\Users\\someone\\project"
    (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    assert any("local path" in p for p in X.validate(d))


def test_validator_rejects_numbers_on_unavailable_days(tmp_path):
    d = _copy_public(tmp_path)
    fc = json.loads((d / "forecast.json").read_text(encoding="utf-8"))
    day = next(x for loc in fc["locations"] for x in loc["days"] if x["availability"] == "available")
    day["availability"] = "unavailable"
    (d / "forecast.json").write_text(json.dumps(fc), encoding="utf-8")
    assert any("unavailable day carries numbers" in p for p in X.validate(d))


def test_validator_rejects_missing_files(tmp_path):
    d = _copy_public(tmp_path)
    (d / "performance.json").unlink()
    assert X.validate(d) == ["missing performance.json"]


def test_export_does_not_read_model_artifacts():
    src = (SITE / "tools" / "export_site_data.py").read_text(encoding="utf-8")
    assert "joblib" not in src.replace("(.joblib)", "").replace(".joblib|", "") or "import joblib" not in src
    assert "import joblib" not in src and "joblib.load" not in src
