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
# exactly what deploy/github-pages-workflow.yml copies: top-level pages, assets/, data/v1/
PUBLISHABLE = sorted([*SITE.glob("*.html"), *(SITE / "assets").rglob("*"), *(SITE / "data" / "v1").rglob("*")])
PUBLISHABLE = [p for p in PUBLISHABLE if p.is_file()]


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
    assert not re.search(r"[\w.+-]+@[A-Za-z][\w-]*\.[A-Za-z]{2,}", text), f"{path}: e-mail address"
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
                assert f["verification"] in {"verified", "checked_at_issue", "unverified", "failed"}
                assert isinstance(f["heldout_scored"], bool)
                if f["is_fallback"] and not f["heldout_scored"]:
                    pass   # the card must say the backup model is unscored (format.test.mjs covers the wording)
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


ALLOWED_HOSTS = {"tiles.openfreemap.org", "gibs.earthdata.nasa.gov", "tiles.maps.eox.at", "cdn.jsdelivr.net",
                 # plain links (not loaded automatically)
                 "open-meteo.com", "openfreemap.org", "www.openmaptiles.org", "www.openstreetmap.org",
                 "earthdata.nasa.gov", "cloudless.eox.at"}


def test_only_approved_external_hosts():
    for p in PUBLISHABLE:
        if p.suffix not in (".html", ".js", ".css"):
            continue
        hosts = set(re.findall(r"https://([a-z0-9.-]+)", p.read_text(encoding="utf-8")))
        assert hosts <= ALLOWED_HOSTS, (p.name, hosts - ALLOWED_HOSTS)


def test_third_party_code_is_pinned_and_integrity_checked():
    html = (SITE / "map.html").read_text(encoding="utf-8")
    for tag in re.findall(r"<(?:script|link)[^>]+cdn\.jsdelivr\.net[^>]+>", html):
        assert re.search(r"@\d+\.\d+\.\d+/", tag), tag
        assert 'integrity="sha384-' in tag and 'crossorigin="anonymous"' in tag, tag
    for page in ("index.html", "performance.html"):
        assert "https://" not in re.sub(r'<a [^>]*href="https://[^"]+"', "", (SITE / page).read_text(encoding="utf-8")), \
            f"{page} must load no third-party resources"


def test_map_marks_planned_gauges_as_not_in_use():
    m = json.loads((PUBLIC / "map.json").read_text(encoding="utf-8"))
    planned = [g for g in m["gauges"] if g["status"] == "planned"]
    assert planned and all(g["id"] == "009264" for g in planned)
    fc = json.loads((PUBLIC / "forecast.json").read_text(encoding="utf-8"))
    assert all(l["gauge"]["id"] != "009264" for l in fc["locations"]), "a planned gauge must not appear as in use"


def test_challenge_export_keeps_probabilities_and_missing_honest():
    p = PUBLIC / "challenge.json"
    if not p.exists():
        pytest.skip("challenge.json not exported yet")
    d = json.loads(p.read_text(encoding="utf-8"))
    for track in ("historical", "prospective"):
        for g in (d.get(track) or {}).get("groups", []):
            for pr in g["probabilities"]:
                if pr["key"] not in ("champion", "clim"):
                    assert pr["Brier"] is None and "unavailable" in pr["note"]
            n = {a["n_days"] for a in g["amounts"]}
            assert len(n) == 1, "every competitor must be scored on the same days"
            for row in g["paired"]:
                if row["n_days"] < 60:
                    assert row["verdict"] == "insufficient evidence"
    text = p.read_text(encoding="utf-8").lower()
    assert "most accurate" not in text


def test_validator_rejects_nan_which_browsers_cannot_parse(tmp_path):
    d = _copy_public(tmp_path)
    (d / "manifest.json").write_text('{"schema_version": 1, "x": NaN}', encoding="utf-8")
    assert any("invalid JSON for browsers" in p for p in X.validate(d))


def test_outlook_keeps_raw_guidance_apart_from_calibrated_chances(tmp_path):
    p = PUBLIC / "outlook.json"
    if not p.exists():
        pytest.skip("outlook.json not exported yet")
    o = json.loads(p.read_text(encoding="utf-8"))
    assert len(o["days"]) == 7 and all(isinstance(d["calibrated"], bool) for d in o["days"])
    for day in o["days"]:
        if day["calibrated"]:
            assert day["chance_of_any_rain"]
        else:
            assert day["chance_of_any_rain"] is None and "%" not in (day["model_agreement"] or "")
    assert "Not an official forecast" in o["sources"] and o["findings"]
    d = _copy_public(tmp_path)
    o["days"][0]["calibrated"] = False
    o["days"][0]["chance_of_any_rain"] = "60%"
    (d / "outlook.json").write_text(json.dumps(o), encoding="utf-8")
    assert any("raw model guidance shown as a percentage" in x for x in X.validate(d))


def test_failed_forecasts_are_withdrawn_not_published():
    fc = json.loads((PUBLIC / "forecast.json").read_text(encoding="utf-8"))
    for loc in fc["locations"]:
        for day in loc["days"]:
            assert all(f["verification"] != "failed" for f in day["forecasts"])
            if day.get("withdrawn") and not day["forecasts"]:
                assert day["availability"] == "unavailable" and "withdrawn" in day["unavailable_reason"]


# ------------------------------------------------------------------- empty verification detail (found 2026-10-05)
def test_missing_verification_detail_never_becomes_nan_or_the_word_nan():
    """A row the verifier could not check has an empty detail, which pandas reads back as NaN. The first export of
    real data after PR #1 crashed on it (json refuses NaN)."""
    import math
    for empty in (float("nan"), None, ""):
        text = X.detail_text(empty, "live response not found in cache")
        assert isinstance(text, str) and text and "nan" not in text.lower()
        assert not (isinstance(text, float) and math.isnan(text))
    assert X.detail_text("sr_ecmwf_ifs: run 2026-10-04 12Z (explicit)", "x") == "sr_ecmwf_ifs: run 2026-10-04 12Z (explicit)"
    assert "could not be checked" in X.detail_text(float("nan"), float("nan"))


@pytest.mark.local_data
def test_every_location_exports_to_strict_json_from_the_local_outputs():
    import tomllib
    import pandas as pd
    status = pd.read_csv(X.ROOT / "reports" / "served_row_status.csv")
    verif = pd.read_csv(X.ROOT / "reports" / "served_row_verification.csv")
    for e in tomllib.loads((X.ROOT / "config.toml").read_text())["locations"]:
        json.dumps(X.export_location(e["name"], X.slug(e["name"]), status, verif), allow_nan=False)
