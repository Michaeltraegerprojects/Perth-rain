"""Regression tests for the NOAA CPC weekly SST parser (DIAGNOSTIC ONLY - never a model input).

Every ROW below was copied verbatim from https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for
(downloaded 2026-09-30); expected values were read off by hand with a column ruler:

    0         1         2         3         4         5         6
    0123456789012345678901234567890123456789012345678901234567890123
     02SEP1981     20.6-0.1     24.8-0.1     26.5-0.2     28.3-0.3
                   ^^^^----     Nino1+2 SST [15:19] SSTA [19:23]   (negative anomaly glued: "20.6-0.1")
                                             ^^^^----  Nino34 SST [41:45], SSTA [45:49]
     23SEP2026     25.4 4.7     ...          (positive anomaly: the 4-char SSTA field is " 4.7", sign slot blank)
"""
from datetime import date
from pathlib import Path

import pytest

from perthrain.noaa_cpc import CPCFormatError, parse_cpc_weekly, parse_line

HEADER = (" Weekly SST data starts week centered on 2Sept1981\n\n"
          "                Nino1+2      Nino3        Nino34        Nino4\n"
          " Week          SST SSTA     SST SSTA     SST SSTA     SST SSTA\n")

# (raw row, week centre, {region: (SST, SSTA)}) for all four regions
ROWS = [
    # negative glued pairs in every region, incl. the canonical "20.6-0.1"
    (" 02SEP1981     20.6-0.1     24.8-0.1     26.5-0.2     28.3-0.3", date(1981, 9, 2),
     {"nino12": (20.6, -0.1), "nino3": (24.8, -0.1), "nino34": (26.5, -0.2), "nino4": (28.3, -0.3)}),
    # CPC writes a zero anomaly as "-0.0"
    (" 21OCT1981     20.4-0.7     24.9-0.2     26.7-0.0     28.7-0.0", date(1981, 10, 21),
     {"nino12": (20.4, -0.7), "nino3": (24.9, -0.2), "nino34": (26.7, 0.0), "nino4": (28.7, 0.0)}),
    # strong negative anomalies
    (" 29DEC2010     23.2-0.3     24.0-1.3     25.2-1.3     27.1-1.2", date(2010, 12, 29),
     {"nino12": (23.2, -0.3), "nino3": (24.0, -1.3), "nino34": (25.2, -1.3), "nino4": (27.1, -1.2)}),
    (" 30DEC2020     22.2-1.3     24.6-0.7     25.5-1.0     27.3-1.0", date(2020, 12, 30),
     {"nino12": (22.2, -1.3), "nino3": (24.6, -0.7), "nino34": (25.5, -1.0), "nino4": (27.3, -1.0)}),
    # mixed row: glued negative pair in Nino1+2, positive pairs elsewhere
    (" 15SEP2004     20.0-0.7     25.1 0.2     27.2 0.6     29.3 0.7", date(2004, 9, 15),
     {"nino12": (20.0, -0.7), "nino3": (25.1, 0.2), "nino34": (27.2, 0.6), "nino4": (29.3, 0.7)}),
    # positive pairs; Nino3 differs from Nino34 so a column shift would be caught
    (" 06JAN2016     25.5 1.6     28.0 2.6     28.8 2.3     29.5 1.3", date(2016, 1, 6),
     {"nino12": (25.5, 1.6), "nino3": (28.0, 2.6), "nino34": (28.8, 2.3), "nino4": (29.5, 1.3)}),
    (" 27DEC2023     24.2 0.8     27.4 2.1     28.6 2.0     29.7 1.4", date(2023, 12, 27),
     {"nino12": (24.2, 0.8), "nino3": (27.4, 2.1), "nino34": (28.6, 2.0), "nino4": (29.7, 1.4)}),
    # latest week on 2026-09-30
    (" 23SEP2026     25.4 4.7     28.8 3.9     29.7 3.1     29.8 1.1", date(2026, 9, 23),
     {"nino12": (25.4, 4.7), "nino3": (28.8, 3.9), "nino34": (29.7, 3.1), "nino4": (29.8, 1.1)}),
]


@pytest.mark.parametrize("row,center,expected", ROWS, ids=[r[0][1:10] for r in ROWS])
def test_manually_checked_rows_all_four_regions(row, center, expected):
    r = parse_line(row)
    assert r["week_center"] == center
    for region, (sst, ssta) in expected.items():
        assert r[f"{region}_sst"] == sst, region
        assert r[f"{region}_ssta"] == ssta, region


def test_the_canonical_glued_negative_pair():
    r = parse_line(ROWS[0][0])
    assert (r["nino12_sst"], r["nino12_ssta"]) == (20.6, -0.1)       # "20.6-0.1"


def test_anomaly_is_never_the_absolute_sst():
    r = parse_line(ROWS[-1][0])
    assert r["nino34_ssta"] == 3.1 and r["nino34_ssta"] != r["nino34_sst"]
    df = parse_cpc_weekly(HEADER + "\n".join(r[0] for r in ROWS) + "\n")
    assert (df.nino34_ssta.abs() < 10).all() and (df.nino34_sst > 20).all()


def test_headers_and_blank_lines_are_skipped_and_rows_sorted():
    text = HEADER + "\n".join(r[0] for r in reversed(ROWS)) + "\n\n"
    df = parse_cpc_weekly(text)
    assert len(df) == len(ROWS) and df.week_center.is_monotonic_increasing
    assert df.nino34_ssta.tolist() == [-0.2, 0.0, 0.6, -1.3, 2.3, -1.0, 2.0, 3.1]


@pytest.mark.parametrize("bad", [
    " 23SEP2026     25.4 4.7     28.8 3.9     29.7",                            # truncated
    " 23SEP2026     25.4 4.7     28.8 3.9     2x.7 3.1     29.8 1.1",          # non-numeric
    "  23SEP2026     25.4 4.7     28.8 3.9     29.7 3.1     29.8 1.1",         # whole row shifted right
    " 23SEP2026      25.4 4.7     28.8 3.9     29.7 3.1     29.8 1.1",         # fields shifted right
    "23SEP2026     25.4 4.7     28.8 3.9     29.7 3.1     29.8 1.1",           # whole row shifted left
])
def test_malformed_or_misaligned_rows_raise_instead_of_being_skipped(bad):
    with pytest.raises(CPCFormatError):
        parse_line(bad)


def test_one_misaligned_row_fails_the_whole_file():
    text = HEADER + ROWS[0][0] + "\n" + "  23SEP2026     25.4 4.7     28.8 3.9     29.7 3.1     29.8 1.1\n"
    with pytest.raises(CPCFormatError):
        parse_cpc_weekly(text)


def test_cpc_parser_is_diagnostic_only():
    """No pipeline, model or prediction module may import the NOAA CPC parser."""
    pkg = Path(__file__).resolve().parents[1] / "src" / "perthrain"
    users = sorted(p.name for p in pkg.glob("*.py")
                   if p.name != "noaa_cpc.py" and "noaa_cpc" in p.read_text(encoding="utf-8"))
    assert users == [], f"modules importing the diagnostic CPC parser: {users}"


FIXTURE = Path(__file__).parent / "data" / "cpc_wksst9120_excerpt.txt"


def test_rows_in_this_file_are_verbatim_from_the_fixture():
    """The rows the tests are written against come from a committed excerpt of the real CPC file
    (tests/data/cpc_wksst9120_excerpt.SOURCE.txt records its origin and the full file's sha256)."""
    lines = FIXTURE.read_text().split("\n")
    assert "Weekly SST data starts week centered on 2Sept1981" in lines[0]
    for row, *_ in ROWS:
        assert row in lines, row
    df = parse_cpc_weekly(FIXTURE.read_text())
    assert len(df) == len(ROWS)


@pytest.mark.parametrize("pos", list(range(1, 62)))
@pytest.mark.parametrize("shift", ["delete", "insert"])
def test_every_single_character_shift_raises(pos, shift):
    base = ROWS[-1][0]
    bad = base[:pos] + base[pos + 1:] if shift == "delete" else base[:pos] + " " + base[pos:]
    if bad == base:
        pytest.skip("shifting a space next to a space is a no-op")
    with pytest.raises(CPCFormatError):
        parse_line(bad.ljust(62))


def test_impossible_date_raises_cpc_error():
    with pytest.raises(CPCFormatError):
        parse_line(" 31SEP2026     25.4 4.7     28.8 3.9     29.7 3.1     29.8 1.1")
