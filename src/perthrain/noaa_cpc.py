"""NOAA CPC weekly Nino-region SST file parser - DIAGNOSTIC ONLY.

Used only to cross-check BoM's index file in the ENSO source-verification report. It is never used as a
training feature, for clipping bounds, for feature selection or for predictions.

Source: https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for ("9120" = 1991-2020 climatology in CPC's
file naming). Each data row is one week, labelled by the week's CENTRE date (header: "Weekly SST data starts
week centered on 2Sept1981").

Fixed-width layout, verified against the raw file (0-based, end-exclusive slices):

    [1:10]   week centre date, DDMMMYYYY
    Nino1+2  SST [15:19]  SSTA [19:23]
    Nino3    SST [28:32]  SSTA [32:36]
    Nino34   SST [41:45]  SSTA [45:49]
    Nino4    SST [54:58]  SSTA [58:62]

SST and SSTA are 4-character fields with NO separator, so a negative anomaly is glued to the SST
(" 26.5-0.2"). Splitting on whitespace or greedy regular expressions mis-assigns columns; an earlier ad-hoc
diagnostic did exactly that and reported SST values as anomalies. Parse by position only.
"""
from __future__ import annotations

import re
from datetime import datetime

import pandas as pd

URL = "https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for"
COLUMNS = {"nino12": (15, 19, 23), "nino3": (28, 32, 36), "nino34": (41, 45, 49), "nino4": (54, 58, 62)}
_DATE = re.compile(r"^ (\d{2}[A-Z]{3}\d{4})")
_LOOKS_LIKE_DATA = re.compile(r"^\s*\d")     # headers never start with a digit; a digit-led line IS a data row
# exact layout of a data row: date, then four 'SST(4 chars) SSTA(4 chars)' blocks separated by 5 spaces. SSTA keeps a
# sign slot (blank or '-') glued to the SST field. Any 1-character shift breaks this pattern.
_BLOCK = r"\d{2}\.\d[ -]\d\.\d"
_ROW = re.compile(r"^ \d{2}[A-Z]{3}\d{4}" + (r" {5}" + _BLOCK) * 4 + r"$")


class CPCFormatError(ValueError):
    pass


def parse_line(line: str) -> dict | None:
    """Parse one data row; returns None for header/blank lines, raises on a malformed data row.

    A line that starts with a date but not at the expected column is a misaligned data row and raises,
    so a shifted row can never be skipped silently."""
    line = line.rstrip("\r\n")
    m = _DATE.match(line)
    if not m:
        if _LOOKS_LIKE_DATA.match(line):
            raise CPCFormatError(f"date not at columns [1:10] (misaligned row): {line!r}")
        return None
    if len(line) < 62:
        raise CPCFormatError(f"data row shorter than 62 characters: {line!r}")
    if not _ROW.match(line):
        raise CPCFormatError(f"data row does not match the fixed-width layout: {line!r}")
    try:
        week = datetime.strptime(m.group(1), "%d%b%Y").date()
    except ValueError as exc:
        raise CPCFormatError(f"invalid date {m.group(1)!r} in {line!r}") from exc
    out: dict = {"week_center": week}
    for name, (a, b, c) in COLUMNS.items():
        try:
            sst, ssta = float(line[a:b]), float(line[b:c])
        except ValueError as exc:
            raise CPCFormatError(f"non-numeric {name} field in {line!r}") from exc
        # plausibility: tropical Pacific SST and its 1991-2020 climatology; anomalies within +/-10 C
        if not (15.0 <= sst <= 35.0) or abs(ssta) >= 10.0 or not (15.0 <= sst - ssta <= 32.0):
            raise CPCFormatError(f"implausible {name} SST={sst} SSTA={ssta} in {line!r}")
        out[f"{name}_sst"], out[f"{name}_ssta"] = sst, ssta
    return out


def parse_cpc_weekly(text: str) -> pd.DataFrame:
    rows = [r for r in (parse_line(ln) for ln in text.splitlines()) if r]
    if not rows:
        raise CPCFormatError("no data rows found")
    df = pd.DataFrame(rows)
    df["week_center"] = pd.to_datetime(df["week_center"])
    if df.week_center.duplicated().any():
        raise CPCFormatError("duplicate week rows")
    return df.sort_values("week_center").reset_index(drop=True)
