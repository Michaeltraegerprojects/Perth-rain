"""BoM station metadata and primary-gauge selection."""
from __future__ import annotations

import io
import math
import re
import zipfile

import pandas as pd

FTP_BASE = "ftp://ftp.bom.gov.au/anon/gen/clim_data/IDCKWCDEA0/tables"
STATIONS_DB_URL = f"{FTP_BASE}/stations_db.txt"
# IDCJMC0014: all BoM stations incl. manual rain gauges (data for most only via CDO).
ALL_STATIONS_URL = "ftp://ftp.bom.gov.au/anon2/home/ncc/metadata/sitelists/stations.zip"

_DB_RE = re.compile(
    r"^(?P<id>\d{6})\s+(?P<state>[A-Z]+)\s+(?P<district>\S+)\s+(?P<name>.+?)\s+"
    r"(?P<start>\d{8})\.\.(?P<end>\d{8})?\s+(?P<lat>-?\d+(?:\.\d+)?)\s+(?P<lon>-?\d+(?:\.\d+)?)")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def parse_stations_db(text: str) -> pd.DataFrame:
    """Parse the IDCKWCDEA0 ``stations_db.txt`` (stations with downloadable daily files)."""
    rows = []
    for line in text.splitlines():
        m = _DB_RE.match(line.strip())
        if m:
            d = m.groupdict()
            rows.append({
                "station_id": d["id"], "state": d["state"], "district": d["district"],
                "station_name": d["name"].strip(),
                "db_start_date": pd.to_datetime(d["start"], format="%Y%m%d").date(),
                "db_end_date": pd.to_datetime(d["end"], format="%Y%m%d").date() if d["end"] else None,
                "latitude": float(d["lat"]), "longitude": float(d["lon"]),
            })
    return pd.DataFrame(rows)


def parse_all_stations_zip(blob: bytes) -> pd.DataFrame:
    """Parse IDCJMC0014 ``stations.txt`` (fixed width, widths taken from the dashed rule)."""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        text = zf.read(zf.namelist()[0]).decode("latin-1")
    lines = text.splitlines()
    rule_idx = next(i for i, ln in enumerate(lines) if ln.startswith("-------"))
    spans, pos = [], 0
    for m in re.finditer(r"-+", lines[rule_idx]):
        spans.append((m.start(), m.end()))
    names = ["site", "dist", "name", "start", "end", "lat", "lon", "source", "sta", "height",
             "bar_ht", "wmo"]
    rows = []
    for ln in lines[rule_idx + 1:]:
        if not ln.strip() or not ln.strip()[0].isdigit():
            continue
        # widen spans so values that overflow slightly to the left are captured
        vals = [ln[s:(spans[i + 1][0] if i + 1 < len(spans) else len(ln))].strip()
                for i, (s, _) in enumerate(spans)]
        rows.append(dict(zip(names, vals)))
    df = pd.DataFrame(rows)
    df = df[pd.to_numeric(df["lat"], errors="coerce").notna()].copy()
    df["station_id"] = df["site"].str.zfill(6)
    df["latitude"] = df["lat"].astype(float)
    df["longitude"] = df["lon"].astype(float)
    df["open"] = df["end"].eq("..")
    return df.rename(columns={"name": "station_name", "start": "start_year", "end": "end_year",
                              "sta": "state_code"})


def ftp_folder_name(station_name: str) -> str:
    """Map a BoM station name to its IDCKWCDEA0 folder name (e.g. PERTH AIRPORT M.O. -> perth_airport_mo)."""
    s = station_name.strip().lower().replace(".", "")
    return re.sub(r"\s+", "_", s)


STATE_FOLDERS = {"WA": "wa", "NSW": "nsw", "VIC": "vic", "QLD": "qld", "SA": "sa", "TAS": "tas",
                 "NT": "nt", "ACT": "nsw"}


def nearby(df: pd.DataFrame, lat: float, lon: float, radius_km: float) -> pd.DataFrame:
    out = df.copy()
    out["distance_km"] = [haversine_km(lat, lon, a, b) for a, b in zip(out.latitude, out.longitude)]
    return out[out.distance_km <= radius_km].sort_values("distance_km").reset_index(drop=True)


def choose_station(candidates: pd.DataFrame, period_start, min_completeness: float,
                   forced_id: str = "") -> tuple[pd.Series, str]:
    """Pick the primary gauge.

    ``candidates`` needs columns station_id, distance_km, completeness, db_start_date,
    id_ambiguous. Rule: nearest station whose record is complete enough over the
    period and whose station number did not change within the period.
    """
    if forced_id:
        hit = candidates[candidates.station_id == forced_id]
        if hit.empty:
            raise SystemExit(f"Station {forced_id} is not among downloadable IDCKWCDEA0 candidates "
                             f"within the search radius; increase --radius or supply CDO CSVs.")
        return hit.iloc[0], "chosen explicitly by the user (--station / config station.id)"
    ok = candidates[(candidates.completeness >= min_completeness)
                    & (pd.to_datetime(candidates.db_start_date) <= pd.Timestamp(period_start))
                    & (~candidates.id_ambiguous)]
    if ok.empty:
        raise SystemExit("No candidate station satisfies completeness/stability rules.")
    best = ok.sort_values("distance_km").iloc[0]
    reason = (f"nearest downloadable gauge ({best.distance_km:.1f} km) with rainfall completeness "
              f"{best.completeness:.1%} >= {min_completeness:.0%} over the period and an unchanged "
              f"station number since {best.db_start_date}")
    return best, reason
