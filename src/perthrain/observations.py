"""Observed daily rainfall: BoM IDCKWCDEA0 monthly CSVs (anonymous FTP) and user-supplied
BoM Climate Data Online (IDCJAC0009) CSVs.

Accumulation convention (verified from the file header "Rain 0900-0900" and BoM's
statement that daily rainfall for a date is the 24-hour total from 9am local clock
time on the previous day to 9am on that date): a value dated D covers
[D-1 09:00, D 09:00) local time. Missing values are NEVER converted to zero.
"""
from __future__ import annotations

import csv
import io
import re
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

OBS_WINDOW_END_LOCAL = time(9, 0)
FTP_PRODUCT = "IDCKWCDEA0 (file id IDCKWCDE11) Daily Evapotranspiration tables, BoM anonymous FTP"
CDO_PRODUCT = "IDCJAC0009 BoM Climate Data Online daily rainfall (user supplied)"


class ObservationFormatError(ValueError):
    pass


def obs_window(label_date: date, tz: str, period_days: int = 1) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start_utc, end_utc) for a BoM 9am-to-9am total labelled ``label_date``."""
    zone = ZoneInfo(tz)
    end_local = datetime.combine(label_date, OBS_WINDOW_END_LOCAL, tzinfo=zone)
    start_local = datetime.combine(label_date - timedelta(days=period_days), OBS_WINDOW_END_LOCAL,
                                   tzinfo=zone)
    return pd.Timestamp(start_local).tz_convert("UTC"), pd.Timestamp(end_local).tz_convert("UTC")


def parse_idckwcdea0(blob: bytes) -> tuple[pd.DataFrame, dict]:
    """Parse one monthly IDCKWCDEA0 CSV. Returns (rows, header metadata).

    Validates that the rain column is the 0900-0900 total in millimetres; raises
    ObservationFormatError otherwise so that an unexpected format is never joined.
    """
    text = blob.decode("latin-1")
    lines = text.splitlines()
    meta: dict = {}
    for ln in lines[:12]:
        m = re.search(r"Issued at (\d{2}:\d{2}) GMT on \w+ (\d{2} \w+ \d{4})", ln)
        if m:
            meta["file_issued_at_utc"] = pd.Timestamp(
                datetime.strptime(f"{m.group(2)} {m.group(1)}", "%d %B %Y %H:%M")).tz_localize("UTC")
        m = re.search(r"Daily Evapotranspiration for (.+?) (Western Australia|New South Wales|Victoria|"
                      r"Queensland|South Australia|Tasmania|Northern Territory) for", ln)
        if m:
            meta["title_station_name"] = m.group(1).strip()
    reader = list(csv.reader(io.StringIO(text)))
    hdr_idx = next((i for i, r in enumerate(reader) if r and r[0].strip() == "Station Name"), None)
    if hdr_idx is None:
        raise ObservationFormatError("header row 'Station Name' not found")
    period_row, unit_row = reader[hdr_idx], reader[hdr_idx + 1]
    group_row = reader[hdr_idx - 1]
    if not (group_row[3].strip() == "Rain" and period_row[3].strip() == "0900-0900"
            and unit_row[3].strip() == "(mm)"):
        raise ObservationFormatError(
            f"rain column is not 'Rain / 0900-0900 / (mm)': {group_row[3]!r} {period_row[3]!r} {unit_row[3]!r}")
    meta["rain_column"] = "Rain 0900-0900 (mm)"
    rows = []
    for r in reader[hdr_idx + 2:]:
        if len(r) < 4 or not re.fullmatch(r"\d{2}/\d{2}/\d{4}", r[1].strip()):
            continue  # blank lines, 'Totals:' row
        raw = r[3].strip()
        rows.append({"station_name_in_file": r[0].strip(),
                     "label_date": datetime.strptime(r[1].strip(), "%d/%m/%Y").date(),
                     "raw_value": raw})
    return pd.DataFrame(rows), meta


def to_mm(raw: str) -> float | None:
    if raw is None or raw.strip() == "" or raw.strip() in {"-", "--"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def build_ftp_observations(monthly: list[tuple[str, bytes | None, str | None, str]],
                           station: dict, tz: str, first_day: date, last_day: date,
                           suspicious_mm: float) -> tuple[pd.DataFrame, list[dict]]:
    """Combine monthly files into one row per label date in [first_day, last_day].

    ``monthly`` items: (url, content or None, retrieved_at_utc, yyyymm).
    Days with no row are kept with a null value and status 'missing_row'.
    """
    problems: list[dict] = []
    frames = []
    for url, blob, retrieved, ym in monthly:
        if blob is None:
            problems.append({"stage": "observations", "item": ym, "url": url,
                             "problem": "monthly file not downloaded"})
            continue
        try:
            df, meta = parse_idckwcdea0(blob)
        except ObservationFormatError as exc:
            problems.append({"stage": "observations", "item": ym, "url": url, "problem": str(exc)})
            continue
        df["source_url"] = url
        df["retrieved_at_utc"] = retrieved
        df["file_issued_at_utc"] = meta.get("file_issued_at_utc")
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["station_name_in_file", "label_date", "raw_value", "source_url", "retrieved_at_utc",
                 "file_issued_at_utc"])

    # Rows for other station names would indicate a mixed file; never use them.
    other = raw[raw.station_name_in_file.str.upper() != station["station_name"].upper()]
    for _, r in other.iterrows():
        problems.append({"stage": "observations", "item": str(r.label_date), "url": r.source_url,
                         "problem": f"row for different station name {r.station_name_in_file!r} ignored"})
    raw = raw[raw.station_name_in_file.str.upper() == station["station_name"].upper()]

    # Duplicates: exact duplicates removed, conflicting ones flagged.
    raw = raw.drop_duplicates(subset=["label_date", "raw_value"])
    dup_dates = set(raw.label_date[raw.label_date.duplicated(keep=False)])
    raw = raw.drop_duplicates(subset=["label_date"], keep="last").set_index("label_date")

    records = []
    for d in pd.date_range(first_day, last_day, freq="D").date:
        start_utc, end_utc = obs_window(d, tz)
        rec = {"station_id": station["station_id"], "station_name": station["station_name"],
               "station_latitude": station["latitude"], "station_longitude": station["longitude"],
               "label_date_local": d, "window_start_utc": start_utc, "window_end_utc": end_utc,
               "window_start_local": start_utc.tz_convert(tz), "window_end_local": end_utc.tz_convert(tz),
               "period_days": 1, "source_product": FTP_PRODUCT,
               "observation_quality_flag": "not_provided_unqc_realtime",
               "trace_flag": None, "raw_value": None, "observed_precip_mm": None,
               "obs_status": "ok", "source_url": None, "retrieved_at_utc": None,
               "file_issued_at_utc": None}
        if d in raw.index:
            r = raw.loc[d]
            rec.update(raw_value=r.raw_value, source_url=r.source_url, retrieved_at_utc=r.retrieved_at_utc,
                       file_issued_at_utc=r.file_issued_at_utc)
            val = to_mm(r.raw_value)
            if d in dup_dates:
                rec["obs_status"] = "conflicting_duplicate"
            elif val is None:
                rec["obs_status"] = "missing_value" if r.raw_value.strip() == "" else "unparseable_value"
            elif val < 0:
                rec["obs_status"] = "negative_value"
            else:
                rec["observed_precip_mm"] = val
                if val > suspicious_mm:
                    rec["obs_status"] = "suspicious_extreme_kept"
        else:
            rec["obs_status"] = "missing_row"
        records.append(rec)
    return pd.DataFrame(records), problems


def load_cdo_csv(path: str, expected_station_id: str, tz: str) -> pd.DataFrame:
    """Load a manually downloaded BoM CDO daily rainfall CSV (IDCJAC0009).

    Columns: Product code, Bureau of Meteorology station number, Year, Month, Day,
    Rainfall amount (millimetres), Period over which rainfall was measured (days), Quality.
    """
    df = pd.read_csv(path, dtype=str)
    cols = {c.lower(): c for c in df.columns}
    need = ["product code", "bureau of meteorology station number", "year", "month", "day",
            "rainfall amount (millimetres)", "period over which rainfall was measured (days)", "quality"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise ObservationFormatError(f"{path}: not an IDCJAC0009 file, missing columns {missing}")
    df = df.rename(columns={cols[c]: c for c in need})
    stations = set(df["bureau of meteorology station number"].str.zfill(6))
    if stations != {expected_station_id}:
        raise ObservationFormatError(
            f"{path}: station(s) {stations} differ from primary station {expected_station_id}; "
            "refusing to mix stations")
    out = []
    for _, r in df.iterrows():
        d = date(int(r.year), int(r.month), int(r.day))
        period = r["period over which rainfall was measured (days)"]
        period = int(float(period)) if isinstance(period, str) and period.strip() else None
        val = to_mm(r["rainfall amount (millimetres)"] if isinstance(r["rainfall amount (millimetres)"], str) else "")
        start_utc, end_utc = obs_window(d, tz, period or 1)
        out.append({"label_date_local": d, "cdo_precip_mm": val, "cdo_period_days": period,
                    "cdo_quality": r.quality if isinstance(r.quality, str) else None,
                    "cdo_window_start_utc": start_utc, "cdo_window_end_utc": end_utc,
                    "cdo_source_file": str(path)})
    return pd.DataFrame(out)


def merge_cdo(obs: pd.DataFrame, cdo: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Overlay CDO values/quality flags on the FTP series for the SAME station.

    CDO values take precedence (quality-controlled product); disagreements are flagged.
    Multi-day accumulations (period > 1) get their true window and are not 24-h totals.
    """
    if cdo.empty:
        return obs
    m = obs.merge(cdo, on="label_date_local", how="left")
    has = m.cdo_precip_mm.notna() | m.cdo_period_days.notna()
    conflict = has & m.observed_precip_mm.notna() & m.cdo_precip_mm.notna() & \
        ((m.observed_precip_mm - m.cdo_precip_mm).abs() > 0.05)
    m.loc[conflict, "obs_status"] = "ftp_cdo_value_conflict_cdo_used"
    m.loc[has & m.cdo_precip_mm.notna(), "observed_precip_mm"] = m.cdo_precip_mm
    m.loc[has & m.cdo_precip_mm.notna() & m.obs_status.isin(["missing_row", "missing_value"]),
          "obs_status"] = "ok"
    m.loc[has, "observation_quality_flag"] = m.cdo_quality.map(
        {"Y": "Y_quality_controlled", "N": "N_not_yet_quality_controlled"}).fillna("cdo_flag_blank")
    m.loc[has, "source_product"] = CDO_PRODUCT
    multi = has & (m.cdo_period_days.fillna(1) > 1)
    m.loc[multi, "period_days"] = m.cdo_period_days
    m.loc[multi, "window_start_utc"] = m.cdo_window_start_utc
    m.loc[multi, "window_start_local"] = m.cdo_window_start_utc.map(
        lambda t: t.tz_convert(tz) if pd.notna(t) else t)
    m.loc[multi, "obs_status"] = "multi_day_accumulation"
    return m.drop(columns=[c for c in m.columns if c.startswith("cdo_window")])
