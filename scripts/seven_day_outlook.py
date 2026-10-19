"""7-day forecast for Perth, written in the Bureau of Meteorology's plain style, with a short explanation of what our
testing found. Writes reports/outlook/outlook_site.json (exported to the website) plus Markdown and HTML copies.

Two kinds of information, always labelled:
* Our calibrated forecast (from the newest served prediction, at the Perth Metro gauge), where the pipeline issued
  one and its inputs passed verification. Only these days get a "chance of any rain" percentage.
* Raw guidance from five global models (ECMWF, ECMWF AI, DWD ICON, NOAA GFS, JMA) via Open-Meteo's forecast API, at
  the Perth city coordinates. Raw guidance is never turned into a percentage: it is shown as model agreement.

"Possible rainfall" follows the Bureau's convention: a 50% chance of at least the first number and a 25% chance of
at least the second. For calibrated days those are our median and 75th percentile; for raw days the range of the
models' totals is shown instead and labelled as such.
"""
from __future__ import annotations

import json
import time
import tomllib
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "outlook"
TZ = "Australia/Perth"
UA = {"User-Agent": "perthrain-dataset/0.1 (non-commercial forecast verification research)"}
MODELS = {"ecmwf_ifs025": "ECMWF", "ecmwf_aifs025_single": "ECMWF AI", "icon_global": "German ICON",
          "gfs_global": "US GFS", "jma_gsm": "Japanese JMA"}
COMPASS = ["northerly", "north to northeasterly", "northeasterly", "east to northeasterly", "easterly",
           "east to southeasterly", "southeasterly", "south to southeasterly", "southerly", "south to southwesterly",
           "southwesterly", "west to southwesterly", "westerly", "west to northwesterly", "northwesterly",
           "north to northwesterly"]
FINE = ("Sunny", "Mostly sunny", "Partly cloudy", "Cloudy")


# ------------------------------------------------------------------------------------------------ data
def fetch(lat, lon):
    p = {"latitude": lat, "longitude": lon, "forecast_days": 9, "timezone": "GMT", "models": ",".join(MODELS),
         "hourly": "precipitation,temperature_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m,cloud_cover"}
    for attempt in range(3):                       # transient 5xx: back off and retry, then give up loudly
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=p, headers=UA, timeout=60)
        if r.status_code < 500:
            break
        time.sleep(10 * (attempt + 1))
    r.raise_for_status()
    h = pd.DataFrame(r.json()["hourly"])
    h["time"] = pd.to_datetime(h.time).dt.tz_localize("UTC")
    return h.set_index("time"), datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ours(slug, name):
    """Our calibrated forecasts by rain-day END date (shortest lead first), excluding rows that failed verification."""
    d = ROOT / "data" / slug / "predictions"
    if not (d / "latest.csv").exists():
        return {}, None
    pred = pd.read_csv(d / "latest.csv")
    made = sorted(d.glob("forecast_*.csv"))[-1].stem.split("_")[1]
    ok = pred[pred.status == "ok"].copy()
    st = ROOT / "reports" / "served_row_status.csv"
    if st.exists():
        s_ = pd.read_csv(st)
        bad = {(str(r.gauge_day), r.lead) for r in s_[(s_.location == name) & (s_.row_status == "FAILED")].itertuples()}
        ok = ok[[(str(a), b) not in bad for a, b in zip(ok.label_date_local, ok.lead_group)]]
    if ok.empty:
        return {}, made
    ok["lead_n"] = ok.lead_group.str[-1].astype(int)
    best = ok.sort_values("lead_n").groupby("label_date_local").first()
    return ({pd.Timestamp(k).date(): {"p": float(r.p_rain_ge_0_2mm), "q50": float(r.q50_mm), "q75": float(r.q75_mm),
                                      "backup": str(r.route).startswith("fallback"), "lead": r.lead_group}
             for k, r in best.iterrows()}, made)


def rain_window(h, start_day):
    """Model totals for the rain day that STARTS at 9am on ``start_day`` (BoM 9am-to-9am), complete hours only."""
    start = pd.Timestamp(start_day, tz=TZ) + pd.Timedelta(hours=9)
    end = start + pd.Timedelta(days=1)
    hrs = h.loc[(h.index > start.tz_convert("UTC")) & (h.index <= end.tz_convert("UTC"))]
    tot = {}
    for m in MODELS:
        v = hrs[f"precipitation_{m}"]
        tot[m] = float(v.sum()) if len(v) == 24 and v.notna().all() else np.nan
    return tot


def day_weather(h, day):
    local = h.tz_convert(TZ)
    d = local[local.index.date == day]
    daytime = d[(d.index.hour >= 9) & (d.index.hour <= 18)]

    def med(col, frame, how):
        vals = []
        for m in MODELS:
            s = frame[f"{col}_{m}"].dropna()
            if len(frame) and len(s) >= max(6, len(frame) // 2):
                vals.append(getattr(s, how)())
        return float(np.median(vals)) if vals else np.nan

    out = {"max": med("temperature_2m", d, "max"), "min": med("temperature_2m", d, "min"),
           "gust": med("wind_gusts_10m", d, "max"), "cloud": med("cloud_cover", daytime, "mean")}
    # daytime wind: middle model at each hour, then the usual range and the prevailing direction
    speed = pd.concat([daytime[f"wind_speed_10m_{m}"] for m in MODELS], axis=1).median(axis=1).dropna()
    dirs = pd.concat([daytime[f"wind_direction_10m_{m}"] for m in MODELS], axis=1).dropna(how="all")
    if len(speed) >= 4 and len(dirs):
        rad = np.deg2rad(dirs.to_numpy(float))
        mean_dir = (np.rad2deg(np.arctan2(np.nanmean(np.sin(rad)), np.nanmean(np.cos(rad)))) + 360) % 360
        out["wind"] = (COMPASS[int((mean_dir + 11.25) // 22.5) % 16], float(speed.quantile(0.25)),
                       float(speed.quantile(0.75)))
    return out


# ------------------------------------------------------------------------------------------------ wording
def round5(x):
    return int(5 * round(x / 5))


def wind_text(wx):
    if "wind" not in wx:
        return ""
    name, lo, hi = wx["wind"]
    lo, hi = max(round5(lo), 5), round5(hi)
    if hi < 10:
        return "Winds light."
    rng = f"{lo} to {hi}" if hi > lo else f"around {hi}"
    return f"Winds {name} {rng} km/h."


def precis(wx, wet, n, med, our):
    """BoM-style short forecast word(s)."""
    if our is not None:
        p = our["p"]
        if p >= 0.7:
            return "Showers" if our["q50"] < 5 else "Rain"
        if p >= 0.5:
            return "Shower or two"
        if p >= 0.3:
            return "Possible shower"
    elif n:
        if wet >= 4:
            return "Showers" if (np.isnan(med) or med < 10) else "Rain"
        if wet >= 3:
            return "Shower or two"
        if wet >= 1:
            return "Possible shower"
    c = wx["cloud"]
    if np.isnan(c):
        return "Partly cloudy"
    return "Sunny" if c < 25 else "Mostly sunny" if c < 50 else "Partly cloudy" if c < 75 else "Cloudy"


def mm_text(x):
    """Whole millimetres, rounding half up (Python's round() would turn 0.5 mm into 0); 10 mm and above to the
    nearest 5, as the Bureau does."""
    if x < 0.5:
        return "0"
    if x < 10:
        return str(int(np.floor(x + 0.5)))
    return str(int(5 * np.floor(x / 5 + 0.5)))


def fmt_pct(p):
    v = 5 * round(100 * p / 5)              # the Bureau rounds rain chances to the nearest 5%
    return "less than 5%" if v < 5 else "more than 95%" if v > 95 else f"{v}%"


def findings(perf, challenge):
    """Plain-language findings from the published held-out results (computed, not hard-coded)."""
    g = next(x for x in perf["gauges"] if x["gauge_name"] == "Perth Metro")
    better = [1 - l["brier_ge_0_2mm"]["calibrated"] / l["brier_ge_0_2mm"]["seasonal_climatology"] for l in g["leads"]]
    days = min(l["n_days"] for l in g["leads"])
    first, last = g["leads"][0]["test_first"], g["leads"][0]["test_last"]
    amount = [p for grp in (challenge.get("historical") or {}).get("groups", []) for p in grp["paired"]
              if p.get("score", "absolute error") != "CRPS" and str(p.get("first_key", "")).startswith("champion")]
    close = sum(p["verdict"] in ("inconclusive", "insufficient evidence") for p in amount)
    t = pd.Timestamp
    return [
        f"Our rain chances were tested on about {days} past days that were kept aside while the model was built "
        f"({t(first):%B} to {t(last):%B %Y}), at the Perth Metro gauge. Judged on whether it rained or not, they "
        f"scored about {round(100 * min(better))} to {round(100 * max(better))}% better than simply using the usual "
        f"chance of rain for that time of year.",
        (f"For how much rain falls, our forecasts did about as well as the weather models themselves: "
         f"{close} of {len(amount)} side-by-side comparisons with them were too close to call." if amount else
         "For how much rain falls, we have not shown a clear advantage over the weather models themselves."),
        "So the rain chances are the most useful part of this forecast for the next two or three days. Further out, "
        "the forecast shows how many of five international weather models expect rain. That is a guide, not a "
        "probability.",
        "Rain is measured at the Bureau's Perth Metro gauge in Mount Lawley, from 9am to 9am. Showers are patchy, so "
        "your suburb can get more or less.",
    ]


def headline(days):
    named = lambda d: f"{pd.Timestamp(d['date']):%A}"  # noqa: E731
    temps = [d for d in days if d["max_c"] is not None]
    hottest = max(temps, key=lambda d: d["max_c"]) if temps else None
    wet = [d for d in days if d["precis"] not in FINE]
    if wet and hottest and pd.Timestamp(hottest["date"]) < pd.Timestamp(wet[0]["date"]):
        return f"Fine and warming to {hottest['max_c']} degrees on {named(hottest)}, then showers from {named(wet[0])}"
    if wet:
        return f"Showers likely on {named(wet[0])}" + (f", top of {hottest['max_c']} degrees on {named(hottest)}"
                                                       if hottest else "")
    return f"Mostly fine week, reaching {hottest['max_c']} degrees on {named(hottest)}" if hottest else "Mostly fine week"


# ------------------------------------------------------------------------------------------------ main
def main():
    cfg = tomllib.loads((ROOT / "config.toml").read_text())
    perth = next(e for e in cfg["locations"] if e["name"] == "Perth")
    h, retrieved = fetch(perth["latitude"], perth["longitude"])
    our, made = ours("perth", "Perth")
    today = pd.Timestamp.now(tz=TZ).date()
    days = []
    for k in range(1, 8):
        day = today + pd.Timedelta(days=k)
        wx = day_weather(h, day)
        tot = rain_window(h, day)
        vals = [v for v in tot.values() if not np.isnan(v)]
        n, wet = len(vals), sum(v >= 1.0 for v in vals)
        med = float(np.median(vals)) if vals else np.nan
        o = our.get(day + pd.Timedelta(days=1))           # our label = the day the 9am-9am window ENDS
        word = precis(wx, wet, n, med, o)
        text = [f"{word}."]
        wt = wind_text(wx)
        if wt and not np.isnan(wx.get("gust", np.nan)) and wx["gust"] >= 50:
            text.append(f"{wt[:-1]}, with gusts to about {round5(wx['gust'])} km/h.")
        elif wt:
            text.append(wt)
        entry = {"date": str(day), "precis": word, "forecast": " ".join(text),
                 "min_c": None if np.isnan(wx["min"]) else round(wx["min"]),
                 "max_c": None if np.isnan(wx["max"]) else round(wx["max"]),
                 "calibrated": o is not None, "models_with_rain": wet, "models_total": n}
        if o is not None:
            entry["chance_of_any_rain"] = fmt_pct(o["p"])
            lo, hi = mm_text(o["q50"]), mm_text(max(o["q75"], o["q50"]))
            entry["possible_rainfall"] = f"{lo} mm" if lo == hi else f"{lo} to {hi} mm"
            entry["rain_source"] = "Our calibrated forecast for the Perth Metro gauge" + (
                " (backup model; its accuracy has not been measured yet)" if o["backup"] else "")
            entry["model_agreement"] = None
        else:
            entry["chance_of_any_rain"] = None
            dry = bool(vals) and max(vals) < 0.5
            entry["model_agreement"] = ("none of the five weather models show rain" if dry else
                                        f"{wet} of {n} weather models show 1 mm or more" if n
                                        else "no complete model guidance yet")
            entry["possible_rainfall"] = (None if (not vals or dry) else
                                          f"the models range from {mm_text(min(vals))} to {mm_text(max(vals))} mm")
            entry["rain_source"] = "Raw guidance from five weather models (not a probability)"
        days.append(entry)

    perf = json.loads((ROOT / "website" / "data" / "v1" / "performance.json").read_text(encoding="utf-8"))
    ch_p = ROOT / "website" / "data" / "v1" / "challenge.json"
    challenge = json.loads(ch_p.read_text(encoding="utf-8")) if ch_p.exists() else {}
    made_utc = (pd.Timestamp(pd.to_datetime(made, format="%Y%m%dT%H%MZ"), tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
                if made else None)
    site = {"schema_version": 2, "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_guidance_retrieved_utc": retrieved, "calibrated_forecast_made_utc": made_utc,
            "location": "Perth", "headline": headline(days), "days": days, "findings": findings(perf, challenge),
            "how_to_read": [
                "Chance of any rain: the chance of at least 0.2 mm at the gauge, from our calibrated forecast. It is "
                "shown only for the days we can forecast this way.",
                "Possible rainfall: a 50% chance of at least the first number and a 25% chance of at least the "
                "second, as the Bureau uses it. On days marked as model guidance, it is the range of the five "
                "models' totals instead.",
                "Each day's rain covers 9am that day to 9am the next.",
                "Temperatures and winds are the middle value of the five weather models."],
            "sources": "Forecast data: Open-Meteo.com (CC BY 4.0), from ECMWF, DWD, NOAA and JMA. Observations: "
                       "Bureau of Meteorology. Not an official forecast: for warnings, use the Bureau of Meteorology."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "outlook_site.json").write_text(json.dumps(site, indent=1, ensure_ascii=False, allow_nan=False),
                                           encoding="utf-8")

    md = ["# Perth 7-day forecast\n", f"**{site['headline']}**\n"]
    for d in days:
        t = pd.Timestamp(d["date"])
        md.append(f"### {t:%A} {t.day} {t:%B}")
        md.append(f"{d['forecast']} Min {d['min_c']}, max {d['max_c']}.  ")
        if d["calibrated"]:
            md.append(f"Chance of any rain: {d['chance_of_any_rain']}. Possible rainfall: {d['possible_rainfall']}.  ")
        else:
            md.append(f"Rain: {d['model_agreement']}" + (f"; {d['possible_rainfall']}" if d["possible_rainfall"] else "")
                      + ".  ")
        md.append(f"*{d['rain_source']}*\n")
    md += ["## What we found\n", *[f"- {f}" for f in site["findings"]], "", "## How to read this\n",
           *[f"- {x}" for x in site["how_to_read"]], "", site["sources"]]
    (OUT / "seven_day_outlook.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "seven_day_outlook.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Perth 7-day forecast</title><pre>"
        + escape("\n".join(md)) + "</pre>", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
