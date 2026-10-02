"""7-day outlook written as an easy-to-read local weather article (Markdown + HTML).

Two kinds of information, always labelled:
* Our calibrated rain chances (gauge-based) from the newest served prediction, where the pipeline issued one.
* Raw multi-model guidance from Open-Meteo's live forecast API (ECMWF IFS, ECMWF AIFS, DWD ICON, NOAA GFS, JMA GSM)
  at each suburb, for every day of the week. Raw guidance is uncalibrated and is reported as model agreement,
  never as a probability.

Writes reports/outlook/seven_day_outlook.{md,html,json}. Read-only with respect to models and predictions.
"""
from __future__ import annotations

import json
import sys
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


def fetch(lat, lon):
    p = {"latitude": lat, "longitude": lon, "hourly": "precipitation,temperature_2m,wind_gusts_10m,cloud_cover",
         "forecast_days": 9, "timezone": "GMT", "models": ",".join(MODELS)}
    import time
    for attempt in range(3):                       # transient 5xx: back off and retry, then give up loudly
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=p, headers=UA, timeout=60)
        if r.status_code < 500:
            break
        time.sleep(10 * (attempt + 1))
    r.raise_for_status()
    h = pd.DataFrame(r.json()["hourly"])
    h["time"] = pd.to_datetime(h.time).dt.tz_localize("UTC")
    return h.set_index("time"), datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rain_windows(h, start_date, days=7):
    """9am-9am totals labelled by END date (BoM convention); a total needs all 24 hours."""
    rows = []
    for d in pd.date_range(start_date, periods=days, freq="D"):
        end = pd.Timestamp(d.date(), tz=TZ) + pd.Timedelta(hours=9)
        start = end - pd.Timedelta(days=1)
        hrs = h.loc[(h.index > start.tz_convert("UTC")) & (h.index <= end.tz_convert("UTC"))]
        tot = {}
        for m in MODELS:
            v = hrs[f"precipitation_{m}"]
            tot[m] = float(v.sum()) if len(v) == 24 and v.notna().all() else np.nan
        rows.append({"label": d.date(), "start": start, "end": end, **tot})
    return pd.DataFrame(rows)


def day_weather(h, day):
    local = h.tz_convert(TZ)
    d = local[local.index.date == day]
    day_hours = d[(d.index.hour >= 7) & (d.index.hour <= 18)]
    out = {}
    for k, col in (("max", "temperature_2m"), ("min", "temperature_2m"), ("gust", "wind_gusts_10m"), ("cloud", "cloud_cover")):
        vals = []
        for m in MODELS:
            s = (d if k != "cloud" else day_hours)[f"{col}_{m}"].dropna()
            if len(s) >= (20 if k in ("max", "min") else 8):
                vals.append(s.max() if k in ("max", "gust") else s.min() if k == "min" else s.mean())
        out[k] = float(np.median(vals)) if vals else np.nan
        out[f"{k}_n"] = len(vals)
    return out


def sky(cloud):
    if np.isnan(cloud):
        return "Sky unclear in the models"
    return "Sunny" if cloud < 25 else "Mostly sunny" if cloud < 50 else "Partly cloudy" if cloud < 75 else "Cloudy"


def rain_phrase(row, our):
    vals = [row[m] for m in MODELS if not np.isnan(row[m])]
    n = len(vals)
    if n == 0:
        return "no complete model guidance", 0, n, np.nan
    wet = sum(v >= 1.0 for v in vals)
    med = float(np.median(vals))
    if our is not None:
        p = our["p"]
        txt = (f"our calibrated chance of rain at the gauge is {_pct(p)}"
               + (" (from our backup model, whose accuracy has not been scored yet)" if our.get("fallback") else ""))
    elif wet == 0:
        txt = "none of the models show a meaningful fall"
    elif wet <= 2:
        txt = f"only {wet} of {n} models show 1 mm or more"
    else:
        txt = f"{wet} of {n} models show 1 mm or more (middle estimate {med:.0f} mm)" if med >= 1 else \
              f"{wet} of {n} models show 1 mm or more"
    return txt, wet, n, med


def _pct(p):
    return "under 1%" if p < 0.01 else "over 99%" if p > 0.99 else f"{round(p * 100)}%"


def ours(slug, name):
    p = ROOT / "data" / slug / "predictions" / "latest.csv"
    if not p.exists():
        return {}, None
    d = pd.read_csv(p)
    made = sorted((ROOT / "data" / slug / "predictions").glob("forecast_*.csv"))[-1].stem.split("_")[1]
    ok = d[d.status == "ok"].copy()
    # never quote a forecast whose inputs failed verification (scripts/verify_served_rows.py)
    st = ROOT / "reports" / "served_row_status.csv"
    if st.exists():
        s_ = pd.read_csv(st)
        bad = {(str(r.gauge_day), r.lead) for r in s_[(s_.location == name) & (s_.row_status == "FAILED")].itertuples()}
        ok = ok[[(str(a), b) not in bad for a, b in zip(ok.label_date_local, ok.lead_group)]]
    ok["lead_n"] = ok.lead_group.str[-1].astype(int)
    best = ok.sort_values("lead_n").groupby("label_date_local").first()
    return ({pd.Timestamp(k).date(): {"p": r.p_rain_ge_0_2mm, "p1": r.p_ge_1mm, "exp": r.mean_estimate_mm,
                                      "fallback": str(r.route).startswith("fallback")}
             for k, r in best.iterrows()}, made)


def main():
    cfg = tomllib.loads((ROOT / "config.toml").read_text())
    now_local = pd.Timestamp.now(tz=TZ)
    first_day = (now_local + pd.Timedelta(days=1)).date()          # the next seven calendar days
    first_label = first_day + pd.Timedelta(days=1)                   # rain window 9am that day -> 9am next day
    data, retrieved = {}, None
    for e in cfg["locations"]:
        slug = e["name"].lower().replace(" ", "_")
        st = json.loads((ROOT / "reports" / slug / "run_summary.json").read_text())["station"]
        h, retrieved = fetch(e["latitude"], e["longitude"])
        o, made = ours(slug, e["name"])
        data[e["name"]] = {"h": h, "rain": rain_windows(h, first_label), "ours": o, "made": made,
                           "gauge": st["station_name"].title(), "gauge_km": round(float(st["distance_km"]), 1)}

    perth = data["Perth"]
    days = []
    for _, row in perth["rain"].iterrows():
        label = row.label - pd.Timedelta(days=1)      # calendar day on which the 9am-9am window starts
        wx = day_weather(perth["h"], label)
        our = perth["ours"].get(row.label)
        txt, wet, n, med = rain_phrase(row, our)
        days.append({"label": label, "wx": wx, "rain_txt": txt, "wet": wet, "n": n, "med": med, "ours": our,
                     "row": row})

    hottest = max(days, key=lambda d: d["wx"]["max"] if not np.isnan(d["wx"]["max"]) else -99)
    wettest = max(days, key=lambda d: (d["wet"], 0 if np.isnan(d["med"]) else d["med"]))
    any_rain = wettest["wet"] >= 3
    later_cool = [d for d in days if d["label"] > hottest["label"] and d["wx"]["max"] <= hottest["wx"]["max"] - 5]
    if any_rain and wettest["label"] > hottest["label"] and later_cool:
        head = (f"Warm {hottest['label']:%A} for Perth before rain and a cooler change")
    elif any_rain:
        head = f"Mostly settled week for Perth, with showers likely on {wettest['label']:%A}"
    else:
        head = f"Dry week ahead for Perth, with the mercury reaching {hottest['wx']['max']:.0f}°C on {hottest['label']:%A}"
    stand = (f"Plenty of sunshine for most of the week, with the city climbing to about {hottest['wx']['max']:.0f}°C on "
             f"{hottest['label']:%A}. ")
    if any_rain:
        first_wet = next(d for d in days if d["wet"] >= 3)
        when_txt = (f"on {wettest['label']:%A}" if first_wet is wettest
                    else f"from {first_wet['label']:%A}, heaviest on {wettest['label']:%A}")
        stand += (f"Most of the weather models then bring rain {when_txt}"
                  + (f", with the top easing to around {wettest['wx']['max']:.0f}°C." if not np.isnan(wettest['wx']['max']) else "."))
    else:
        stand += "No day stands out for rain."

    lines_md, lines_html, site_days = [], [], []
    for d in days:
        wx = d["wx"]
        when = f"{d['label']:%A %-d %B}" if sys.platform != "win32" else f"{d['label']:%A} {d['label'].day} {d['label']:%B}"
        cond = sky(wx["cloud"])
        if d["wet"] >= 3:
            cond = "Showers" if (not np.isnan(d["med"]) and d["med"] < 5) else "Rain"
        elif d["wet"] >= 1 or (d["ours"] and d["ours"]["p"] >= 0.3):
            cond += ", chance of a shower"
        temps = (f"{wx['min']:.0f}–{wx['max']:.0f}°C" if not np.isnan(wx["max"]) else "temperatures not yet available")
        gust = f" Gusts to about {wx['gust']:.0f} km/h." if not np.isnan(wx["gust"]) and wx["gust"] >= 45 else ""
        tag = ""
        rain = f"Rain from 9am today to 9am tomorrow: {d['rain_txt']}{tag}."
        lines_md.append(f"**{when}** — {cond}. {temps}.{gust} {rain}")
        site_days.append({"date": str(d["label"]), "condition": cond, "temps": temps, "gust": gust.strip(),
                          "rain": rain, "calibrated": bool(d["ours"]), "models_with_1mm": int(d["wet"]),
                          "models_total": int(d["n"])})
        lines_html.append(f"<li><b>{escape(when)}</b><span class='cond'>{escape(cond)} · {escape(temps)}</span>"
                          f"<span class='rain'>{escape(rain.replace('*', ''))}</span>{escape(gust)}</li>")

    burbs = []
    for name in ("Clarkson", "Ocean Reef", "Fremantle"):
        r = data[name]["rain"]
        wetdays = [f"{(row.label - pd.Timedelta(days=1)):%A}" for _, row in r.iterrows() if sum(row[m] >= 1 for m in MODELS if not np.isnan(row[m])) >= 3]
        burbs.append(f"**{name}:** " + (f"models agree on rain {', '.join(wetdays)}." if wetdays
                                         else "no day where most models show 1 mm or more.")
                     + f" (Rain gauge used for our calibrated forecast: {data[name]['gauge']}, {data[name]['gauge_km']} km away.)")

    calib_days = ", ".join(f"{(pd.Timestamp(d) - pd.Timedelta(days=1)):%A}" for d in sorted(perth["ours"]))
    made_local = pd.Timestamp(pd.to_datetime(perth["made"], format="%Y%m%dT%H%MZ"), tz="UTC").tz_convert(TZ)
    made_txt = f"{made_local:%I:%M%p}".lstrip("0").lower() + f" {made_local:%A}"
    ret_local = pd.Timestamp(retrieved).tz_convert(TZ)
    ret_txt = f"{ret_local:%I:%M%p}".lstrip("0").lower() + f" {ret_local:%A}"
    method = (f"Our own calibrated rain chances cover only the days the pipeline could issue ({calib_days or 'none today'}), "
              f"from the forecast made at {made_txt} (Perth time), and describe the Bureau of Meteorology gauge, not your backyard. "
              f"Every other figure is raw guidance from five global weather models (ECMWF, ECMWF AI, German ICON, US GFS, "
              f"Japanese JMA) via Open-Meteo, retrieved at {ret_txt}, taken at each suburb. Temperatures are the middle value "
              f"across the models. Model agreement is not a probability. Rain days run 9am to 9am, as the Bureau reports "
              f"them, and are named by the day they end. This is not an official forecast: for warnings, use the Bureau of "
              f"Meteorology.")
    method = method.replace("Rain days run 9am to 9am, as the Bureau reports them, and are named by the day they end.",
                            "Rain is measured 9am to 9am, as the Bureau reports it; each day's rain line covers 9am that "
                            "day to 9am the next.")

    md = [f"# {head}\n", f"*{stand}*\n", "## Perth: the week ahead\n", *[f"- {x}" for x in lines_md], "",
          "## Around the suburbs\n", *[f"- {x}" for x in burbs], "", "## How this was put together\n", method, "",
          "Forecast data: Open-Meteo.com (CC BY 4.0), ECMWF, DWD, NOAA, JMA. Observations: Bureau of Meteorology."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "seven_day_outlook.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    html = f"""<!doctype html><html lang="en-AU"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Perth 7-Day Outlook</title>
<link rel="stylesheet" href="../../website/assets/style.css">
<style>main{{max-width:760px}} .wk{{list-style:none;padding:0;display:grid;gap:10px}}
.wk li{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;display:grid;gap:4px}}
.wk .cond{{color:var(--teal);font-weight:600}} .wk .rain{{color:var(--ink-2);font-size:15px}}
.stand{{font-size:18px;color:var(--ink-2)}}</style></head><body><main>
<h1>{escape(head)}</h1><p class="stand">{escape(stand)}</p>
<h2>Perth: the week ahead</h2><ul class="wk">{''.join(lines_html)}</ul>
<h2>Around the suburbs</h2><ul>{''.join('<li>' + escape(b.replace('**', '')) + '</li>' for b in burbs)}</ul>
<h2>How this was put together</h2><p class="small muted">{escape(method)}</p>
<p class="small muted">Forecast data: Open-Meteo.com (CC BY 4.0), ECMWF, DWD, NOAA, JMA. Observations: Bureau of Meteorology.</p>
</main></body></html>"""
    (OUT / "seven_day_outlook.html").write_text(html, encoding="utf-8")
    made_utc = pd.Timestamp(pd.to_datetime(perth["made"], format="%Y%m%dT%H%MZ"), tz="UTC") if perth["made"] else None
    site = {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_guidance_retrieved_utc": retrieved,
            "calibrated_forecast_made_utc": made_utc.strftime("%Y-%m-%dT%H:%M:%SZ") if made_utc is not None else None,
            "headline": head, "standfirst": stand, "location": "Perth", "days": site_days,
            "suburbs": [b.replace("**", "") for b in burbs], "method": method,
            "sources": "Forecast data: Open-Meteo.com (CC BY 4.0), ECMWF, DWD, NOAA, JMA. Observations: Bureau of "
                       "Meteorology. Not an official forecast."}
    (OUT / "outlook_site.json").write_text(json.dumps(site, indent=1, ensure_ascii=False, allow_nan=False),
                                           encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
