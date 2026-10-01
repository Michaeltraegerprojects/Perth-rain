"""Build reports/perth_rain_report.html from the project's outputs (forecast, verification, held-out scores,
paired intervals, review impact). Static HTML: everything is rendered here, no script runs in the page.

Reads only files written by the package or by the diagnostic scripts; computes no model results itself.
"""
import html
import json
import re
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perthrain.config import slug  # noqa: E402

E = html.escape
LOCS = [(e["name"], slug(e["name"])) for e in tomllib.loads((ROOT / "config.toml").read_text())["locations"]]
GAUGE = {}
for name, s in LOCS:
    st = json.loads((ROOT / "reports" / s / "run_summary.json").read_text())["station"]
    GAUGE[s] = {"id": st["station_id"], "name": st["station_name"].title(), "km": float(st["distance_km"]), "loc": name}

# one entry per physical gauge, listing the locations it serves
gauges = {}
for name, s in LOCS:
    g = gauges.setdefault(GAUGE[s]["id"], {"name": GAUGE[s]["name"], "slug": s, "serves": []})
    g["serves"].append(f"{name} ({GAUGE[s]['km']:.1f} km)")


def fmt_day(d):
    t = pd.Timestamp(d)
    return f"{t:%a} {t.day} {t:%b}"


def pct(x):
    return "–" if pd.isna(x) else f"{100 * x:.0f}%"


def mm(x, nd=1):
    return "–" if pd.isna(x) else f"{x:.{nd}f}"


def signed(x, nd=2):
    return "–" if pd.isna(x) else f"{x:+.{nd}f}"


# ------------------------------------------------------------------------------------------------ forecast
status = pd.read_csv(ROOT / "reports" / "served_row_status.csv")
verif = pd.read_csv(ROOT / "reports" / "served_row_verification.csv")
latest_rec = sorted((ROOT / "data" / "perth" / "predictions").glob("forecast_*.csv"))[-1]
issued = pd.Timestamp(pd.to_datetime(latest_rec.stem.split("_")[1], format="%Y%m%dT%H%MZ"), tz="UTC")
issued_local = issued.tz_convert("Australia/Perth")

# Perth and Ocean Reef share one gauge: their served rows must be identical, otherwise both are shown
a = pd.read_csv(ROOT / "data" / "perth" / "predictions" / "latest.csv")
b = pd.read_csv(ROOT / "data" / "ocean_reef" / "predictions" / "latest.csv")
num = ["p_rain_ge_0_2mm", "p_ge_1mm", "p_ge_5mm", "q50_mm", "q90_mm", "mean_estimate_mm"]
assert (a.status == b.status).all() and np.allclose(a[num].fillna(-1), b[num].fillna(-1), rtol=0, atol=1e-12)


def forecast_block(gid, g):
    s = g["slug"]
    loc_name = GAUGE[s]["loc"]
    d = pd.read_csv(ROOT / "data" / s / "predictions" / "latest.csv")
    rows = []
    for day, grp in d.groupby("label_date_local", sort=True):
        served = grp[grp.status == "ok"].sort_values("lead_group")
        w = grp.iloc[0]
        ws, we = pd.Timestamp(w.window_start_local), pd.Timestamp(w.window_end_local)
        window = f"9am {ws:%a} {ws.day} → 9am {we:%a} {we.day}"
        if served.empty:
            rows.append(f"""<tr class="na"><th scope="row"><span class="day">{fmt_day(day)}</span><span class="win">{window}</span></th>
<td colspan="8" class="na-text">Not available yet. The model runs this day needs had not been issued at {issued_local:%H:%M} AWST.</td></tr>""")
            continue
        for i, (_, r) in enumerate(served.iterrows()):
            st = status[(status.location == loc_name) & (status.gauge_day == day) & (status.lead == r.lead_group)]
            st = st.row_status.iloc[0] if len(st) else "UNVERIFIED"
            v = verif[(verif.location == loc_name) & (verif.gauge_day == day) & (verif.lead == r.lead_group)]
            why = "; ".join(f"{x.input}: {x.detail}" for x in v.itertuples() if x.status != "VERIFIED")
            runs = "; ".join(sorted({x.detail for x in v.itertuples() if str(x.input).startswith("sr_")}))
            chip = {"VERIFIED": ("ok", "Verified"), "UNVERIFIED": ("warn", "Unverified"), "FAILED": ("bad", "Failed")}[st]
            title = E(why or runs or "all inputs checked against the named model runs")
            lead = {"day2": "2 days", "day3": "3 days", "day1": "1 day"}[r.lead_group]
            head = (f'<th scope="row" rowspan="{len(served)}"><span class="day">{fmt_day(day)}</span>'
                    f'<span class="win">{window}</span></th>') if i == 0 else ""
            p = r.p_rain_ge_0_2mm
            rows.append(f"""<tr>{head}
<td class="lead">{lead}</td>
<td class="prob"><span class="bar" style="--p:{p:.3f}"><span></span></span><span class="num">{pct(p)}</span></td>
<td class="num">{pct(r.p_ge_1mm)}</td><td class="num">{pct(r.p_ge_5mm)}</td>
<td class="num">{mm(r.q50_mm)}</td><td class="num">{mm(r.q90_mm)}</td><td class="num">{mm(r.mean_estimate_mm, 2)}</td>
<td class="model"><code>{E(r.feature_set_used)}</code><span class="route">{E(r.route)}</span>
<span class="chip {chip[0]}" title="{title}">{chip[1]}</span></td></tr>""")
    return f"""<section class="gauge" aria-labelledby="g{gid}">
<div class="gauge-head"><h3 id="g{gid}">{E(g['name'])} <span class="gid">BoM {gid}</span></h3>
<p class="serves">Gauge used for {E(', '.join(g['serves']))}</p></div>
<div class="scroll"><table class="fc">
<thead><tr><th scope="col">Rain day</th><th scope="col">Lead</th><th scope="col">Chance ≥ 0.2 mm</th>
<th scope="col">≥ 1 mm</th><th scope="col">≥ 5 mm</th><th scope="col">Median mm</th><th scope="col">90th pct mm</th>
<th scope="col">Expected mm</th><th scope="col">Model · status</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></section>"""


forecast_html = "\n".join(forecast_block(gid, g) for gid, g in gauges.items())
n_served = int((status.row_status != "").sum()) // 2 + 0   # rows per distinct gauge counted below
unver = status[status.row_status == "UNVERIFIED"]
unver_note = ""
if len(unver):
    x = verif[(verif.status == "UNVERIFIED")].iloc[0]
    unver_note = (f"Rows marked <b>Unverified</b> used a JMA GSM input whose hours depend on the "
                  f"{E(x.detail.split('not archived: ')[-1])} run, which is not in Open-Meteo's single-run archive. The "
                  f"archived runs that are available match hour for hour ({E(x.detail.split('; ')[1])}), but the missing one "
                  "cannot be checked, so these rows are kept but not counted as fully verified.")

# ------------------------------------------------------------------------------------------------ held-out
T = pd.read_csv(ROOT / "reports" / "acceptance_heldout_table.csv")
ho_blocks = []
for gid, g in gauges.items():
    s = g["slug"]
    m = pd.read_csv(ROOT / "reports" / s / "calibration_metrics.csv")
    trs = []
    for lead in ("day1", "day2", "day3"):
        t = T[(T.gauge.str.startswith(gid)) & (T.lead == lead)]
        h = m[(m.lead == lead) & (m.split == "holdout") & (m.method == "hurdle")].iloc[0]
        c = m[(m.lead == lead) & (m.split == "holdout") & (m.method == "climatology")].iloc[0]

        def pick(meth, col):
            r = t[t.method == meth]
            return r[col].iloc[0] if len(r) else np.nan
        trs.append(f"""<tr><th scope="row">{lead[-1]} day{'s' if lead != 'day1' else ''}</th>
<td class="num">{int(h.n)}</td><td class="num">{100 * h.pinball_skill_vs_clim:+.0f}%</td>
<td class="num">{h['brier_0.2']:.3f} <span class="vs">vs {c['brier_0.2']:.3f}</span></td>
<td class="num">{mm(pick('calibrated median', 'MAE'), 2)} <span class="vs">{signed(pick('calibrated median', 'bias'))}</span></td>
<td class="num">{mm(pick('calibrated expected total', 'MAE'), 2)} <span class="vs">{signed(pick('calibrated expected total', 'bias'))}</span></td>
<td class="num">{mm(pick('raw sr_ecmwf_ifs', 'MAE'), 2)} <span class="vs">{signed(pick('raw sr_ecmwf_ifs', 'bias'))}</span></td>
<td class="num">{mm(pick('raw equal_weight_blend', 'MAE'), 2)}</td></tr>""")
    period = T[T.gauge.str.startswith(gid)].test.iloc[0].replace("..", " to ")
    ho_blocks.append(f"""<h3 class="sub">{E(g['name'])} <span class="gid">held-out {E(period)}</span></h3>
<div class="scroll"><table class="ho"><thead><tr><th scope="col">Lead</th><th scope="col">Days</th>
<th scope="col">Skill vs seasonal climatology</th><th scope="col">Brier ≥ 0.2 mm <span class="vs">vs climatology</span></th>
<th scope="col">Median MAE <span class="vs">bias</span></th><th scope="col">Expected MAE <span class="vs">bias</span></th>
<th scope="col">Raw ECMWF IFS MAE <span class="vs">bias</span></th><th scope="col">Raw blend MAE</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table></div>""")
heldout_html = "\n".join(ho_blocks)

# ------------------------------------------------------------------------------------------------ forest plot
COMPS = [("MAE: calibrated median  minus  raw:sr_ecmwf_ifs", "Median vs raw ECMWF IFS"),
         ("MAE: calibrated expected total  minus  raw:sr_ecmwf_ifs", "Expected vs raw ECMWF IFS"),
         ("MAE: calibrated expected total  minus  raw:equal_weight_blend", "Expected vs raw blend")]
fr = []
for gid, g in gauges.items():
    p = pd.read_csv(ROOT / "reports" / g["slug"] / "paired_differences.csv")
    for lead in ("day1", "day2", "day3"):
        for key, lab in COMPS:
            r = p[(p.lead == lead) & (p.comparison == key)]
            if len(r):
                r = r.iloc[0]
                fr.append({"gauge": g["name"], "lead": lead, "label": lab, "mean": r.mean_diff, "lo": r.ci95_lo,
                           "hi": r.ci95_hi, "sig": bool(r.significant_at_95)})
F = pd.DataFrame(fr)
lo = np.floor(min(F.lo.min(), 0) / 0.5) * 0.5
hi = np.ceil(max(F.hi.max(), 0) / 0.5) * 0.5
W, LW, RW, ROWH, TOP = 760, 300, 40, 22, 34
PW = W - LW - RW


def X(v):
    return LW + (v - lo) / (hi - lo) * PW


svg = []
y = TOP
last = None
for _, r in F.iterrows():
    grp = f"{r.gauge} · {r.lead[-1]} day{'s' if r.lead != 'day1' else ''}"
    if grp != last:
        y += 10
        svg.append(f'<text x="0" y="{y + 4}" class="fp-grp">{E(grp)}</text>')
        y += ROWH - 6
        last = grp
    cls = ("better" if r["mean"] < 0 else "worse") if r.sig else "ns"
    svg.append(f'<text x="14" y="{y + 4}" class="fp-lab">{E(r.label)}</text>'
               f'<line x1="{X(r.lo):.1f}" x2="{X(r.hi):.1f}" y1="{y}" y2="{y}" class="fp-ci {cls}"/>'
               f'<circle cx="{X(r["mean"]):.1f}" cy="{y}" r="4" class="fp-pt {cls}"/>'
               f'<text x="{W - 2}" y="{y + 4}" class="fp-val" text-anchor="end">{r["mean"]:+.2f}</text>')
    y += ROWH
H = y + 8
ticks = np.arange(lo, hi + 1e-9, 0.5)
axis = "".join(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{TOP - 6}" y2="{H - 8}" class="{"fp-zero" if abs(t) < 1e-9 else "fp-grid"}"/>'
               f'<text x="{X(t):.1f}" y="{TOP - 12}" class="fp-tick" text-anchor="middle">{t:+.1f}</text>' for t in ticks)
forest_svg = (f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
              f'aria-label="Paired MAE differences with 95 percent intervals">{axis}{"".join(svg)}</svg>')
n_sig_better = int(((F.sig) & (F["mean"] < 0)).sum())
n_sig_worse = int(((F.sig) & (F["mean"] > 0)).sum())

# Brier vs seasonal climatology intervals
br = []
for gid, g in gauges.items():
    p = pd.read_csv(ROOT / "reports" / g["slug"] / "paired_differences.csv")
    for lead in ("day1", "day2", "day3"):
        r = p[(p.lead == lead) & p.comparison.str.startswith("Brier (event rain >= 0.2 mm): calibrated minus seasonal")]
        if len(r):
            r = r.iloc[0]
            br.append(f"<tr><th scope='row'>{E(g['name'])}</th><td>{lead[-1]}</td><td class='num'>{r.mean_diff:+.3f}</td>"
                      f"<td class='num'>{r.ci95_lo:+.3f} to {r.ci95_hi:+.3f}</td></tr>")
brier_html = "".join(br)

# ------------------------------------------------------------------------------------------------ review impact
md = (ROOT / "reports" / "acceptance_report.md").read_text(encoding="utf-8")
sec = md.split("## C. Which reviewer findings affected this run", 1)[1]
review_rows = []
for line in sec.splitlines():
    if line.startswith("| ") and not line.startswith("| finding") and not line.startswith("|---"):
        cells = [c.strip() for c in line.strip("|").split(" | ")]
        if len(cells) == 3 and not cells[0].startswith("location"):
            eff = cells[1].strip("*")
            kind = ("bad" if eff.startswith("TRIGGERED") else "warn" if eff.startswith("AFFECTED")
                    else "info" if "ASSURANCE" in eff else "ok")
            label = (eff.replace("TRIGGERED - ", "").replace("AFFECTED", "affected").replace("ASSURANCE", "assurance")
                     .replace("not triggered here", "did not occur here").replace("not triggered", "did not occur"))
            label = label[:1].upper() + label[1:]
            review_rows.append(f"<tr><td>{E(cells[0])}</td><td><span class='chip {kind}'>{label}</span></td>"
                               f"<td class='evid'>{E(cells[2])}</td></tr>")
review_html = "".join(review_rows)
sel_line = re.search(r"Compared programmatically.*?: (\d+) of (\d+) differ", md)
sel_diff, sel_n = (int(sel_line.group(1)), int(sel_line.group(2))) if sel_line else (None, None)

# ------------------------------------------------------------------------------------------------ integrity
tests = re.search(r"(\d+) passed", (ROOT / "logs" / "pytest_final.txt").read_text()).group(1)
failed = re.search(r"(\d+) failed", (ROOT / "logs" / "pytest_final.txt").read_text())
scan = (ROOT / "reports" / "enso_reference_scan_final.md").read_text(encoding="utf-8")
active_viol = sum(int(m.group(1)) for m in re.finditer(r"\| True \|.*\| (\d+) \|\s*$", scan, re.M))
n_active = len(re.findall(r"\| True \|", scan))
hist = pd.read_csv(ROOT / "reports" / "historical_timing_audit.csv")
wd = pd.read_csv(sorted(ROOT.glob("archive/WITHDRAWN_*/row_status.csv"))[0]).row_status.value_counts()
ids = re.findall(r"\| (PERTH METRO|GINGIN AERO|SWANBOURNE) \| day\d \| \d+ \| \[\d+\] \| (\d+) \| (True|False) \|", md)
ident_ok = sum(1 for _, n, f in ids if n == "1" and f == "True")
run_ids = {json.loads((ROOT / "data" / s / "models" / "manifest.json").read_text())["meta"]["run_id"] for _, s in LOCS}

checks = [
    (f"{tests}", "tests passed" + (f", {failed.group(1)} failed" if failed else ", 0 failed"), "ok" if not failed else "bad"),
    (f"{active_viol}", f"ENSO or climate-index references in {n_active} active model files", "ok" if active_viol == 0 else "bad"),
    (f"{ident_ok}/{len(ids)}", "gauge × lead comparisons scored on identical rows (one fingerprint each)",
     "ok" if ident_ok == len(ids) else "bad"),
    (f"{int(hist.rows_failing_as_of_rule.sum())}", f"of {int(hist.eligible_rows.sum()):,} training and held-out rows (all locations) fail the timing rule",
     "ok" if hist.rows_failing_as_of_rule.sum() == 0 else "bad"),
    (f"{sel_diff}/{sel_n}", "model choices changed by the fixes (selection is from pre-holdout folds only)",
     "ok" if sel_diff == 0 else "warn"),
]
checks_html = "".join(f"<li class='{k}'><span class='big'>{E(v)}</span><span>{E(t)}</span></li>" for v, t, k in checks)

# ------------------------------------------------------------------------------------------------ page
CSS = """
/* Layout: one reading column; forecast tables first, then evidence. Hairline tables, no card grid. */
:root{
  --bg:#f4f6f7; --surface:#ffffff; --ink:#16212a; --muted:#56646f; --rule:#d5dce1;
  --accent:#1c5a84; --accent-soft:#d9e7f0;
  --ok:#2c7349; --ok-soft:#dfefe5; --warn:#8f5d00; --warn-soft:#f6ead0; --bad:#a03628; --bad-soft:#f6dfdb;
  --info:#4b5a8c; --info-soft:#e3e6f2; --na:#6b7780; --na-soft:#e8ecef;
  --display:"Barlow Condensed","Arial Narrow",Arial,sans-serif;
  --body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0f161b; --surface:#152028; --ink:#e2e9ee; --muted:#93a2ad; --rule:#2a3843;
  --accent:#72b2de; --accent-soft:#1b3446;
  --ok:#6fc392; --ok-soft:#173428; --warn:#e2b158; --warn-soft:#3a2d12; --bad:#ec8a78; --bad-soft:#3d1d18;
  --info:#a9b4e3; --info-soft:#232a45; --na:#8d99a2; --na-soft:#202b33; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0f161b; --surface:#152028; --ink:#e2e9ee; --muted:#93a2ad; --rule:#2a3843;
  --accent:#72b2de; --accent-soft:#1b3446;
  --ok:#6fc392; --ok-soft:#173428; --warn:#e2b158; --warn-soft:#3a2d12; --bad:#ec8a78; --bad-soft:#3d1d18;
  --info:#a9b4e3; --info-soft:#232a45; --na:#8d99a2; --na-soft:#202b33; color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:15px/1.55 var(--body);margin:0}
.wrap{max-width:1080px;margin:0 auto;padding-inline:20px;padding-block:28px 64px;display:grid;grid-template-columns:minmax(0,1fr);gap:44px}
.wrap>*,.gauge>*,header>*{min-width:0}
header{display:grid;grid-template-columns:minmax(0,1fr);gap:10px;border-bottom:1px solid var(--rule);padding-bottom:22px}
.eyebrow{font:600 12px/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}
h1{font:600 clamp(34px,5vw,52px)/1.02 var(--display);margin:0;letter-spacing:.005em;text-wrap:balance}
h2{font:600 28px/1.1 var(--display);margin:0 0 6px;letter-spacing:.01em;text-wrap:balance}
h3{font:600 21px/1.2 var(--display);margin:0;letter-spacing:.01em}
h3.sub{margin-top:18px}
.lede{max-width:68ch;color:var(--muted);margin:0}
.meta{display:flex;flex-wrap:wrap;gap:6px 22px;font:13px/1.4 var(--mono);color:var(--muted);overflow-wrap:anywhere}
.meta>span{min-width:0}
.meta b{color:var(--ink);font-weight:500}
section>p,.note{max-width:72ch}
.note{color:var(--muted);font-size:14px}
.gauge{display:grid;grid-template-columns:minmax(0,1fr);gap:8px;margin-top:22px}
.gauge-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 16px}
.gid{font:500 12px/1 var(--mono);color:var(--muted);letter-spacing:.04em;margin-left:6px}
.serves{margin:0;color:var(--muted);font-size:14px}
.scroll{overflow-x:auto;min-width:0}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;background:var(--surface)}
th,td{padding:9px 10px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
thead th{font:600 11.5px/1.25 var(--body);text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
  border-bottom:1.5px solid var(--ink);vertical-align:bottom;background:var(--bg)}
td.num,th.num{text-align:right;font-family:var(--mono);font-size:13.5px;white-space:nowrap}
.vs{display:block;font:12px/1.2 var(--mono);color:var(--muted);text-transform:none;letter-spacing:0}
thead .vs{display:inline;font-family:var(--body)}
table.fc{min-width:860px}
.fc th[scope=row]{min-width:150px}
.day{display:block;font:600 18px/1.15 var(--display);letter-spacing:.01em}
.win{display:block;font:12px/1.3 var(--mono);color:var(--muted);margin-top:2px}
.lead{white-space:nowrap;color:var(--muted)}
.prob{display:flex;align-items:center;gap:8px;min-width:150px}
.bar{position:relative;flex:1;height:8px;border-radius:4px;background:var(--accent-soft);overflow:hidden;min-width:70px}
.bar span{position:absolute;inset:0 auto 0 0;width:calc(var(--p)*100%);background:var(--accent);border-radius:4px}
.prob .num{font:500 14px/1 var(--mono);min-width:3.2ch;text-align:right}
.model{display:flex;flex-wrap:wrap;align-items:center;gap:4px 8px}
.model code{font:12.5px/1.2 var(--mono)}
.route{font:12px/1 var(--mono);color:var(--muted)}
tr.na td,tr.na th{background:var(--na-soft)}
.na-text{color:var(--muted);font-size:14px}
.chip{display:inline-block;font:600 11px/1.2 var(--body);letter-spacing:.05em;text-transform:uppercase;
  padding:4px 7px;border-radius:3px}
.model .chip{white-space:nowrap}
.chip.ok{color:var(--ok);background:var(--ok-soft)}
.chip.warn{color:var(--warn);background:var(--warn-soft)}
.chip.bad{color:var(--bad);background:var(--bad-soft)}
.chip.info{color:var(--info);background:var(--info-soft)}
.chip[title]{cursor:help}
.checks{list-style:none;margin:14px 0 0;padding:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:1px;
  background:var(--rule);border:1px solid var(--rule)}
.checks li{background:var(--surface);padding:14px 14px 16px;display:grid;gap:6px;align-content:start}
.checks .big{font:600 30px/1 var(--display);letter-spacing:.01em}
.checks li.ok .big{color:var(--ok)} .checks li.bad .big{color:var(--bad)} .checks li.warn .big{color:var(--warn)}
.checks li span:last-child{font-size:13.5px;color:var(--muted);line-height:1.4}
table.ho{min-width:820px}
table.rv td{font-size:14px}
table.rv .evid{color:var(--muted);font-size:13px;max-width:60ch}
table.rv{min-width:760px}
table.br{max-width:560px}
.fp{background:var(--surface);border:1px solid var(--rule);padding:10px 12px 4px}
.fp svg{display:block;max-width:none}
.fp-grp{font:600 13px var(--body);fill:var(--ink)}
.fp-lab{font:12.5px var(--body);fill:var(--muted)}
.fp-val{font:12px var(--mono);fill:var(--muted)}
.fp-tick{font:11.5px var(--mono);fill:var(--muted)}
.fp-grid{stroke:var(--rule);stroke-width:1}
.fp-zero{stroke:var(--ink);stroke-width:1.2}
.fp-ci{stroke-width:2.2;stroke-linecap:round}
.fp-ci.better{stroke:var(--ok)} .fp-pt.better{fill:var(--ok)}
.fp-ci.worse{stroke:var(--bad)} .fp-pt.worse{fill:var(--bad)}
.fp-ci.ns{stroke:var(--na)} .fp-pt.ns{fill:var(--surface);stroke:var(--na);stroke-width:2}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--muted);margin:8px 0 0}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.legend .b{background:var(--ok)} .legend .w{background:var(--bad)} .legend .n{border:2px solid var(--na)}
ul.lim{margin:10px 0 0;padding-left:20px;display:grid;gap:8px;max-width:76ch}
footer{border-top:1px solid var(--rule);padding-top:16px;font-size:13px;color:var(--muted);display:grid;gap:6px}
code{font-family:var(--mono)}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (max-width:560px){.wrap{padding-inline:16px;gap:36px} h2{font-size:24px}}
"""

page = f"""<title>Perth Rainfall Calibration</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <span class="eyebrow">Gauge-calibrated rainfall · Perth, WA</span>
  <h1>Perth Rainfall Calibration</h1>
  <p class="lede">Archived weather-model rain forecasts (ECMWF, JMA, NOAA GFS via Open-Meteo) calibrated against Bureau of
  Meteorology rain gauges. Each rain day runs from 9am the previous day to 9am on the labelled day, as BoM reports it.</p>
  <div class="meta"><span>Forecast issued <b>{issued_local:%H:%M} AWST {issued_local:%a} {issued_local.day} {issued_local:%b %Y}</b> ({issued:%H:%M} UTC)</span>
  <span>Models <b>{E(', '.join(sorted(run_ids)))}</b></span><span>No El Niño input</span></div>
</header>

<section aria-labelledby="fc">
  <h2 id="fc">Forecast</h2>
  <p>Chance of measurable rain (≥ 0.2 mm, one gauge step), heavier thresholds, and the forecast total. A forecast is only
  issued when every model run it needs had been published before the issue time, with the same lead time the
  model was trained on. Rain day 2 Oct is not listed because its window started at 9am today.</p>
  {forecast_html}
  <p class="note">{unver_note} Hover a status chip for the model runs behind a row. Medians of 0.0 mm mean a dry day is
  more likely than not; the expected total averages over wet and dry outcomes.</p>
</section>

<section aria-labelledby="ck">
  <h2 id="ck">Checks on this build</h2>
  <ul class="checks">{checks_html}</ul>
  <p class="note">These show the pipeline is internally consistent. They say nothing about forecast accuracy; the held-out
  results below do.</p>
</section>

<section aria-labelledby="ho">
  <h2 id="ho">How it scored on unseen days</h2>
  <p>The last 20 % of dates (mid-March to 29 Sep 2026) were never used to fit or choose models. Every method below is
  scored on exactly the same station, rain-day windows and gauge readings. MAE and bias are in mm per rain day;
  bias is forecast minus observed. Skill compares the full forecast distribution (pinball loss) with a climatology
  for the same time of year.</p>
  {heldout_html}
  <p class="note">Probability scores exist only for the calibrated forecast and climatology. The raw models give amounts,
  not probabilities, so no probability comparison with them is possible. Perth and Ocean Reef use the same gauge
  and are counted once.</p>
</section>

<section aria-labelledby="pd">
  <h2 id="pd">Is the difference real?</h2>
  <p>Per-day error differences on the held-out days, with 95 % intervals from a block bootstrap (7-day blocks).
  Left of zero means the calibrated forecast had the smaller error. A hollow point means the interval includes zero:
  the ranking is descriptive, not significant.</p>
  <div class="scroll fp">{forest_svg}</div>
  <p class="legend"><span><i class="b"></i>calibrated significantly better ({n_sig_better})</span>
  <span><i class="w"></i>significantly worse ({n_sig_worse})</span>
  <span><i class="n"></i>not significant ({len(F) - n_sig_better - n_sig_worse})</span><span>MAE difference, mm per rain day</span></p>
  <p>Against climatology the calibrated rain probability is better at every gauge and lead (Brier ≥ 0.2 mm, calibrated
  minus seasonal climatology):</p>
  <div class="scroll"><table class="br"><thead><tr><th scope="col">Gauge</th><th scope="col">Lead (days)</th>
  <th scope="col" class="num">Difference</th><th scope="col" class="num">95 % interval</th></tr></thead>
  <tbody>{brier_html}</tbody></table></div>
  <p class="note">Bottom line: the calibrated forecast clearly beats climatology. Against the raw models and their blend,
  most rain-amount differences are within noise, so no general accuracy gain over the raw models is claimed.</p>
</section>

<section aria-labelledby="rv">
  <h2 id="rv">Code review findings and their effect</h2>
  <p>Three independent reviews found bugs. All confirmed ones are fixed with a regression test each. This table records
  whether each one actually changed anything in this run, based on the evidence rather than assumed.</p>
  <div class="scroll"><table class="rv"><thead><tr><th scope="col">Finding</th><th scope="col">Effect</th>
  <th scope="col">Evidence</th></tr></thead><tbody>{review_html}</tbody></table></div>
  <p class="note">Earlier forecasts made before the timing fix were withdrawn, not deleted: {int(wd.get('WITHDRAWN_TIMING_FAIL', 0))}
  rows failed the timing rule, {int(wd.get('TIMING_PASS_REISSUED', 0))} passed and were reissued, and {int(wd.get('NOT_SERVED', 0))}
  were never served. Training data passed the same rule, so the held-out scores stand.</p>
</section>

<section aria-labelledby="lm">
  <h2 id="lm">Limits</h2>
  <ul class="lim">
    <li>Forecasts describe the <b>gauge</b>, not the suburb. Clarkson's nearest suitable gauge is Gingin Aero,
    {GAUGE['clarkson']['km']:.1f} km away; Ocean Reef uses Perth Metro, {GAUGE['ocean_reef']['km']:.1f} km away.</li>
    <li>The held-out period is about 180 days per gauge (autumn to early spring 2026). Heavy-rain events in it are few, so
    scores for ≥ 5 and ≥ 10 mm are uncertain.</li>
    <li>Day-1 forecasts and anything beyond three days ahead are not issued: the archive only supports the leads the
    models were trained on, and the needed runs must already exist.</li>
    <li>El Niño and Indian Ocean Dipole indices are not used. Their source definition could not be verified, and no
    claim is made that they would improve these forecasts.</li>
    <li>Several comparisons are made per lead, so a single significant result should not be over-read.</li>
  </ul>
</section>

<footer>
  <span>Built from <code>reports/acceptance_report.md</code>, <code>reports/served_row_verification.csv</code> and each
  location's <code>calibration_metrics.csv</code> and <code>paired_differences.csv</code> by <code>scripts/build_html_report.py</code>.</span>
  <span>Forecast data: Open-Meteo.com (CC BY 4.0), from ECMWF, JMA and NOAA NCEP model output. Observations © Commonwealth of
  Australia, Bureau of Meteorology.</span>
</footer>
</div>
"""
out = ROOT / "reports" / "perth_rain_report.html"
out.write_text(page, encoding="utf-8")
print(out, len(page))
