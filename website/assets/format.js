// Pure formatting and state rules (no DOM). Unit-tested by website/tests/format.test.mjs.

export const TZ = "Australia/Perth";
export const STALE_AFTER_HOURS = 18;

/** Probability (0..1) -> percentage text. Never shows 0 % or 100 %: tiny chances are "<1%", not impossible. */
export function pct(p) {
  if (p === null || p === undefined || Number.isNaN(p)) return "—";
  if (p < 0.01) return "<1%";
  if (p > 0.99) return ">99%";
  return `${Math.round(p * 100)}%`;
}

/** Rainfall in mm. Exact 0 stays "0"; small non-zero amounts never round to 0. */
export function mm(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  if (x === 0) return "0";
  if (x < 0.1) return "<0.1";
  if (x < 10) return (Math.round(x * 10) / 10).toFixed(1);
  return String(Math.round(x));
}

export function rangeText(lo, hi) {
  if (lo === null || hi === null || lo === undefined || hi === undefined) return "—";
  return `${mm(lo)}–${mm(hi)} mm`;
}

const fmt = (opts) => new Intl.DateTimeFormat("en-AU", { timeZone: TZ, ...opts });
const parts = (d, opts) => Object.fromEntries(fmt(opts).formatToParts(d).map((x) => [x.type, x.value]));

/** "Sat 3 Oct" for a label date string "2026-10-03" (date only: no timezone shift). */
export function dayLabel(labelDate) {
  const [y, m, d] = labelDate.split("-").map(Number);
  const dt = new Date(Date.UTC(y, m - 1, d, 4)); // 12:00 AWST on that date
  const p = parts(dt, { weekday: "short", day: "numeric", month: "short" });
  return `${p.weekday} ${p.day} ${p.month}`;
}

/** "9am Fri 2 Oct" in Perth time. */
export function localMoment(iso) {
  const d = new Date(iso);
  const p = parts(d, { weekday: "short", day: "numeric", month: "short", hour: "numeric", minute: "2-digit", hour12: true });
  const time = p.minute === "00" ? `${p.hour}${p.dayPeriod.toLowerCase()}` : `${p.hour}:${p.minute}${p.dayPeriod.toLowerCase()}`;
  return `${time} ${p.weekday} ${p.day} ${p.month}`;
}

export function windowText(startIso, endIso) {
  return `${localMoment(startIso)} → ${localMoment(endIso)} (AWST)`;
}

/** "10:30am Thu 1 Oct 2026 AWST" */
export function stamp(iso) {
  if (!iso) return "—";
  const p = parts(new Date(iso), { weekday: "short", day: "numeric", month: "short", year: "numeric", hour: "numeric", minute: "2-digit", hour12: true });
  return `${p.hour}:${p.minute}${p.dayPeriod.toLowerCase()} ${p.weekday} ${p.day} ${p.month} ${p.year} AWST`;
}

export function hoursBetween(aIso, bIso) {
  return (new Date(bIso) - new Date(aIso)) / 3.6e6;
}

export function leadText(f) {
  if (f.lead_hours_to_window_start === null || f.lead_hours_to_window_start === undefined) return "lead unknown";
  return `${Math.round(f.lead_hours_to_window_start)}–${Math.round(f.lead_hours_to_window_end)} h after model run`;
}

/** Whole-forecast freshness, judged against the viewer's clock (the site cannot refresh the data itself). */
export function freshness(forecastMadeIso, nowMs = Date.now()) {
  const age = (nowMs - new Date(forecastMadeIso).getTime()) / 3.6e6;
  return { ageHours: age, stale: age > STALE_AFTER_HOURS };
}

/** State of one rain-day window relative to now. */
export function windowState(day, nowMs = Date.now()) {
  const s = new Date(day.window_start_utc).getTime();
  const e = new Date(day.window_end_utc).getTime();
  if (nowMs >= e) return "ended";
  if (nowMs >= s) return "started";
  return "upcoming";
}

/** The forecast to show first: the shortest lead (most recent model information). */
export function primaryForecast(day) {
  const fs = [...(day.forecasts || [])].filter((f) => f.status === "ok");
  fs.sort((a, b) => (a.lead_hours_to_window_start ?? 1e9) - (b.lead_hours_to_window_start ?? 1e9));
  return fs[0] || null;
}

/** Plain-language note when the median is 0 mm (never "no rain"). */
export function medianNote(f) {
  const med = f.amounts_mm.median;
  const p = f.probabilities.ge_0_2mm;
  if (med === 0 && p !== null) {
    return `A median of 0 mm means a dry rain day is more likely than a wet one. It does not mean rain is impossible: the chance of at least 0.2 mm is ${pct(p)}.`;
  }
  return "";
}

export function verificationChip(v) {
  if (v === "verified") return { cls: "ok", text: "Verified inputs" };
  if (v === "unverified") return { cls: "warn", text: "Unverified inputs" };
  if (v === "failed") return { cls: "stale", text: "Verification failed" };
  return { cls: "na", text: "Not checked" };
}

export function routeText(f) {
  if (f.route === "primary") return "Selected model";
  const m = /^fallback#(\d+)$/.exec(f.route || "");
  return m ? `Fallback model (choice ${Number(m[1]) + 1})` : f.route || "—";
}

export const LEAD_NAME = { day1: "1-day lead", day2: "2-day lead", day3: "3-day lead" };

const RAW_NAMES = { sr_ecmwf_ifs: "ECMWF IFS", pr_ecmwf_ifs025: "ECMWF IFS 0.25°", pr_jma_gsm: "JMA GSM",
  pr_ncep_gfs_global: "NOAA GFS", equal_weight_blend: "equal-weight blend" };
export function comparisonLabel(c) {
  let t = c.replace(/raw:(\w+)/g, (_, k) => `raw ${RAW_NAMES[k] || k}`);
  t = t.replace(/^MAE: calibrated (median|expected total) minus /, (_, w) => `Error of calibrated ${w} vs `);
  t = t.replace(/^Brier \(event rain >= 0\.2 mm\): calibrated minus (seasonal|flat) climatology$/,
    (_, w) => `Rain/no-rain score vs ${w === "flat" ? "season-blind" : "seasonal"} climatology`);
  t = t.replace(/^pinball loss: calibrated minus seasonal climatology$/, "Whole-distribution score vs seasonal climatology");
  return t;
}
