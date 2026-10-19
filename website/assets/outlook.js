// 7-day forecast in the Bureau of Meteorology's plain style. Reads data/v1/outlook.json (schema 2).
import { stamp, freshness } from "./format.js";

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined && k !== false) n.append(k);
  return n;
}

/** Label for one day's rain information: our calibrated gauge forecast, or raw model agreement. */
export function sourceLabel(day) {
  return day.calibrated ? "Our calibrated forecast" : "Raw model guidance";
}

/** The rain line for one day. A percentage only ever comes from our calibrated forecast. */
export function rainLines(day) {
  if (day.calibrated) {
    return [["Chance of any rain", day.chance_of_any_rain], ["Possible rainfall", day.possible_rainfall]];
  }
  const out = [["Rain", day.model_agreement]];
  if (day.possible_rainfall) out.push(["Model range", day.possible_rainfall.replace(/^the models range from /, "")]);
  return out;
}

export function dayName(iso) {
  const [y, m, d] = iso.split("-").map(Number);
  const dt = new Date(Date.UTC(y, m - 1, d, 4));
  return new Intl.DateTimeFormat("en-AU", { timeZone: "Australia/Perth", weekday: "long", day: "numeric", month: "long" }).format(dt);
}

function dayRow(day) {
  return el("li", { class: `bom-day${day.calibrated ? " calibrated" : ""}` },
    el("div", { class: "bom-head" },
      el("h3", { class: "bom-date", text: dayName(day.date) }),
      el("p", { class: "bom-temps" },
        el("span", { class: "min" }, "Min ", el("b", { text: day.min_c ?? "—" })),
        el("span", { class: "max" }, "Max ", el("b", { text: day.max_c ?? "—" })))),
    el("p", { class: "bom-precis", text: day.forecast }),
    el("dl", { class: "bom-rain" }, ...rainLines(day).map(([k, v]) => el("div", {}, el("dt", { text: k }), el("dd", { text: v })))),
    el("p", { class: "bom-source" }, el("span", { class: `chip ${day.calibrated ? "ok" : "na"}`, text: sourceLabel(day) }),
      day.calibrated && /backup/.test(day.rain_source) ? el("span", { class: "small muted", text: " Backup model: its accuracy has not been measured yet." }) : null));
}

async function main() {
  const root = document.getElementById("outlook");
  let d;
  try {
    const r = await fetch("data/v1/outlook.json", { cache: "no-cache" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    d = await r.json();
  } catch (e) {
    root.replaceChildren(el("p", { class: "banner error", text: `The forecast could not be loaded (${e.message}).` }));
    return;
  }
  const fr = freshness(d.generated_utc);
  root.replaceChildren(
    el("section", { class: "loc-head" },
      el("h1", { text: "Perth 7-day forecast" }),
      el("p", { class: "stand", text: `${d.headline}.` }),
      el("div", { class: "updates" },
        el("span", {}, "Issued ", el("b", { text: stamp(d.generated_utc) })),
        d.calibrated_forecast_made_utc ? el("span", {}, "Calibrated rain chances from ", el("b", { text: stamp(d.calibrated_forecast_made_utc) })) : null),
      fr.stale ? el("p", { class: "banner stale", text: `This forecast is ${Math.round(fr.ageHours)} hours old. It is a snapshot and does not update by itself.` }) : null),
    el("section", { "aria-label": "Forecast by day" }, el("ol", { class: "bom-week" }, d.days.map(dayRow))),
    el("section", { class: "section", "aria-labelledby": "found" },
      el("h2", { id: "found", text: "What we found" }),
      el("ul", { class: "plain" }, d.findings.map((f) => el("li", { text: f })))),
    el("section", { "aria-labelledby": "read" },
      el("h2", { id: "read", text: "How to read this forecast" }),
      el("ul", { class: "plain" }, d.how_to_read.map((f) => el("li", { text: f }))),
      el("p", { class: "small muted", text: d.sources })));
}

if (typeof document !== "undefined") main();
