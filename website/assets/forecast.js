import {
  pct, mm, rangeText, dayLabel, windowText, stamp, leadText, freshness, windowState,
  primaryForecast, medianNote, verificationChip, routeText, routeChip, routeNote, VERIFICATION_NOTE, LEAD_NAME,
} from "./format.js";

const DATA = "data/v1/";

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined && k !== false) n.append(k);
  return n;
}
const chip = (cls, text) => el("span", { class: `chip ${cls}`, text });
const kv = (pairs) => el("dl", { class: "kv" }, pairs.filter(([, v]) => v !== null && v !== undefined && v !== "")
  .flatMap(([k, v]) => [el("dt", { text: k }), el("dd", {}, v)]));

function forecastBody(f) {
  const v = verificationChip(f.verification);
  const p = f.probabilities;
  const a = f.amounts_mm;
  const runs = f.input_runs.map((r) => `${r.model}: ${r.run_init_utc.replace("T", " ")} UTC (${r.run_identification})`);
  return [
    el("div", { class: "chips" },
      chip(v.cls, v.text),
      chip("info", LEAD_NAME[f.lead_group] || f.lead_group),
      routeChip(f) ? chip(routeChip(f).cls, routeChip(f).text) : null),
    el("div", { class: "headline" },
      el("span", { class: "big", text: pct(p.ge_0_2mm) }),
      el("span", { class: "big-label", text: "chance of at least 0.2 mm (measurable rain)" })),
    el("dl", { class: "amounts" },
      el("div", {}, el("dt", { text: "Expected total" }), el("dd", { text: `${mm(a.expected)} mm` })),
      el("div", {}, el("dt", { text: "Median" }), el("dd", { text: `${mm(a.median)} mm` })),
      el("div", {}, el("dt", { text: "Likely range (10–90%)" }), el("dd", { text: rangeText(a.p10, a.p90) }))),
    el("dl", { class: "thresholds", "aria-label": "Chance of reaching each rainfall amount" },
      ...[["≥ 0.2 mm", p.ge_0_2mm], ["≥ 1 mm", p.ge_1mm], ["≥ 5 mm", p.ge_5mm], ["≥ 10 mm", p.ge_10mm]]
        .map(([k, x]) => el("div", {}, el("dt", { text: k }), el("dd", { text: pct(x) })))),
    medianNote(f) ? el("p", { class: "note", text: medianNote(f) }) : null,
    routeNote(f) ? el("p", { class: "note", text: routeNote(f) }) : null,
    el("details", { class: "more" },
      el("summary", { text: "Raw model totals (uncalibrated)" }),
      el("div", { class: "body" },
        el("span", { class: "raw-label", text: "Comparison only, not a calibrated forecast" }),
        f.raw_model_totals_uncalibrated.length
          ? kv(f.raw_model_totals_uncalibrated.map((r) => [r.model, `${mm(r.total_mm)} mm (raw model output for this window)`]))
          : el("p", { text: "No raw totals were recorded for this forecast." }),
        el("p", { class: "note", text: "These are the weather models' own rainfall totals for the same 9am–9am window, before calibration against the gauge. Only the models used by this forecast are listed." }))),
    el("details", { class: "more" },
      el("summary", { text: "Provenance and model routing" }),
      el("div", { class: "body" },
        kv([
          ["Model runs used", el("span", {}, ...runs.map((r) => el("div", { text: r })))],
          ["Lead time", leadText(f) + " (from the latest model run used to the window start and end)"],
          ["Calibration", `${f.calibrated_model_label || f.calibrated_model} · ${routeText(f)}`],
          ["Models available", f.models_available.join(", ") || "—"],
          ["Inputs not yet usable", f.inputs_blocked_by_timing.join(", ") || "none"],
          ["Input check status", VERIFICATION_NOTE[f.verification] || "Not checked."],
          ["Input check", el("span", {}, ...f.verification_detail.map((x) => el("div", { text: `${x.input}: ${x.status} — ${x.detail}` })))],
          ["Inputs fetched", f.inputs_retrieved_utc ? stamp(f.inputs_retrieved_utc) : "—"],
          ["Model trained on", f.model_training_days ? `${f.model_training_days} days up to ${f.model_trained_to}` : "—"],
          ["Model release", el("code", { text: f.model_run_id })],
          ["Model file hash", el("code", { text: (f.artifact_sha256 || "").slice(0, 16) + "…" })],
        ]))),
  ];
}

function dayCard(day, nowMs) {
  const state = windowState(day, nowMs);
  const head = el("div", { class: "card-head" },
    el("h3", { class: "day", text: dayLabel(day.label_date) }),
    el("p", { class: "window", text: `Rain from ${windowText(day.window_start_local, day.window_end_local)}` }));
  if (day.availability !== "available") {
    const withdrawn = (day.withdrawn || []).length > 0;
    return el("article", { class: "card unavailable" }, head,
      el("div", { class: "chips" }, chip("na", withdrawn ? "Withdrawn" : "Unavailable")),
      el("p", { class: "unavail-big", text: withdrawn ? "Forecast withdrawn" : "Not available yet" }),
      el("p", { class: "small muted", text: day.unavailable_reason }),
      el("p", { class: "note", text: "Unavailable is not a forecast of zero rain. Check again after the next data export." }));
  }
  const f = primaryForecast(day);
  const others = day.forecasts.filter((x) => x !== f);
  return el("article", { class: "card" }, head,
    state !== "upcoming" ? el("div", { class: "chips" }, chip("stale", state === "started" ? "Window in progress" : "Window ended")) : null,
    ...forecastBody(f),
    others.length ? el("details", { class: "more" },
      el("summary", { text: `Earlier forecast for this day (${others.map((o) => LEAD_NAME[o.lead_group]).join(", ")})` }),
      el("div", { class: "body" }, ...others.map((o) => el("div", { class: "card", style: "padding:12px" }, ...forecastBody(o))))) : null);
}

function renderLocation(root, fc, loc, nowMs) {
  const fr = freshness(loc.forecast_made_utc, nowMs);
  const shared = fc.locations.filter((l) => l.gauge.id === loc.gauge.id && l.id !== loc.id).map((l) => l.name);
  const days = loc.days.filter((d) => windowState(d, nowMs) !== "ended");
  root.replaceChildren(
    el("section", { class: "loc-head", "aria-labelledby": "loc-title" },
      el("h1", { id: "loc-title", text: `${loc.name} rainfall` }),
      el("p", { class: "gauge-line" }, `Forecasts are for the Bureau of Meteorology gauge `,
        el("b", { text: `${loc.gauge.name} (${loc.gauge.id})` }), `, ${loc.gauge.distance_km} km from ${loc.name}.`,
        shared.length ? ` The same gauge is used for ${shared.join(", ")}.` : ""),
      el("p", { class: "notice", text: "Gauge forecast, not a property forecast. It predicts what this rain gauge will record. Rain at your address can differ, especially for showers and when the gauge is far away." }),
      el("div", { class: "updates" },
        el("span", {}, "Forecast made ", el("b", { text: stamp(loc.forecast_made_utc) })),
        el("span", {}, "Latest gauge reading ", el("b", { text: dayLabel(loc.latest_observation.label_date) })),
        el("span", {}, "Data exported ", el("b", { text: stamp(fc.exported_utc) }))),
      fr.stale ? el("p", { class: "banner stale", text: `This forecast is ${Math.round(fr.ageHours)} hours old. The site shows exported data only and cannot refresh it; a newer forecast needs a new export from the pipeline.` }) : null),
    el("section", { class: "cards", "aria-label": "Rain-day forecasts" },
      days.length ? days.map((d) => dayCard(d, nowMs)) : el("p", { class: "banner stale", text: "No forecast windows in this export are still upcoming." })));
}

async function main() {
  const root = document.getElementById("forecast");
  const picker = document.getElementById("picker");
  let fc;
  try {
    const r = await fetch(DATA + "forecast.json", { cache: "no-cache" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    fc = await r.json();
  } catch (e) {
    root.replaceChildren(el("p", { class: "banner error", text: `The forecast data could not be loaded (${e.message}). No forecast is shown rather than an empty one.` }));
    return;
  }
  const ids = fc.locations.map((l) => l.id);
  const pick = (id) => {
    const loc = fc.locations.find((l) => l.id === id) || fc.locations[0];
    for (const b of picker.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.id === loc.id));
    renderLocation(root, fc, loc, Date.now());
    if (location.hash !== `#${loc.id}`) history.replaceState(null, "", `#${loc.id}`);
    try { localStorage.setItem("perthrain.location", loc.id); } catch { /* storage unavailable */ }
  };
  picker.replaceChildren(...fc.locations.map((l) => {
    const b = el("button", { type: "button", "data-id": l.id, "aria-pressed": "false", text: l.name });
    b.addEventListener("click", () => pick(l.id));
    return b;
  }));
  let start = location.hash.slice(1);
  if (!ids.includes(start)) { try { start = localStorage.getItem("perthrain.location"); } catch { start = null; } }
  pick(ids.includes(start) ? start : ids[0]);
  window.addEventListener("hashchange", () => { const h = location.hash.slice(1); if (ids.includes(h)) pick(h); });
}

main();
