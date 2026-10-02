import { pct, mm, dayLabel, stamp } from "./format.js";

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined && k !== false) n.append(k);
  return n;
}
const num = (x, d = 2) => (x === null || x === undefined ? "—" : Number(x).toFixed(d));
const sgn = (x, d = 2) => (x === null || x === undefined ? "—" : `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`);
const LEAD = { day1: "1 day ahead", day2: "2 days ahead", day3: "3 days ahead" };

/** Plain-language verdict for one paired comparison (never a win from missing data or tiny samples). */
export function verdictText(p) {
  if (p.verdict === "insufficient evidence") return "Insufficient evidence";
  if (p.verdict === "inconclusive") return "Cannot tell apart";
  if (p.verdict === "first better") return `${p.first} better`;
  if (p.verdict === "second better") return `${p.second} better`;
  return p.verdict;
}

export function verdictClass(p) {
  if (p.verdict === "first better") return p.first_key.startsWith("champion") ? "sig" : "info";
  if (p.verdict === "second better") return p.first_key.startsWith("champion") ? "worse" : "info";
  return "ns";
}

function amountTable(g) {
  const rows = [...g.amounts].filter((a) => a.MAE !== null && a.MAE !== undefined).sort((a, b) => a.MAE - b.MAE);
  return el("div", { class: "scroll" }, el("table", {},
    el("thead", {}, el("tr", {}, ...["Rank", "Forecast", "MAE", "RMSE", "Bias", "Wet-day MAE"].map((h, i) =>
      el("th", { class: i > 1 ? "n" : null, text: h })))),
    el("tbody", {}, rows.map((a, i) => el("tr", { class: a.key.startsWith("champion") ? "cal" : a.key === "blend" ? "raw" : "raw" },
      el("td", { class: "n", text: String(i + 1) }), el("td", { text: a.competitor }),
      el("td", { class: "n", text: num(a.MAE) }), el("td", { class: "n", text: num(a.RMSE) }),
      el("td", { class: "n", text: sgn(a.bias) }), el("td", { class: "n", text: num(a.wet_MAE) }))))));
}

function probTable(g) {
  const pr = g.probabilities.filter((p) => p.key === "champion" || p.key === "clim");
  const ths = [...new Set(pr.map((p) => p.threshold_mm))];
  return el("div", { class: "scroll" }, el("table", { class: "fit" },
    el("thead", {}, el("tr", {}, el("th", { text: "Rain/no-rain score (Brier, lower is better)" }),
      ...ths.map((t) => el("th", { class: "n", text: `≥ ${t} mm` })))),
    el("tbody", {},
      ...["champion", "clim"].map((k) => el("tr", { class: k === "champion" ? "cal" : "clim" },
        el("td", { text: k === "champion" ? "Our forecast" : "Season-aware climatology" }),
        ...ths.map((t) => { const r = pr.find((p) => p.key === k && p.threshold_mm === t); return el("td", { class: "n", text: num(r?.Brier, 3) }); }))),
      el("tr", { class: "clim" }, el("td", { text: "Event days in sample" }),
        ...ths.map((t) => el("td", { class: "n", text: String(pr.find((p) => p.threshold_mm === t)?.n_events ?? "—") }))),
      el("tr", { class: "clim" }, el("td", { text: "Raw models and challengers" }),
        el("td", { colspan: String(ths.length), text: "Unavailable: they give a rain amount, not a probability" })))));
}

function crpsLine(g) {
  const d = g.amounts.find((a) => a.key === "champion_distribution");
  const c = g.amounts.find((a) => a.key === "clim");
  if (!d) return null;
  const best = g.amounts.filter((a) => a.CRPS !== null && a.CRPS !== undefined && !["champion_distribution", "clim"].includes(a.key))
    .sort((a, b) => a.CRPS - b.CRPS)[0];
  return el("p", { class: "small muted", text: `Whole-distribution score (CRPS, mm, lower is better): our forecast ${num(d.CRPS)}, climatology ${num(c?.CRPS)}`
    + (best ? `, best raw model as a point forecast ${num(best.CRPS)} (${best.competitor}; for a single number the CRPS equals its MAE).` : ".") });
}

function reliabilityBlock(g) {
  const rel = g.reliability.filter((r) => r.threshold_mm === 0.2);
  if (!rel.length) return null;
  return el("details", { class: "more" }, el("summary", { text: "Reliability of our rain chances (≥ 0.2 mm)" }),
    el("div", { class: "body" },
      el("div", { class: "scroll" }, el("table", { class: "fit" },
        el("thead", {}, el("tr", {}, ...["Forecast chance", "Days", "Average forecast", "Rain observed"].map((h, i) => el("th", { class: i ? "n" : null, text: h })))),
        el("tbody", {}, rel.map((r) => el("tr", { class: r.enough_data ? "" : "clim" },
          el("td", { text: r.bin.replace("-", "–") }), el("td", { class: "n", text: String(r.n) }),
          el("td", { class: "n", text: r.enough_data ? pct(r.mean_forecast) : "—" }),
          el("td", { class: "n", text: r.enough_data ? pct(r.observed_frequency) : "too few days" })))))),
      g.interval && g.interval.n_days ? el("p", { class: "note", text: `Range check: the observed total was above our 90th percentile on ${pct(g.interval.above_90th)} of days (should be about 10%).` }) : null));
}

/** Split paired comparisons by the score they use: rain-amount error vs whole-distribution score (CRPS). */
export function splitPaired(paired) {
  const crps = paired.filter((p) => p.score === "CRPS");
  const amount = paired.filter((p) => p.score !== "CRPS");
  return { amount, crps };
}

function pairedTable(rows, header) {
  return el("div", { class: "scroll" }, el("table", {},
    el("thead", {}, el("tr", {}, ...["Comparison", header, "95% interval", "Result"].map((h, i) =>
      el("th", { class: i === 1 || i === 2 ? "n" : null, text: h })))),
    el("tbody", {}, rows.map((p) => el("tr", {},
      el("td", { text: `${p.first} vs ${p.second}` }), el("td", { class: "n", text: sgn(p.mean_abs_error_diff, 3) }),
      el("td", { class: "n", text: `${sgn(p.ci95[0], 3)} to ${sgn(p.ci95[1], 3)}` }),
      el("td", { class: verdictClass(p), text: verdictText(p) }))))));
}

function pairedBlock(g) {
  if (!g.paired.length) return el("p", { class: "small muted", text: "No paired comparisons (too few common days)." });
  const { amount, crps } = splitPaired(g.paired);
  return el("div", {},
    el("h4", { text: "Rain-amount error (absolute error, mm per rain day)" }),
    amount.length ? pairedTable(amount, "Error difference (mm)") : el("p", { class: "small muted", text: "None." }),
    crps.length ? el("h4", { text: "Whole-distribution score (CRPS): our full forecast vs single-number forecasts" }) : null,
    crps.length ? el("p", { class: "note", text: "This compares our full range of possible outcomes with a single number. "
      + "A forecast that expresses uncertainty usually scores better than a single number on this measure, so a win "
      + "here does not mean our rain amounts are more accurate; see the rain-amount table above for that." }) : null,
    crps.length ? pairedTable(crps, "CRPS difference (mm)") : null);
}

/** Availability evidence for one competitor: observed (explicit run / model metadata) or the 6-hour estimate. */
export function availabilityText(t) {
  if (!t || !t.availability_basis) return "—";
  return t.availability_basis === "observed" ? "Observed" : "Estimated (+6 h rule)";
}

export function availabilityClass(t) {
  return t && t.availability_basis === "observed" ? "sig" : "ns";
}

function timingTable(g) {
  if (!g.timing || !g.timing.length) return null;
  return el("div", {},
    el("h3", { text: "Information age at the cutoff" }),
    el("p", { class: "small muted", text: "Hours from the latest model run each forecast used to the start of the rain-day "
      + "window. Competitors share the lead group and cutoff, but not the age of their information. Availability of "
      + "archived runs is estimated (run start + 6 h): the archive does not record when each run was published." }),
    el("div", { class: "scroll" }, el("table", { class: "fit" },
      el("thead", {}, el("tr", {}, ...["Forecast", "Median age", "Range", "Availability"].map((h, i) =>
        el("th", { class: i === 1 || i === 2 ? "n" : null, text: h })))),
      el("tbody", {}, g.timing.map((t) => el("tr", {},
        el("td", { text: t.competitor }), el("td", { class: "n", text: `${t.median_age_h} h` }),
        el("td", { class: "n", text: `${t.min_age_h}–${t.max_age_h} h` }),
        el("td", { class: "ns", text: "Estimated (+6 h rule)" })))))));
}

function coverageLine(g) {
  return el("p", { class: "small muted", text: "Coverage of eligible windows: " +
    g.coverage.map((c) => `${c.competitor} ${c.with_forecast}/${c.eligible_windows}`).join(" · ") +
    ". Leaderboards use only windows where every forecast exists." });
}

function trackView(rep, emptyText) {
  if (!rep || !rep.groups || !rep.groups.length) return el("p", { class: "banner stale", text: emptyText });
  return el("div", {},
    el("p", { class: "small muted", text: `${rep.multiple_comparisons} ${rep.verdict_rule}` }),
    ...rep.groups.map((g) => el("section", { class: "section" },
      el("h2", { text: `${g.gauge_name} gauge · ${LEAD[g.lead]}` }),
      el("div", { class: "stats" },
        el("span", {}, "Common days ", el("b", { text: String(g.common_days) })),
        el("span", {}, "Dates ", el("b", { text: g.first_date ? `${g.first_date} to ${g.last_date}` : "—" })),
        el("span", {}, "Gauge distance ", el("b", { text: g.distances.map((d) => `${d.location} ${d.distance_km} km`).join(", ") })),
        el("span", {}, "Our model ", el("b", { text: `${g.champion_model || "—"} (${g.champion_run_id || "—"})` }))),
      g.common_days < 60 ? el("p", { class: "banner stale", text: "Insufficient evidence: fewer than 60 common days. Rankings here are descriptive only." }) : null,
      coverageLine(g), timingTable(g),
      el("h3", { text: "Rain amount leaderboard (mm per rain day)" }), amountTable(g),
      el("h3", { text: "Rain probability scores" }), probTable(g), crpsLine(g), reliabilityBlock(g),
      el("h3", { text: "Paired comparisons" }), pairedBlock(g))));
}

function liveView(live) {
  if (!live || !live.rows || !live.rows.length) {
    return el("p", { class: "banner stale", text: "No forecasts have been collected for the challenge yet." });
  }
  const byLoc = {};
  for (const r of live.rows) (byLoc[r.location] ||= []).push(r);
  return el("div", {},
    el("p", { class: "small muted", text: `Latest issue ${stamp(live.issued_at_utc)} (scheduled ${live.schedule_slot_utc}). ${live.ledger_records} forecasts in the ledger.` }),
    ...Object.entries(byLoc).map(([loc, rows]) => el("section", { class: "section" },
      el("h2", { text: `${loc} · ${rows[0].gauge_name} gauge (${rows[0].gauge_distance_km} km)` }),
      ...rows.map((r) => el("div", { class: "lead-block" },
        el("h3", { text: `${dayLabel(r.label_date)} · ${LEAD[r.lead]}` }),
        el("p", { class: "small muted", text: r.observed_mm === null || r.observed_mm === undefined ? "Gauge reading: not yet available" : `Gauge reading: ${mm(r.observed_mm)} mm` }),
        el("div", { class: "scroll" }, el("table", { class: "fit" },
          el("thead", {}, el("tr", {}, ...["Forecast", "Amount (mm)", "Chance ≥ 0.2 mm", "Model run (UTC)", "Information age", "Availability", "Status"].map((h, i) => el("th", { class: i === 1 || i === 2 || i === 4 ? "n" : null, text: h })))),
          el("tbody", {}, r.competitors.map((c) => el("tr", { class: c.key === "champion" ? "cal" : "raw" },
            el("td", { text: c.competitor + (c.key === "champion" ? " (expected total)" : "") }),
            el("td", { class: "n", text: c.status === "ok" ? mm(c.amount_mm) : "—" }),
            el("td", { class: "n", text: c.key === "champion" && c.status === "ok" ? pct(c.probabilities?.["ge_0.2"]) : c.status === "ok" ? "n/a (amount only)" : "—" }),
            el("td", { text: c.timing?.run_init_utc ? c.timing.run_init_utc.replace("T", " ") : "—" }),
            el("td", { class: "n", text: c.timing?.info_age_h != null ? `${c.timing.info_age_h} h` : "—" }),
            el("td", { class: availabilityClass(c.timing), text: availabilityText(c.timing) }),
            el("td", { class: c.status === "ok" ? "" : "ns", text: c.status === "ok" ? "issued" : `missing: ${c.missing_reason}` })))))))))));
}

function sourcesView(src) {
  return el("div", { class: "scroll" }, el("table", {},
    el("thead", {}, el("tr", {}, ...["Model", "Role", "Status", "Archive (previous runs)", "Single runs from", "Native step", "Notes"].map((h) => el("th", { text: h })))),
    el("tbody", {}, src.map((s) => el("tr", {},
      el("td", { text: s.model_id }), el("td", { text: s.role }), el("td", { class: s.status === "included" ? "sig" : "ns", text: s.status }),
      el("td", { text: s.previous_runs_from ? `${s.previous_runs_from} to ${s.previous_runs_to}` : "none" }),
      el("td", { text: s.single_runs_from || "none" }), el("td", { text: s.native_step_h ? `${s.native_step_h} h` : "—" }),
      el("td", { class: "small", text: s.note }))))));
}

async function main() {
  const root = document.getElementById("challenge");
  let d;
  try {
    const r = await fetch("data/v1/challenge.json", { cache: "no-cache" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    d = await r.json();
  } catch (e) {
    root.replaceChildren(el("p", { class: "banner error", text: `Challenge data could not be loaded (${e.message}).` }));
    return;
  }
  const views = {
    historical: () => trackView(d.historical, "No historical results exported."),
    prospective: () => trackView(d.prospective, "No completed prospective windows yet: forecasts are scored only after the gauge reading for their 9am–9am window is published."),
    live: () => liveView(d.live),
    sources: () => sourcesView(d.sources),
  };
  const body = el("div", {});
  const tabs = el("div", { class: "picker", role: "group", "aria-label": "Challenge views" });
  const show = (k) => {
    for (const b of tabs.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.k === k));
    body.replaceChildren(views[k]());
    history.replaceState(null, "", `#${k}`);
  };
  for (const [k, label] of [["live", "Latest forecasts"], ["historical", "Historical results"], ["prospective", "Live results"], ["sources", "Sources"]]) {
    const b = el("button", { type: "button", "data-k": k, "aria-pressed": "false", text: label });
    b.addEventListener("click", () => show(k));
    tabs.append(b);
  }
  root.replaceChildren(
    el("p", { class: "small muted", text: `Data exported ${stamp(d.exported_utc)}.` }),
    el("p", { class: "notice", text: d.rules.fair_contest }),
    el("p", { class: "small muted", text: `${d.rules.probabilities} ${d.rules.satellite}` }),
    tabs, body);
  const start = location.hash.slice(1);
  show(views[start] ? start : "historical");
}

if (typeof document !== "undefined") main();
