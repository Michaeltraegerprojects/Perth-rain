import { pct, dayLabel, stamp, comparisonLabel } from "./format.js";

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined && k !== false) n.append(k);
  return n;
}
const f2 = (x) => (x === null || x === undefined ? "—" : x.toFixed(2));
const sgn = (x, d = 2) => (x === null || x === undefined ? "—" : `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`);
const LEAD = { day1: "1 day ahead", day2: "2 days ahead", day3: "3 days ahead" };

function leadBlock(l) {
  const rows = l.point_accuracy_mm.map((r) =>
    el("tr", { class: r.kind === "calibrated" ? "cal" : r.kind === "raw" ? "raw" : "clim" },
      el("td", { text: r.method }), el("td", { class: "n", text: f2(r.mae) }),
      el("td", { class: "n", text: sgn(r.bias) }), el("td", { class: "n", text: f2(r.rmse) })));
  const pd = l.paired_differences.filter((d) => d.comparison.startsWith("MAE"));
  const pb = l.paired_differences.filter((d) => !d.comparison.startsWith("MAE"));
  const diffRow = (d, unit) => el("tr", {},
    el("td", { text: comparisonLabel(d.comparison) }),
    el("td", { class: "n", text: sgn(d.mean, unit === "Brier" ? 3 : 2) }),
    el("td", { class: "n", text: `${sgn(d.ci95[0], unit === "Brier" ? 3 : 2)} to ${sgn(d.ci95[1], unit === "Brier" ? 3 : 2)}` }),
    el("td", { class: d.significant ? (d.mean < 0 ? "sig" : "worse") : "ns",
      text: d.significant ? (d.mean < 0 ? "Calibrated better" : "Calibrated worse") : "Not significant" }));
  return el("div", { class: "lead-block" },
    el("h3", { text: LEAD[l.lead_group] }),
    el("div", { class: "stats" },
      el("span", {}, "Test days ", el("b", { text: String(l.n_days) })),
      el("span", {}, "Test period ", el("b", { text: `${dayLabel(l.test_first)} ${l.test_first.slice(0, 4)} – ${dayLabel(l.test_last)} ${l.test_last.slice(0, 4)}` })),
      el("span", {}, "Model trained up to ", el("b", { text: l.trained_to })),
      el("span", {}, "Selected model ", el("b", { text: l.selected_model_label || l.selected_model })),
      el("span", {}, "Rain days in test ", el("b", { text: pct(l.observed_rain_day_frequency) }))),
    el("div", { class: "stats" },
      el("span", {}, "Brier score, rain ≥ 0.2 mm (lower is better): calibrated ",
        el("b", { text: l.brier_ge_0_2mm.calibrated.toFixed(3) }), " vs seasonal climatology ",
        el("b", { text: l.brier_ge_0_2mm.seasonal_climatology.toFixed(3) })),
      el("span", {}, "Distribution skill vs seasonal climatology ",
        el("b", { text: `${Math.round(l.pinball_skill_vs_seasonal_climatology * 100)}%` }))),
    el("div", { class: "scroll" }, el("table", { class: "fit" },
      el("thead", {}, el("tr", {}, el("th", { text: "Rainfall amount (mm per rain day)" }), el("th", { class: "n", text: "MAE" }),
        el("th", { class: "n", text: "Bias" }), el("th", { class: "n", text: "RMSE" }))),
      el("tbody", {}, rows))),
    el("div", { class: "scroll" }, el("table", {},
      el("thead", {}, el("tr", {}, el("th", { text: "Paired difference (per day; negative = calibrated better)" }), el("th", { class: "n", text: "Mean" }),
        el("th", { class: "n", text: "95% interval" }), el("th", { text: "Result" }))),
      el("tbody", {}, pd.map((d) => diffRow(d, "mm")), pb.map((d) => diffRow(d, d.comparison.startsWith("Brier") ? "Brier" : "mm"))))));
}

async function main() {
  const root = document.getElementById("perf");
  let pf;
  try {
    const r = await fetch("data/v1/performance.json", { cache: "no-cache" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    pf = await r.json();
  } catch (e) {
    root.replaceChildren(el("p", { class: "banner error", text: `Performance data could not be loaded (${e.message}).` }));
    return;
  }
  root.replaceChildren(
    el("p", { class: "small muted", text: `Data exported ${stamp(pf.exported_utc)}.` }),
    el("div", { class: "notice" },
      el("p", { text: pf.not_evaluated.join(" ") })),
    ...pf.gauges.map((g) => el("section", { class: "section" },
      el("h2", { text: `${g.gauge_name} gauge (${g.gauge_id})` }),
      el("p", { class: "small muted", text: `Used for ${g.serves.join(" and ")}.${g.serves.length > 1 ? " These locations share one gauge, so they are one test, not several." : ""}` }),
      ...g.leads.map(leadBlock))));
}

main();
