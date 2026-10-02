import { stamp, freshness, dayLabel } from "./format.js";

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined && k !== false) n.append(k);
  return n;
}

/** Label for one day's rain line: our calibrated gauge forecast, or raw model agreement. */
export function sourceLabel(day) {
  return day.calibrated ? "Our calibrated forecast" : "Raw model guidance";
}

async function main() {
  const root = document.getElementById("outlook");
  let d;
  try {
    const r = await fetch("data/v1/outlook.json", { cache: "no-cache" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    d = await r.json();
  } catch (e) {
    root.replaceChildren(el("p", { class: "banner error", text: `The outlook could not be loaded (${e.message}).` }));
    return;
  }
  const fr = freshness(d.generated_utc);
  root.replaceChildren(
    el("section", { class: "loc-head" },
      el("h1", { text: d.headline }),
      el("p", { class: "stand", text: d.standfirst }),
      el("div", { class: "updates" },
        el("span", {}, "Written ", el("b", { text: stamp(d.generated_utc) })),
        el("span", {}, "Model guidance fetched ", el("b", { text: stamp(d.model_guidance_retrieved_utc) }))),
      fr.stale ? el("p", { class: "banner stale", text: `This outlook is ${Math.round(fr.ageHours)} hours old. It is a snapshot and does not update by itself.` }) : null),
    el("section", { "aria-labelledby": "week" },
      el("h2", { id: "week", text: "Perth: the week ahead" }),
      el("ul", { class: "wk" }, d.days.map((day) => el("li", {},
        el("h3", { class: "wk-day", text: dayLabel(day.date) }),
        el("span", { class: "cond", text: `${day.condition} · ${day.temps}` }),
        day.gust ? el("span", { class: "small muted", text: day.gust }) : null,
        el("span", { class: "rain", text: day.rain }),
        el("span", { class: `chip ${day.calibrated ? "ok" : "na"}`, text: sourceLabel(day) }))))),
    el("section", { "aria-labelledby": "burbs" },
      el("h2", { id: "burbs", text: "Around the suburbs" }),
      el("ul", { class: "burbs" }, d.suburbs.map((s) => el("li", { text: s })))),
    el("section", { "aria-labelledby": "how" },
      el("h2", { id: "how", text: "How this was put together" }),
      el("p", { class: "small muted", text: d.method }),
      el("p", { class: "small muted", text: d.sources })));
}

if (typeof document !== "undefined") main();
