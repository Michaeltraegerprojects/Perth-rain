// Run with: npm test (in website/) or: node --test website/tests/format.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import * as F from "../assets/format.js";

test("probabilities are percentages and never 0% or 100%", () => {
  assert.equal(F.pct(0.1446), "14%");
  assert.equal(F.pct(0.0036), "<1%");
  assert.equal(F.pct(0), "<1%");          // a rounded zero is shown as small, not impossible
  assert.equal(F.pct(0.995), ">99%");
  assert.equal(F.pct(null), "—");
  assert.equal(F.pct(0.5), "50%");
});

test("small non-zero rain amounts never display as 0", () => {
  assert.equal(F.mm(0), "0");
  assert.equal(F.mm(0.03), "<0.1");
  assert.equal(F.mm(0.121), "0.1");
  assert.equal(F.mm(12.6), "13");
  assert.equal(F.mm(null), "—");
  assert.equal(F.rangeText(0, 0.2), "0–0.2 mm");
});

test("the 9am-9am window is shown in Perth time and the label is the END date", () => {
  assert.equal(F.dayLabel("2026-10-03"), "Sat 3 Oct");
  assert.equal(F.windowText("2026-10-02T01:00:00Z", "2026-10-03T01:00:00Z"), "9am Fri 2 Oct → 9am Sat 3 Oct (AWST)");
});

test("median 0 mm is explained as 'dry more likely', never as impossible", () => {
  const note = F.medianNote({ amounts_mm: { median: 0 }, probabilities: { ge_0_2mm: 0.14 } });
  assert.match(note, /does not mean rain is impossible/);
  assert.match(note, /14%/);
  assert.equal(F.medianNote({ amounts_mm: { median: 1.2 }, probabilities: { ge_0_2mm: 0.7 } }), "");
});

test("stale after 18 hours, judged on the viewer's clock", () => {
  const made = "2026-10-01T02:30:00Z";
  assert.equal(F.freshness(made, Date.parse("2026-10-01T10:00:00Z")).stale, false);
  assert.equal(F.freshness(made, Date.parse("2026-10-01T21:00:00Z")).stale, true);
});

test("window states: upcoming, started, ended", () => {
  const d = { window_start_utc: "2026-10-02T01:00:00Z", window_end_utc: "2026-10-03T01:00:00Z" };
  assert.equal(F.windowState(d, Date.parse("2026-10-01T12:00:00Z")), "upcoming");
  assert.equal(F.windowState(d, Date.parse("2026-10-02T12:00:00Z")), "started");
  assert.equal(F.windowState(d, Date.parse("2026-10-03T02:00:00Z")), "ended");
});

test("primary forecast is the shortest lead; unavailable days have none", () => {
  const day = { forecasts: [
    { status: "ok", lead_group: "day3", lead_hours_to_window_start: 43 },
    { status: "ok", lead_group: "day2", lead_hours_to_window_start: 37 }] };
  assert.equal(F.primaryForecast(day).lead_group, "day2");
  assert.equal(F.primaryForecast({ forecasts: [] }), null);
});

test("status chips and routes are labelled", () => {
  assert.equal(F.verificationChip("verified").cls, "ok");
  assert.equal(F.verificationChip("unverified").text, "Unverified inputs");
  assert.equal(F.routeText({ route: "primary" }), "Selected model");
  assert.equal(F.routeText({ route: "fallback#4" }), "Backup model (choice 5)");
  assert.equal(F.leadText({ lead_hours_to_window_start: 37, lead_hours_to_window_end: 61 }), "37–61 h after model run");
});

test("comparison labels are plain language", () => {
  assert.equal(F.comparisonLabel("MAE: calibrated expected total minus raw:equal_weight_blend"),
    "Error of calibrated expected total vs raw equal-weight blend");
  assert.equal(F.comparisonLabel("Brier (event rain >= 0.2 mm): calibrated minus flat climatology"),
    "Rain/no-rain score vs season-blind climatology");
});

test("latest Himawari image: newest 10-minute time the tile service can serve, else null", async () => {
  const { latestHimawari } = await import("../assets/map.js");
  const now = Date.parse("2026-10-01T03:17:00Z");
  const seen = [];
  const fake = async (url) => {
    seen.push(url);
    const ok = url.includes("2026-10-01T02:30:00Z");
    return { ok, headers: { get: () => (ok ? "image/png" : "application/xml") } };
  };
  assert.equal(await latestHimawari("ir", now, fake), "2026-10-01T02:30:00Z");
  assert.ok(seen[0].includes("2026-10-01T03:10:00Z"));            // starts at the current 10-minute step
  assert.ok(seen.every((u) => u.includes("/5/18/26.png")));       // the tile over Perth
  assert.equal(await latestHimawari("ir", now, async () => ({ ok: false, headers: { get: () => "" } })), null);
});

test("challenge verdicts never declare a winner without evidence", async () => {
  const { verdictText, verdictClass } = await import("../assets/challenge.js");
  const base = { first: "Our forecast (median)", second: "NOAA GFS (raw)", first_key: "champion_median" };
  assert.equal(verdictText({ ...base, verdict: "insufficient evidence" }), "Insufficient evidence");
  assert.equal(verdictText({ ...base, verdict: "inconclusive" }), "Cannot tell apart");
  assert.equal(verdictText({ ...base, verdict: "first better" }), "Our forecast (median) better");
  assert.equal(verdictClass({ ...base, verdict: "second better" }), "worse");
  assert.equal(verdictClass({ ...base, verdict: "inconclusive" }), "ns");
});

test("backup-model forecasts say their accuracy has not been measured", () => {
  const unscored = { is_fallback: true, heldout_scored: false };
  assert.equal(F.routeChip(unscored).text, "Backup model · accuracy not yet measured");
  assert.match(F.routeNote(unscored), /has not been measured/);
  assert.equal(F.routeChip({ is_fallback: false, heldout_scored: true }), null);
  assert.equal(F.routeNote({ is_fallback: false, heldout_scored: true }), "");
  assert.equal(F.routeChip({ is_fallback: true, heldout_scored: true }).text, "Backup model");
});

test("inputs checked only at issue are not labelled as verified", () => {
  const c = F.verificationChip("checked_at_issue");
  assert.equal(c.text, "Inputs checked at issue");
  assert.notEqual(c.cls, "ok");
  assert.match(F.VERIFICATION_NOTE.checked_at_issue, /cannot be repeated/);
});

test("CRPS comparisons are separated from rain-amount comparisons", async () => {
  const { splitPaired } = await import("../assets/challenge.js");
  const rows = [{ score: "absolute error" }, { score: "CRPS" }, { score: "absolute error" }];
  const { amount, crps } = splitPaired(rows);
  assert.equal(amount.length, 2);
  assert.equal(crps.length, 1);
});

test("outlook lines say whether they are calibrated or raw guidance", async () => {
  const { sourceLabel } = await import("../assets/outlook.js");
  assert.equal(sourceLabel({ calibrated: true }), "Our calibrated forecast");
  assert.equal(sourceLabel({ calibrated: false }), "Raw model guidance");
});

test("availability is labelled observed only with evidence, otherwise as the 6-hour estimate", async () => {
  const { availabilityText } = await import("../assets/challenge.js");
  assert.equal(availabilityText({ availability_basis: "observed" }), "Observed");
  assert.equal(availabilityText({ availability_basis: "estimated (initialisation + 6 h)" }), "Estimated (+6 h rule)");
  assert.equal(availabilityText(null), "—");
});
