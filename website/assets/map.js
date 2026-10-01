// Optional map view. Visual only: nothing on this page feeds or changes the forecasts.
// Third-party tiles: OpenFreeMap (base map, always), NASA GIBS Himawari and EOX Sentinel-2 (only when switched on).
import { localMoment } from "./format.js";

const STYLE = "https://tiles.openfreemap.org/styles/liberty";
const GIBS = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best";
const HIMAWARI = {
  ir: { layer: "Himawari_AHI_Band13_Clean_Infrared", matrix: "GoogleMapsCompatible_Level6", maxzoom: 6,
    label: "Cloud (infrared, day and night)" },
  vis: { layer: "Himawari_AHI_Band3_Red_Visible_1km", matrix: "GoogleMapsCompatible_Level7", maxzoom: 7,
    label: "Cloud (visible light, daytime only)" },
};
const EOX_YEAR = 2024;
const EOX = `https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-${EOX_YEAR}_3857/default/g/{z}/{y}/{x}.jpg`;
const ATTR = {
  // base-map credit comes from the OpenFreeMap style itself
  gibs: 'Himawari-9 imagery: JMA, via <a href="https://earthdata.nasa.gov/gibs" target="_blank" rel="noopener">NASA GIBS</a>',
  eox: `<a href="https://cloudless.eox.at" target="_blank" rel="noopener">EOxCloudless</a> by EOX IT Services GmbH (Contains modified Copernicus Sentinel data ${EOX_YEAR})`,
};

const $ = (id) => document.getElementById(id);
const status = (msg) => { $("map-status").textContent = msg; };

/** Newest Himawari image time (10-minute steps) that GIBS can serve over Perth; null if none in the last 3 h. */
export async function latestHimawari(key, nowMs = Date.now(), fetchFn = fetch) {
  const h = HIMAWARI[key];
  const t0 = Math.floor(nowMs / 600000) * 600000;
  for (let k = 0; k <= 18; k += 1) {
    const t = new Date(t0 - k * 600000).toISOString().replace(".000Z", "Z");
    const url = `${GIBS}/${h.layer}/default/${t}/${h.matrix}/5/18/26.png`;   // tile covering Perth at zoom 5 (x 26, y 18)
    try {
      const r = await fetchFn(url, { method: "GET", cache: "no-store" });
      if (r.ok && (r.headers.get("content-type") || "").startsWith("image/")) return t;
    } catch { /* network error: try an earlier time */ }
  }
  return null;
}

function geo(points) {
  return { type: "FeatureCollection", features: points };
}

function buildData(m) {
  const g = Object.fromEntries(m.gauges.map((x) => [x.id, x]));
  const l = Object.fromEntries(m.locations.map((x) => [x.name, x]));
  return {
    gauges: geo(m.gauges.map((x) => ({ type: "Feature", geometry: { type: "Point", coordinates: [x.lon, x.lat] },
      properties: { name: `${x.name} (${x.id})`, status: x.status, serves: x.serves.join(", ") } }))),
    places: geo(m.locations.map((x) => ({ type: "Feature", geometry: { type: "Point", coordinates: [x.lon, x.lat] },
      properties: { name: x.name } }))),
    links: geo(m.links.map((x) => ({ type: "Feature",
      geometry: { type: "LineString", coordinates: [[l[x.location].lon, l[x.location].lat], [g[x.gauge].lon, g[x.gauge].lat]] },
      properties: { status: x.status, label: `${x.distance_km} km${x.status === "planned" ? " (planned)" : ""}` } }))),
  };
}

function addOwnLayers(map, data) {
  map.addSource("links", { type: "geojson", data: data.links });
  map.addSource("gauges", { type: "geojson", data: data.gauges });
  map.addSource("places", { type: "geojson", data: data.places });
  map.addLayer({ id: "links", type: "line", source: "links",
    paint: { "line-color": ["match", ["get", "status"], "planned", "#e2b04e", "#2fb5a5"], "line-width": 2.2,
      "line-dasharray": ["match", ["get", "status"], "planned", ["literal", [2, 2]], ["literal", [1, 0]]] } });
  map.addLayer({ id: "link-labels", type: "symbol", source: "links",
    layout: { "symbol-placement": "line-center", "text-field": ["get", "label"], "text-font": ["Noto Sans Bold"],
      "text-size": 13 },
    paint: { "text-color": "#0b1420", "text-halo-color": "#ffffff", "text-halo-width": 2 } });
  map.addLayer({ id: "gauges", type: "circle", source: "gauges",
    paint: { "circle-radius": 8, "circle-color": ["match", ["get", "status"], "planned", "#0b1420", "#2fb5a5"],
      "circle-stroke-color": ["match", ["get", "status"], "planned", "#e2b04e", "#ffffff"], "circle-stroke-width": 3 } });
  map.addLayer({ id: "places", type: "circle", source: "places",
    paint: { "circle-radius": 6, "circle-color": "#4ea6e0", "circle-stroke-color": "#ffffff", "circle-stroke-width": 2 } });
  map.addLayer({ id: "point-labels", type: "symbol", source: "gauges",
    layout: { "text-field": ["concat", ["get", "name"], ["case", ["==", ["get", "status"], "planned"], "\nplanned gauge", "\ngauge"]],
      "text-font": ["Noto Sans Bold"], "text-size": 13, "text-offset": [0, 1.3], "text-anchor": "top" },
    paint: { "text-color": "#0b1420", "text-halo-color": "#ffffff", "text-halo-width": 2 } });
  map.addLayer({ id: "place-labels", type: "symbol", source: "places",
    layout: { "text-field": ["get", "name"], "text-font": ["Noto Sans Bold"], "text-size": 14,
      "text-offset": [0, -1.2], "text-anchor": "bottom" },
    paint: { "text-color": "#123a5c", "text-halo-color": "#ffffff", "text-halo-width": 2 } });
}

const FIRST_OWN_LAYER = "links";

async function toggleHimawari(map, key, on) {
  const id = `himawari-${key}`;
  if (map.getLayer(id)) map.removeLayer(id);
  if (map.getSource(id)) map.removeSource(id);
  $(`time-${key}`).textContent = "";
  if (!on) return;
  $(`time-${key}`).textContent = "finding the latest image…";
  const t = await latestHimawari(key);
  if (!t) { $(`time-${key}`).textContent = "no image available in the last 3 hours"; $(`ov-${key}`).checked = false; return; }
  const h = HIMAWARI[key];
  map.addSource(id, { type: "raster", tileSize: 256, maxzoom: h.maxzoom, attribution: ATTR.gibs,
    tiles: [`${GIBS}/${h.layer}/default/${t}/${h.matrix}/{z}/{y}/{x}.png`] });
  map.addLayer({ id, type: "raster", source: id, paint: { "raster-opacity": Number($("opacity").value) / 100 } },
    FIRST_OWN_LAYER);
  const mins = Math.round((Date.now() - Date.parse(t)) / 60000);
  $(`time-${key}`).textContent = `image taken ${localMoment(t)} AWST (${mins} min before you opened it)`;
}

function toggleEox(map, on) {
  if (map.getLayer("eox")) map.removeLayer("eox");
  if (map.getSource("eox")) map.removeSource("eox");
  if (!on) return;
  map.addSource("eox", { type: "raster", tileSize: 256, maxzoom: 15, attribution: ATTR.eox, tiles: [EOX] });
  const firstSymbol = map.getStyle().layers.find((l) => l.type === "symbol");
  map.addLayer({ id: "eox", type: "raster", source: "eox" }, firstSymbol ? firstSymbol.id : FIRST_OWN_LAYER);
}

async function main() {
  if (typeof maplibregl === "undefined") {
    status("The map library could not be loaded (offline, or blocked by the browser). The forecasts do not need it.");
    return;
  }
  let m;
  try {
    const r = await fetch("data/v1/map.json", { cache: "no-cache" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    m = await r.json();
  } catch (e) { status(`Map data could not be loaded (${e.message}).`); return; }
  let map;
  try {
    map = new maplibregl.Map({ container: "map", style: STYLE, center: [115.82, -31.76], zoom: 8.6, pitch: 45,
      bearing: -12, maxPitch: 70, attributionControl: false, cooperativeGestures: true });
  } catch (e) { status(`This browser cannot draw the 3D map (${e.message}).`); return; }
  map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
  map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
  map.addControl(new maplibregl.AttributionControl({ compact: false }), "bottom-right");
  map.on("error", (e) => { if (e && e.error && /style/i.test(String(e.error.message))) status("The base map could not be loaded."); });
  map.on("load", () => {
    addOwnLayers(map, buildData(m));
    status("");
    for (const key of Object.keys(HIMAWARI)) {
      $(`ov-${key}`).addEventListener("change", (ev) => {
        if (ev.target.checked) for (const other of Object.keys(HIMAWARI)) if (other !== key && $(`ov-${other}`).checked) {
          $(`ov-${other}`).checked = false; toggleHimawari(map, other, false);
        }
        toggleHimawari(map, key, ev.target.checked);
      });
    }
    $("ov-eox").addEventListener("change", (ev) => toggleEox(map, ev.target.checked));
    $("opacity").addEventListener("input", (ev) => {
      for (const key of Object.keys(HIMAWARI)) if (map.getLayer(`himawari-${key}`)) {
        map.setPaintProperty(`himawari-${key}`, "raster-opacity", Number(ev.target.value) / 100);
      }
    });
    map.on("click", "gauges", (e) => {
      const p = e.features[0].properties;
      new maplibregl.Popup().setLngLat(e.lngLat)
        .setText(`${p.name}: ${p.status === "planned" ? "planned gauge (not used yet)" : "gauge in use"} for ${p.serves}`)
        .addTo(map);
    });
  });
}

if (typeof document !== "undefined") main();
