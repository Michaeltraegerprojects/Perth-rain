# Perth Rain Gauges: website

A static site (plain HTML, CSS and JavaScript; no framework, no build step, no dependencies) that shows the
pipeline's calibrated 9am-to-9am rainfall forecasts and their held-out performance.

The site only reads the reviewed export in `data/v1/` (`manifest.json`, `forecast.json`, `performance.json`,
`map.json`, `challenge.json`, `outlook.json`). It never runs the Python pipeline, never loads model files, and cannot
refresh forecasts by itself. The forecast and outlook pages show when they were made and flag anything older than
18 hours as stale; the other pages show their export time.

Forecast cards say which model produced them. A backup model whose accuracy on past days has not been measured is
labelled as such. Inputs checked against the archived model run at issue time are labelled "checked at issue",
because the downloaded data that was checked is not kept. A forecast whose inputs fail that check (for example, an
input credited to a model run that had not been published yet) is withdrawn: the day shows "Forecast withdrawn",
never a number.

## Pages

- `index.html`: location selector (Perth, Clarkson, Ocean Reef, Fremantle). Each location shows:
  - the gauge name and distance, with a "gauge forecast, not property forecast" notice;
  - one card per rain day, with the exact window times;
  - verified, unverified, unavailable and stale states;
  - expandable raw-model totals and provenance.
- `map.html` (optional, visual only): a 3D map of each suburb, its gauge and the distance between them, with
  Tamala Park shown as the *planned* Clarkson gauge. It has switchable layers:
  - Himawari-9 cloud imagery (infrared or visible) from NASA GIBS, labelled with the time the image was taken;
  - a Sentinel-2 satellite photo base (EOxCloudless 2024).

  Nothing on this page feeds the forecasts.
- `outlook.html`: an easy-to-read 7-day outlook for Perth. Each day is labelled "Our calibrated forecast" (gauge rain
  chance from the pipeline) or "Raw model guidance" (agreement among five global models, never shown as a percentage).
  It is a snapshot with its own time stamp and a stale warning.
- `challenge.html`: the Forecast Challenge. Rain-amount comparisons and whole-distribution (CRPS) comparisons are shown
  in separate tables, because a forecast with a spread usually beats a single number on CRPS.
- `performance.html`: held-out results per gauge and lead time. It shows sample sizes, test dates, raw models, the
  blend, climatology and paired intervals.

## Local preview

From the project root:

```bash
.venv/Scripts/python -m http.server 8765 --bind 127.0.0.1 --directory website
```

Then open http://127.0.0.1:8765/. A web server is needed: the pages use JavaScript modules, which browsers don't
load from `file://`.

## Tests

```bash
.venv/Scripts/python -m pytest -q website/tests
```
```bash
npm --prefix website test
```
```bash
npm --prefix website run check
```

- The pytest suite checks the published data:
  - it passes the export validator;
  - it contains no local paths, e-mail addresses, credentials, model files or datasets;
  - unavailable days carry no numbers, and probabilities are ordered;
  - every referenced file exists.
- The Node tests cover the display rules:
  - percentages, with `<1%` instead of 0 %;
  - small amounts never shown as 0;
  - the window label marks the END of the window;
  - the median-zero wording, staleness, window states and route labels.

## Refreshing the data (manual, reviewed)

1. Run the pipeline as usual (`predict`, and `scripts/verify_served_rows.py` so each row has a verification status).
2. Export to the staging folder. This also validates the output:

   ```bash
   .venv/Scripts/python website/tools/export_site_data.py
   ```

3. Review `website/data/pending/` (open it in the local preview by temporarily copying, or read the JSON).
4. Approve. This re-validates and copies `pending/` to `data/v1/`:

   ```bash
   .venv/Scripts/python website/tools/export_site_data.py --approve
   ```

5. Run the tests above, then commit `website/data/v1/` and publish. Publishing is a manual step; nothing is
   scheduled.

Before step 2, run `scripts/verify_served_rows.py` and `scripts/seven_day_outlook.py` so the export has current
verification statuses and a current outlook.

The exporter reads only CSV, JSON and parquet outputs. It refuses to publish anything containing local paths,
e-mail addresses or credential-like fields.

## Hosting

The site is static, so no backend is needed: all computation happens in the local pipeline before export.

| Option | Cost | Private source repo | Notes |
|---|---|---|---|
| GitHub Pages | Free for public repositories | Pages from a private repo needs a paid GitHub plan | Simplest if the repository is public |
| Cloudflare Pages | Free tier | Yes | Connects to GitHub; deploys a folder |
| Netlify | Free tier | Yes | Similar; drag-and-drop deploys also possible |

`deploy/github-pages-workflow.yml` (in the project root) is a ready GitHub Actions workflow that publishes only the
`website/` folder. It is deliberately **not** in `.github/workflows/`, so nothing runs until it is approved and
copied there. It has no schedule: it runs on manual dispatch only. Its check job is read-only; only the deploy job can
write to Pages. Actions are pinned to commit SHAs and Python packages to exact versions.

To publish with GitHub Pages (repository must be public on a free plan):

1. Copy `deploy/github-pages-workflow.yml` to `.github/workflows/pages.yml`, commit and push.
2. On GitHub: Settings → Pages → Build and deployment → Source: **GitHub Actions**.
3. Actions tab → "Publish website to GitHub Pages" → **Run workflow**.
4. The site appears at `https://<account>.github.io/<repository>/`.

## Privacy and third-party services

- **Forecast and Performance pages:** they load nothing from other sites. There are no analytics, cookies or
  external fonts. The only browser storage is the last chosen location (`localStorage`, on the viewer's device).
- **Map page:** it contacts other services, and says so at the top.

  | Service | What it supplies | When | Terms |
  |---|---|---|---|
  | jsDelivr | MapLibre GL JS 5.24.0 (BSD-3) | always | version pinned, SRI integrity hash checked |
  | OpenFreeMap | base map and 3D buildings | always | free, no key; credit "OpenFreeMap © OpenMapTiles, data from OpenStreetMap" |
  | NASA GIBS | Himawari-9 cloud images (JMA) | only when switched on | free, no key; credit NASA GIBS and JMA |
  | EOX | Sentinel-2 cloudless mosaic | only when switched on | CC BY-NC-SA 4.0, **non-commercial use only**, credit shown; commercial use needs an EOX licence |

  No Windy, Google or BoM imagery is embedded. A Windy assessment was made locally; it is not part of the repository.
