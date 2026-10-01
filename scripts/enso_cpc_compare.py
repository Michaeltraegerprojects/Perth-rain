"""Diagnostic: BoM nino_3.4.csv vs NOAA CPC weekly OISST Nino3.4 anomaly (numbers for enso_source_verification.md).
Uses the cached BoM zip (data/raw/bom_ftp/climate/IDCK000081.zip) and a CPC file path given on the command line."""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perthrain.noaa_cpc import parse_cpc_weekly  # noqa: E402

z = zipfile.ZipFile(ROOT / "data" / "raw" / "bom_ftp" / "climate" / "IDCK000081.zip")
b = pd.read_csv(io.BytesIO(z.read("nino_3.4.csv")))
b = pd.DataFrame({"start": pd.to_datetime(b.start_date.astype(str)), "v": b["nino_3.4"].astype(float)})
b["center"] = b.start + pd.Timedelta(days=3)
cpc = parse_cpc_weekly(Path(sys.argv[1]).read_text()).rename(columns={"week_center": "center"})
m = pd.merge_asof(b.sort_values("center"), cpc[["center", "nino34_ssta"]].sort_values("center"), on="center",
                  direction="nearest", tolerance=pd.Timedelta(days=2))
m["d"] = m.v - m.nino34_ssta
clim_sd = (cpc.nino34_sst - cpc.nino34_ssta).groupby(cpc.center.dt.isocalendar().week).std().max()
for name, g in [("full", m), ("2016+", m[m.center >= "2016-01-01"])]:
    g = g.dropna(subset=["d"])
    print(f"{name}: weeks={len(g)} corr={g[['v', 'nino34_ssta']].corr().iloc[0, 1]:.3f} mean={g.d.mean():+.3f} "
          f"sd={g.d.std():.3f} max={g.d.abs().max():.2f}")
print(f"CPC implied-climatology max sd within calendar week: {clim_sd:.3f}")
print("latest:", m.tail(1)[["start", "v", "center", "nino34_ssta"]].to_dict("records"))
