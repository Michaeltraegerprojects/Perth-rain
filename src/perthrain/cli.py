"""Command-line interface: ``python -m perthrain <command> [options]``."""
from __future__ import annotations

import argparse
import json
from datetime import date

from .config import load_configs


def _common(p: argparse.ArgumentParser, ingest: bool = True) -> None:
    """``ingest`` flags (location, station, period, CDO files) only matter to commands that build datasets;
    calibrate / predict / report / scan-enso read the finished dataset, so they do not accept them."""
    p.add_argument("--config", default=None, help="TOML config file (see config.example.toml)")
    p.add_argument("--location", help="run only this configured location (default: all in [[locations]])")
    p.add_argument("--data-dir", dest="data_dir")
    p.add_argument("--reports-dir", dest="reports_dir")
    if not ingest:
        return
    p.add_argument("--name", dest="location_name", help="location/suburb label")
    p.add_argument("--lat", dest="latitude", type=float, help="target latitude (decimal degrees)")
    p.add_argument("--lon", dest="longitude", type=float, help="target longitude (decimal degrees)")
    p.add_argument("--coord-source", dest="coordinate_source", help="where the coordinates came from")
    p.add_argument("--timezone", help="IANA time zone of the gauge (default Australia/Perth)")
    p.add_argument("--station", dest="station_id", help="force a BoM station number, e.g. 009225")
    p.add_argument("--radius", dest="search_radius_km", type=float, help="station search radius km")
    p.add_argument("--start", help="first label date YYYY-MM-DD (default auto)")
    p.add_argument("--end", help="last label date YYYY-MM-DD (default auto)")
    p.add_argument("--cdo-csv", dest="cdo_csv_paths", action="append",
                   help="manually downloaded BoM CDO IDCJAC0009 CSV for the selected station (repeatable)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="perthrain", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, hlp in [("stations", "list nearby BoM stations and the auto-selected gauge"),
                      ("probe", "small test download + timestamp/unit verification"),
                      ("run", "full download, clean, join, QC, baseline and reports"),
                      ("report", "rebuild the data-quality report from files written by `run`"),
                      ("scan-enso", "scan artifacts, payloads and reports for ENSO / climate-index references"),
                      ("calibrate", "fit + validate the local rainfall calibration models (chronological holdout)"),
                      ("predict", "predict upcoming gauge-day rainfall from live forecasts")]:
        sp = sub.add_parser(name, help=hlp)
        _common(sp, ingest=name in ("stations", "probe", "run"))
        if name == "run":
            sp.add_argument("--skip-single-runs", action="store_true",
                            help="skip the (slow, ~1 request per run) Single Runs download")
        if name in ("predict", "calibrate"):
            sp.add_argument("--enso-experimental", action="store_true",
                            help="EXPERIMENTAL: include the UNVERIFIED El Nino/IOD feature "
                                 "(default: no-ENSO baseline)")
        if name == "predict":
            sp.add_argument("--horizon-days", type=int, default=4, help="how many days ahead to predict")
        if name == "scan-enso":
            sp.add_argument("--stage", default="current", help="label for the scan report file")
        if name == "predict":
            sp.add_argument("--frozen-inputs", action="store_true",
                            help="re-use cached live forecast responses instead of downloading (replay)")
            sp.add_argument("--as-of", help="UTC time to predict as of (with --frozen-inputs), e.g. 2026-09-30T14:13Z")
        if name == "probe":
            sp.add_argument("--probe-start", default="2026-06-01")
            sp.add_argument("--probe-end", default="2026-06-14")
    args = ap.parse_args(argv)
    if getattr(args, "as_of", None) and not getattr(args, "frozen_inputs", False):
        ap.error("--as-of needs --frozen-inputs: live forecasts cannot be replayed as of a past time")
    overrides = {k: v for k, v in vars(args).items()
                 if k not in {"cmd", "config", "skip_single_runs", "probe_start", "probe_end", "location",
                              "horizon_days", "enso_experimental", "stage", "frozen_inputs", "as_of"}}
    cfgs = load_configs(args.config, only=args.location, **overrides)

    from . import pipeline
    pipeline.setup_logging(cfgs[0])
    if args.cmd == "scan-enso":           # project-wide, not per location
        from pathlib import Path
        from .enso_guard import write_scan
        print(write_scan(Path(cfgs[0].raw_root).parent, args.stage))
        return
    summaries = []
    for cfg in cfgs:
        if len(cfgs) > 1:
            print(f"\n===== {cfg.location_name} =====")
        summaries.append(_run_one(args, cfg))
    if len(cfgs) > 1 and args.cmd == "run":
        pipeline.write_combined_summary(cfgs, [s for s in summaries if s])


def _run_one(args, cfg):
    from . import pipeline
    if args.cmd == "stations":
        f = pipeline.make_fetcher(cfg)
        first, last = pipeline.resolve_period(cfg)
        st, _, cand = pipeline.select_station(cfg, f, first, last)
        cols = ["station_id", "station_name", "distance_km", "db_start_date", "downloadable", "completeness",
                "id_ambiguous", "selected"]
        print(cand[cols].to_string(index=False))
        print("\nselected:", st["station_id"], st["station_name"], "-", st["selection_reason"])
    elif args.cmd == "probe":
        from .probe import run_probe
        res = run_probe(cfg, date.fromisoformat(args.probe_start), date.fromisoformat(args.probe_end))
        print(json.dumps(res, indent=1, default=str))
        return None
    elif args.cmd == "report":
        print(pipeline.rebuild_report(cfg))
        return None
    elif args.cmd == "calibrate":
        from .calibrate import run_calibration
        sel = run_calibration(cfg, use_climate=args.enso_experimental)
        print(json.dumps({k: v["selected"] for k, v in sel.items() if not k.startswith("_")}, indent=1, default=str))
        return None
    elif args.cmd == "predict":
        import pandas as pd
        from .predict import run_predict
        now = pd.Timestamp(args.as_of) if args.as_of else None
        if now is not None and now.tzinfo is None:
            now = now.tz_localize("UTC")
        run_predict(cfg, horizon_days=args.horizon_days, now=now, use_climate=args.enso_experimental,
                    refresh_live=not args.frozen_inputs)
        sub = "predictions_enso_experimental" if args.enso_experimental else "predictions"
        print((cfg.data_dir / sub / "latest.md").read_text(encoding="utf-8"))
        return None
    elif args.cmd == "run":
        summary = pipeline.run_all(cfg, skip_single_runs=args.skip_single_runs)
        print(json.dumps(summary, indent=1, default=str))
        return summary


if __name__ == "__main__":
    main()
