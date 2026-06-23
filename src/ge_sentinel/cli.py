"""Command-line interface.

  ge-sentinel demo [--fast]      full offline run: data → detection → eval → memos
  ge-sentinel collect [--loop N] live 5-minute sweeps from the wiki API
  ge-sentinel backfill IDS...    per-item /timeseries history
  ge-sentinel gaps ITEM_ID       missing-bucket report
"""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ge-sentinel",
        description="Market-manipulation & RMT anomaly detection for the OSRS Grand Exchange",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="end-to-end offline demo with evaluation")
    d.add_argument("--fast", action="store_true", help="small market (CI/tests)")
    d.add_argument("--no-memos", action="store_true", help="skip analyst memos")

    c = sub.add_parser("collect", help="collect live prices (needs internet + GE_SENTINEL_UA)")
    c.add_argument("--loop", type=int, default=None, metavar="N",
                   help="run N timed sweeps; omit for a single sweep")
    c.add_argument("--sync-mapping", action="store_true",
                   help="refresh item metadata from /mapping first")

    b = sub.add_parser("backfill", help="per-item /timeseries backfill")
    b.add_argument("ids", nargs="+", type=int)
    b.add_argument("--timesteps", default="5m", help="comma list from 5m,1h,6h,24h")

    g = sub.add_parser("gaps", help="missing 5-minute buckets for an item")
    g.add_argument("item_id", type=int)

    args = p.parse_args(argv)

    if args.cmd == "demo":
        from .pipeline import run_demo
        run_demo(fast=args.fast, with_memos=not args.no_memos)

    elif args.cmd == "collect":
        from . import db
        from .api_client import WikiPricesClient
        from .ingest import collect_loop, collect_once, sync_mapping
        engine = db.init_db()
        with WikiPricesClient() as client:
            if args.sync_mapping:
                print(f"mapping: {sync_mapping(engine, client)} items refreshed")
            if args.loop is None:
                print(f"collected {collect_once(engine, client)} rows")
            else:
                collect_loop(engine, client, iterations=args.loop)

    elif args.cmd == "backfill":
        from . import db
        from .api_client import WikiPricesClient
        from .ingest import backfill
        engine = db.init_db()
        with WikiPricesClient() as client:
            n = backfill(engine, client, args.ids, tuple(args.timesteps.split(",")))
        print(f"backfilled {n} five-minute rows for {len(args.ids)} item(s)")

    elif args.cmd == "gaps":
        from . import db
        from .ingest import gap_report
        engine = db.init_db()
        rep = gap_report(engine, args.item_id)
        print(rep.to_string(index=False) if len(rep) else "no gaps found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
