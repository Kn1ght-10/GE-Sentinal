"""Ingestion: real seed data, live collection, backfill, and gap detection.

Two data paths feed the same prices_5m table:
  * seed: the committed real API excerpt under data/seed/real (offline demo),
  * live: collect_once()/backfill() against the wiki API (production path).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

from . import config, db
from .api_client import WikiPricesClient

SEED_ITEM_NAMES = {4151: "Abyssal whip"}


def _rows_from_timeseries(item_id: int, payload: dict, source: str) -> list[dict]:
    return [
        dict(
            item_id=item_id, ts=int(p["timestamp"]),
            avg_high=p.get("avgHighPrice"), avg_low=p.get("avgLowPrice"),
            high_vol=int(p.get("highPriceVolume") or 0),
            low_vol=int(p.get("lowPriceVolume") or 0),
            source=source,
        )
        for p in payload.get("data", [])
    ]


def load_real_seed(engine: sa.Engine, seed_dir: Path | None = None) -> int:
    """Load committed real API excerpts so the demo runs fully offline."""
    seed_dir = seed_dir or config.SEED_REAL_DIR
    total = 0
    for f in sorted(seed_dir.glob("*.json")):
        payload = json.loads(f.read_text())
        item_id = int(payload["itemId"])
        db.upsert(engine, db.items, [dict(
            id=item_id, name=SEED_ITEM_NAMES.get(item_id, f"item_{item_id}"),
            members=None, ge_limit=None, highalch=None, source="seed_real",
        )])
        total += db.upsert(engine, db.prices_5m,
                           _rows_from_timeseries(item_id, payload, "seed_real"))
    return total


def sync_mapping(engine: sa.Engine, client: WikiPricesClient) -> int:
    rows = [dict(id=m["id"], name=m["name"], members=m.get("members"),
                 ge_limit=m.get("limit"), highalch=m.get("highalch"), source="mapping")
            for m in client.mapping()]
    return db.upsert(engine, db.items, rows)


def collect_once(engine: sa.Engine, client: WikiPricesClient) -> int:
    """One /5m sweep: every item, one HTTP call, idempotent upsert."""
    payload = client.five_minute()
    ts = int(payload["timestamp"])
    rows = [dict(item_id=int(iid), ts=ts,
                 avg_high=v.get("avgHighPrice"), avg_low=v.get("avgLowPrice"),
                 high_vol=int(v.get("highPriceVolume") or 0),
                 low_vol=int(v.get("lowPriceVolume") or 0), source="live")
            for iid, v in payload["data"].items()]
    return db.upsert(engine, db.prices_5m, rows)


def collect_loop(engine: sa.Engine, client: WikiPricesClient,
                 iterations: int | None = None) -> None:
    """Simple resident collector; production uses GH Actions cron or Prefect."""
    i = 0
    while iterations is None or i < iterations:
        n = collect_once(engine, client)
        print(f"[collect] upserted {n} rows at {int(time.time())}")
        i += 1
        # Sleep to the next 5-minute boundary (+15s grace for API aggregation).
        now = time.time()
        time.sleep(max(5.0, config.PERIOD_S - (now % config.PERIOD_S) + 15))


def backfill(engine: sa.Engine, client: WikiPricesClient, item_ids: list[int],
             timesteps: tuple[str, ...] = ("5m", "1h", "6h", "24h"),
             pause_s: float = 1.0) -> int:
    """Per-item /timeseries backfill (≤365 points per granularity), politely paced."""
    total = 0
    for iid in item_ids:
        for step in timesteps:
            payload = client.timeseries(iid, step)
            if step == "5m":  # only 5m lands in prices_5m; coarser steps for context
                total += db.upsert(engine, db.prices_5m,
                                   _rows_from_timeseries(iid, payload, f"backfill_{step}"))
            time.sleep(pause_s)
    return total


def gap_report(engine: sa.Engine, item_id: int) -> pd.DataFrame:
    """Find missing 5-minute buckets — RuneLite downtime, game updates, outages."""
    q = sa.select(db.prices_5m.c.ts).where(db.prices_5m.c.item_id == item_id).order_by(db.prices_5m.c.ts)
    with engine.connect() as cx:
        ts = pd.Series([r[0] for r in cx.execute(q)], dtype="int64")
    if len(ts) < 2:
        return pd.DataFrame(columns=["gap_start", "gap_end", "missing_buckets"])
    d = ts.diff().dropna()
    bad = d[d != config.PERIOD_S]
    return pd.DataFrame({
        "gap_start": ts.shift(1)[bad.index].astype(int),
        "gap_end": ts[bad.index].astype(int),
        "missing_buckets": (bad // config.PERIOD_S - 1).astype(int),
    }).reset_index(drop=True)


def load_prices(engine: sa.Engine, item_ids: list[int] | None = None) -> pd.DataFrame:
    q = sa.select(db.prices_5m)
    if item_ids:
        q = q.where(db.prices_5m.c.item_id.in_(item_ids))
    with engine.connect() as cx:
        return pd.read_sql(q, cx)
