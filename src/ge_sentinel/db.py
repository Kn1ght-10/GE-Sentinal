"""Database layer.

SQLAlchemy Core schema + dialect-aware idempotent upserts. The same schema
ships as TimescaleDB DDL in db/migrations/001_timescale.sql for production.
"""
from __future__ import annotations

import json
import time
from typing import Iterable, Mapping

import sqlalchemy as sa

from . import config

metadata = sa.MetaData()

items = sa.Table(
    "items", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("members", sa.Boolean),
    sa.Column("ge_limit", sa.Integer),
    sa.Column("highalch", sa.Integer),
    sa.Column("source", sa.Text, nullable=False, server_default="unknown"),
)

prices_5m = sa.Table(
    "prices_5m", metadata,
    sa.Column("item_id", sa.Integer, primary_key=True),
    sa.Column("ts", sa.Integer, primary_key=True),  # unix, bucket start
    sa.Column("avg_high", sa.Float),
    sa.Column("avg_low", sa.Float),
    sa.Column("high_vol", sa.Integer, nullable=False, server_default="0"),
    sa.Column("low_vol", sa.Integer, nullable=False, server_default="0"),
    sa.Column("source", sa.Text, nullable=False, server_default="api"),
)

truth_events = sa.Table(
    "truth_events", metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("item_id", sa.Integer, nullable=False),
    sa.Column("kind", sa.Text, nullable=False),  # pump_dump | rmt_spike | patch_shock
    sa.Column("start_ts", sa.Integer, nullable=False),
    sa.Column("end_ts", sa.Integer, nullable=False),
    sa.Column("magnitude", sa.Float),
    sa.Column("note", sa.Text),
)

alerts = sa.Table(
    "alerts", metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("item_id", sa.Integer, nullable=False),
    sa.Column("ts_start", sa.Integer, nullable=False),
    sa.Column("ts_end", sa.Integer, nullable=False),
    sa.Column("score", sa.Float, nullable=False),
    sa.Column("kind_guess", sa.Text),
    sa.Column("evidence_json", sa.Text),
    sa.Column("created_ts", sa.Integer, nullable=False),
)


def get_engine(url: str | None = None) -> sa.Engine:
    return sa.create_engine(url or config.DB_URL, future=True)


def init_db(engine: sa.Engine | None = None) -> sa.Engine:
    engine = engine or get_engine()
    metadata.create_all(engine)
    return engine


def upsert(engine: sa.Engine, table: sa.Table, rows: Iterable[Mapping]) -> int:
    """Idempotent insert-or-update keyed on the table's primary key."""
    rows = [dict(r) for r in rows]
    if not rows:
        return 0
    dialect = engine.dialect.name
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as d_insert
    elif dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as d_insert
    else:  # pragma: no cover - fallback for exotic backends
        with engine.begin() as cx:
            cx.execute(table.insert(), rows)
        return len(rows)
    pk = [c.name for c in table.primary_key.columns]
    stmt = d_insert(table)
    update_cols = {c.name: stmt.excluded[c.name] for c in table.columns if c.name not in pk}
    stmt = stmt.on_conflict_do_update(index_elements=pk, set_=update_cols)
    with engine.begin() as cx:
        for i in range(0, len(rows), 5000):
            cx.execute(stmt, rows[i : i + 5000])
    return len(rows)


def record_alerts(engine: sa.Engine, episodes: list[dict]) -> int:
    now = int(time.time())
    rows = [
        dict(
            item_id=e["item_id"], ts_start=e["ts_start"], ts_end=e["ts_end"],
            score=float(e["score"]), kind_guess=e.get("kind_guess"),
            evidence_json=json.dumps(e.get("evidence", {})), created_ts=now,
        )
        for e in episodes
    ]
    with engine.begin() as cx:
        cx.execute(alerts.delete())
        if rows:
            cx.execute(alerts.insert(), rows)
    return len(rows)
