"""GE-Sentinel REST API.

Serves persisted surveillance output (alerts, prices, memos, eval report)
from the same database the pipeline writes. Run after a demo or alongside
the live collector:

    uvicorn api.main:app --reload
"""
from __future__ import annotations

import json

import sqlalchemy as sa
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ge_sentinel import config, db

app = FastAPI(
    title="GE-Sentinel API",
    version="0.1.0",
    description="Market-manipulation & RMT alerts for the OSRS Grand Exchange",
)
engine = db.init_db()


class Alert(BaseModel):
    id: int
    item_id: int
    ts_start: int
    ts_end: int
    score: float
    kind_guess: str | None
    evidence: dict


class PriceRow(BaseModel):
    ts: int
    avg_high: float | None
    avg_low: float | None
    high_vol: int
    low_vol: int
    source: str


def _alert(r) -> Alert:
    return Alert(id=r.id, item_id=r.item_id, ts_start=r.ts_start, ts_end=r.ts_end,
                 score=r.score, kind_guess=r.kind_guess,
                 evidence=json.loads(r.evidence_json or "{}"))


@app.get("/health")
def health():
    with engine.connect() as cx:
        n_alerts = cx.execute(sa.select(sa.func.count()).select_from(db.alerts)).scalar_one()
        n_prices = cx.execute(sa.select(sa.func.count()).select_from(db.prices_5m)).scalar_one()
    return {"status": "ok", "alerts": int(n_alerts), "price_rows": int(n_prices)}


@app.get("/alerts", response_model=list[Alert])
def list_alerts(limit: int = 20):
    q = sa.select(db.alerts).order_by(db.alerts.c.score.desc()).limit(max(1, min(limit, 200)))
    with engine.connect() as cx:
        return [_alert(r) for r in cx.execute(q)]


@app.get("/alerts/{alert_id}", response_model=Alert)
def get_alert(alert_id: int):
    with engine.connect() as cx:
        r = cx.execute(sa.select(db.alerts).where(db.alerts.c.id == alert_id)).first()
    if r is None:
        raise HTTPException(404, "alert not found")
    return _alert(r)


@app.get("/items/{item_id}/prices", response_model=list[PriceRow])
def item_prices(item_id: int, limit: int = 500):
    q = (sa.select(db.prices_5m).where(db.prices_5m.c.item_id == item_id)
         .order_by(db.prices_5m.c.ts.desc()).limit(max(1, min(limit, 5000))))
    with engine.connect() as cx:
        rows = cx.execute(q).all()
    if not rows:
        raise HTTPException(404, "no prices for item")
    return [PriceRow(ts=r.ts, avg_high=r.avg_high, avg_low=r.avg_low,
                     high_vol=r.high_vol, low_vol=r.low_vol, source=r.source)
            for r in reversed(rows)]


@app.get("/report")
def report():
    p = config.OUTPUT_DIR / "eval_report.json"
    if not p.exists():
        raise HTTPException(404, "no report yet — run: python -m ge_sentinel.cli demo")
    return json.loads(p.read_text())


@app.get("/memos")
def memos():
    return {"memos": sorted(f.name for f in config.MEMO_DIR.glob("alert_*.md"))}


@app.get("/memos/{name}")
def memo(name: str):
    if "/" in name or "\\" in name or ".." in name or not name.endswith(".md"):
        raise HTTPException(400, "bad memo name")
    p = config.MEMO_DIR / name
    if not p.exists():
        raise HTTPException(404, "memo not found")
    return {"name": name, "markdown": p.read_text(encoding="utf-8")}
