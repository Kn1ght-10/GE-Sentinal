"""GE-Sentinel dashboard.

    streamlit run dashboard/app.py

Reads only persisted pipeline output (DB + eval report + memos); never
recomputes models, so it stays instant.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy as sa
import streamlit as st

from ge_sentinel import config, db
from ge_sentinel.ingest import gap_report

st.set_page_config(page_title="GE-Sentinel", layout="wide")
st.title("GE-Sentinel — Grand Exchange market surveillance")


@st.cache_resource
def get_engine():
    return db.init_db()


engine = get_engine()


@st.cache_data(ttl=60)
def load_alerts() -> pd.DataFrame:
    with engine.connect() as cx:
        return pd.read_sql(sa.select(db.alerts).order_by(db.alerts.c.score.desc()), cx)


@st.cache_data(ttl=60)
def load_prices(item_id: int) -> pd.DataFrame:
    q = (sa.select(db.prices_5m).where(db.prices_5m.c.item_id == int(item_id))
         .order_by(db.prices_5m.c.ts))
    with engine.connect() as cx:
        return pd.read_sql(q, cx)


alerts = load_alerts()
if alerts.empty:
    st.warning("No alerts in the database yet. Run `python -m ge_sentinel.cli demo` first.")
    st.stop()

# --- headline validation metrics -------------------------------------------
rep_path = config.OUTPUT_DIR / "eval_report.json"
if rep_path.exists():
    rep = json.loads(rep_path.read_text())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"precision@{rep['k']}", rep["precision_at_k"])
    c2.metric("event recall", rep["recall_manip_events"])
    c3.metric("row-level AUC", rep.get("row_level_auc"))
    c4.metric("median delay (min)", rep.get("median_detection_delay_min"))
    st.caption("Validation on **synthetic injected events** (held-out window); "
               "models train on weak labels only. Real rows are tagged "
               "`seed_real`/`live` and carry no ground truth.")

# --- item selector + price chart with alert overlay --------------------------
ids = list(dict.fromkeys([4151] + alerts["item_id"].tolist()))


def _label(i: int) -> str:
    if i == 4151:
        return "4151 — Abyssal whip (REAL data)"
    return f"{i} — synthetic" if i >= 900_000 else str(i)


sel = st.sidebar.selectbox("Item", ids, format_func=_label)
st.sidebar.caption(f"alert threshold τ = {config.ALERT_TAU}")
st.sidebar.caption(f"db: `{config.DB_URL.split('///')[-1]}`")

prices = load_prices(sel)
if prices.empty:
    st.info("No price rows for this item.")
else:
    mid = prices[["avg_high", "avg_low"]].mean(axis=1)
    t = pd.to_datetime(prices["ts"], unit="s")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, mid, lw=1.1, color="#1f4e79")
    for _, a in alerts[alerts["item_id"] == sel].iterrows():
        ax.axvspan(pd.to_datetime(a["ts_start"], unit="s"),
                   pd.to_datetime(a["ts_end"] + config.PERIOD_S, unit="s"),
                   color="red", alpha=0.20)
    ax.set_ylabel("mid price (gp)")
    ax.set_title(f"{_label(sel)} — red bands are GE-Sentinel alert episodes")
    fig.autofmt_xdate()
    st.pyplot(fig, clear_figure=True)

# --- alerts table + memo viewer ------------------------------------------------
left, right = st.columns([3, 2])
with left:
    st.subheader("Alert queue (highest score first)")
    show = alerts[["id", "item_id", "score", "kind_guess", "ts_start", "ts_end"]].copy()
    show["start"] = pd.to_datetime(show.pop("ts_start"), unit="s")
    show["end"] = pd.to_datetime(show.pop("ts_end"), unit="s")
    show["score"] = show["score"].round(2)
    st.dataframe(show, use_container_width=True, height=380)

with right:
    st.subheader("Analyst memos")
    memo_files = sorted(config.MEMO_DIR.glob("alert_*.md"))
    if memo_files:
        pick = st.selectbox("Memo", [m.name for m in memo_files])
        st.markdown((config.MEMO_DIR / pick).read_text(encoding="utf-8"))
    else:
        st.info("No memos generated yet.")

# --- data quality ---------------------------------------------------------------
st.subheader("Data quality — real seed item (Abyssal whip)")
gaps = gap_report(engine, 4151)
if len(gaps):
    g = gaps.copy()
    g["gap_start"] = pd.to_datetime(g["gap_start"], unit="s")
    g["gap_end"] = pd.to_datetime(g["gap_end"], unit="s")
    st.dataframe(g, use_container_width=True)
    st.caption("Missing 5-minute buckets in the genuine API excerpt — handled "
               "by forward-fill limits in feature engineering, surfaced here "
               "instead of silently papered over.")
else:
    st.write("No gaps detected.")
