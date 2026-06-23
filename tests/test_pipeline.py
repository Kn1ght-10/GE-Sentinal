"""End-to-end pipeline test on the fast synthetic market."""
from __future__ import annotations

import sqlalchemy as sa


def test_demo_metrics_and_artifacts(demo_env):
    rep = demo_env["result"]["report"]
    cfg = demo_env["config"]

    # Detection quality floors on injected ground truth (held-out window).
    assert rep["precision_at_k"] >= 0.6, rep
    assert rep["recall_manip_events"] >= 0.5, rep
    assert rep["row_level_auc"] is None or rep["row_level_auc"] > 0.8, rep
    assert rep["n_manip_events_test"] >= 2

    # Artifacts actually exist where the README says they do.
    assert (cfg.OUTPUT_DIR / "eval_report.json").exists()
    assert len(list(cfg.MEMO_DIR.glob("alert_*.md"))) >= 1
    assert (cfg.ASSETS_DIR / "fig_eval_summary.png").exists()
    assert (cfg.ASSETS_DIR / "fig_real_whip.png").exists()


def test_alerts_persisted_with_evidence(demo_env):
    cfg = demo_env["config"]
    from ge_sentinel import db

    engine = db.get_engine(cfg.DB_URL)
    with engine.connect() as cx:
        n = cx.execute(sa.select(sa.func.count()).select_from(db.alerts)).scalar_one()
        top = cx.execute(sa.select(db.alerts).order_by(db.alerts.c.score.desc())).first()
    assert n > 0
    assert top.score >= 3.0
    assert top.evidence_json and top.evidence_json != "{}"


def test_real_seed_loaded_and_isolated(demo_env):
    """Real rows flow through the system but never enter the eval set."""
    cfg = demo_env["config"]
    import pandas as pd

    from ge_sentinel import db

    engine = db.get_engine(cfg.DB_URL)
    with engine.connect() as cx:
        src = pd.read_sql(sa.select(db.prices_5m.c.source).distinct(), cx)["source"].tolist()
    assert "seed_real" in src and "synth" in src
    for e in demo_env["result"]["episodes"]:
        assert e["item_id"] >= 900_000, "evaluated episodes must be synthetic-only"
