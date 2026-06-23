"""API smoke tests against the session demo database."""
from __future__ import annotations


def _client(demo_env):
    # Import AFTER demo_env has repointed config at the sandbox DB.
    import importlib

    import api.main as api_main
    importlib.reload(api_main)
    from fastapi.testclient import TestClient

    return TestClient(api_main.app)


def test_health_alerts_report_memos(demo_env):
    client = _client(demo_env)

    h = client.get("/health").json()
    assert h["status"] == "ok" and h["alerts"] > 0 and h["price_rows"] > 0

    alerts = client.get("/alerts", params={"limit": 5}).json()
    assert 1 <= len(alerts) <= 5
    assert {"id", "item_id", "score", "evidence"} <= set(alerts[0])

    one = client.get(f"/alerts/{alerts[0]['id']}").json()
    assert one["id"] == alerts[0]["id"]

    prices = client.get("/items/4151/prices", params={"limit": 50}).json()
    assert len(prices) > 0 and prices[0]["source"] == "seed_real"

    rep = client.get("/report").json()
    assert "precision_at_k" in rep

    memos = client.get("/memos").json()["memos"]
    assert memos
    md = client.get(f"/memos/{memos[0]}").json()
    assert md["markdown"].startswith("#")

    assert client.get("/memos/..%2Fsecret.md").status_code in (400, 404)
    assert client.get("/alerts/999999").status_code == 404
