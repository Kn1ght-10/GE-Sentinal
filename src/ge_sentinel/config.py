"""Central configuration for GE-Sentinel.

Everything tunable lives here or in environment variables so the same code
runs locally (SQLite) and in production (Postgres/TimescaleDB) unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repo root = parent of src/
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SEED_REAL_DIR = DATA_DIR / "seed" / "real"
CORPUS_DIR = DATA_DIR / "corpus"
OUTPUT_DIR = DATA_DIR / "outputs"
MEMO_DIR = OUTPUT_DIR / "memos"
ASSETS_DIR = ROOT / "assets"

for _d in (OUTPUT_DIR, MEMO_DIR, ASSETS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Database -----------------------------------------------------------
# Default: local SQLite file. Set GE_SENTINEL_DB to a Postgres/Timescale URL
# (e.g. postgresql+psycopg://user:pass@host/ge) to switch backends.
DB_URL = os.environ.get("GE_SENTINEL_DB", f"sqlite:///{DATA_DIR / 'ge_sentinel.db'}")

# --- OSRS Wiki real-time prices API --------------------------------------
API_BASE = "https://prices.runescape.wiki/api/v1/osrs"
# The wiki's acceptable-use policy REQUIRES a descriptive User-Agent and
# pre-emptively blocks defaults like python-requests. Put your contact in env.
USER_AGENT = os.environ.get(
    "GE_SENTINEL_UA",
    "ge-sentinel market-integrity research - set GE_SENTINEL_UA to your contact",
)

# --- Time grid ------------------------------------------------------------
PERIOD_S = 300              # 5-minute buckets
BUCKETS_PER_DAY = 86400 // PERIOD_S  # 288
BASELINE_W = BUCKETS_PER_DAY          # rolling baseline window (1 day)

# --- Detection ------------------------------------------------------------
ALERT_TAU = float(os.environ.get("GE_SENTINEL_TAU", "3.0"))
EPISODE_GAP = 3             # buckets of silence allowed inside one episode
PATCH_SUPPRESS_W = 6        # buckets around an announced update to down-weight
PATCH_SUPPRESS_FACTOR = 0.3

# --- Optional LLM memo agent ----------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
