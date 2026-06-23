-- Production schema: PostgreSQL + TimescaleDB.
-- SQLite dev schema is created automatically by SQLAlchemy (db.py); this file
-- is the production equivalent with a hypertable on the price series.

CREATE TABLE IF NOT EXISTS items (
  id        INTEGER PRIMARY KEY,
  name      TEXT NOT NULL,
  members   BOOLEAN,
  ge_limit  INTEGER,
  highalch  INTEGER,
  source    TEXT NOT NULL DEFAULT 'unknown'
);

CREATE TABLE IF NOT EXISTS prices_5m (
  item_id   INTEGER NOT NULL,
  ts        INTEGER NOT NULL,            -- unix seconds, bucket start
  avg_high  DOUBLE PRECISION,
  avg_low   DOUBLE PRECISION,
  high_vol  INTEGER NOT NULL DEFAULT 0,
  low_vol   INTEGER NOT NULL DEFAULT 0,
  source    TEXT NOT NULL DEFAULT 'api',
  PRIMARY KEY (item_id, ts)
);

-- Integer-time hypertable, 1-day chunks. (For retention/continuous aggregates
-- on integer time, also register set_integer_now_func per Timescale docs.)
SELECT create_hypertable('prices_5m', 'ts',
                         chunk_time_interval => 86400,
                         if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_prices_ts ON prices_5m (ts);

CREATE TABLE IF NOT EXISTS truth_events (
  id        SERIAL PRIMARY KEY,
  item_id   INTEGER NOT NULL,
  kind      TEXT NOT NULL,
  start_ts  INTEGER NOT NULL,
  end_ts    INTEGER NOT NULL,
  magnitude DOUBLE PRECISION,
  note      TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
  id            SERIAL PRIMARY KEY,
  item_id       INTEGER NOT NULL,
  ts_start      INTEGER NOT NULL,
  ts_end        INTEGER NOT NULL,
  score         DOUBLE PRECISION NOT NULL,
  kind_guess    TEXT,
  evidence_json TEXT,
  created_ts    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_score ON alerts (score DESC);
