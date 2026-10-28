-- Unified schema (project plan §3, with small documented extensions:
--   * UNIQUE(name, kind) on filers so upserts are idempotent
--   * tx_code column on trades so non-P/S Form 4 codes are stored but not sided
--   * poll_state for graceful poller resume, cusip_map for 13F, holdings_13f for cloning)

CREATE TABLE IF NOT EXISTS filers (
  filer_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN ('congress','insider','fund')),
  meta JSON,
  UNIQUE(name, kind)
);

CREATE TABLE IF NOT EXISTS filers_aliases (
  alias TEXT PRIMARY KEY,
  canonical TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
  trade_id INTEGER PRIMARY KEY,
  filer_id INTEGER NOT NULL REFERENCES filers(filer_id),
  source TEXT NOT NULL,
  source_uid TEXT NOT NULL,
  ticker TEXT,
  cusip TEXT,
  side TEXT CHECK(side IN ('buy','sell') OR side IS NULL),
  tx_code TEXT,
  instrument TEXT DEFAULT 'stock',
  transaction_date DATE,
  filing_datetime TIMESTAMP NOT NULL,
  amount_low REAL,
  amount_high REAL,
  shares REAL,
  price REAL,
  raw_ref TEXT,
  UNIQUE(source, source_uid)
);
CREATE INDEX IF NOT EXISTS idx_trades_filer ON trades(filer_id, filing_datetime);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);

CREATE TABLE IF NOT EXISTS prices (
  ticker TEXT NOT NULL,
  date DATE NOT NULL,
  adj_close REAL NOT NULL,
  PRIMARY KEY(ticker, date)
);

CREATE TABLE IF NOT EXISTS poll_state (
  key TEXT PRIMARY KEY,
  value TEXT
);

CREATE TABLE IF NOT EXISTS cusip_map (
  cusip TEXT PRIMARY KEY,
  ticker TEXT
);

CREATE TABLE IF NOT EXISTS holdings_13f (
  filer_id INTEGER NOT NULL REFERENCES filers(filer_id),
  period DATE NOT NULL,
  cusip TEXT NOT NULL,
  value REAL,
  shares REAL,
  PRIMARY KEY(filer_id, period, cusip)
);
