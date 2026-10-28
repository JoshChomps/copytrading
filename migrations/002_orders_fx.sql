-- Phase 8 groundwork + audit fixes:
--   * idx_trades_dedup: the cross-source dedup probe in insert_trade was a
--     full-table scan per insert; this index makes backfills O(log n)
--   * orders: paper-trading ledger. Fills follow THE rule — at the first
--     close AFTER the order timestamp, never same-day, exactly like the
--     backtest, so paper results stay comparable
--   * fx_rates: daily FX (USD/CAD first) for CAD-denominated reporting and
--     honest FX-cost accounting for a Canadian investor

CREATE INDEX IF NOT EXISTS idx_trades_dedup
  ON trades(filer_id, ticker, side, transaction_date, amount_low);

CREATE TABLE IF NOT EXISTS orders (
  order_id INTEGER PRIMARY KEY,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  trade_id INTEGER REFERENCES trades(trade_id),
  broker TEXT NOT NULL DEFAULT 'paper',          -- paper | alpaca_paper
  account TEXT NOT NULL DEFAULT 'paper',         -- paper | tfsa | margin
  ticker TEXT NOT NULL,
  side TEXT NOT NULL CHECK(side IN ('buy','sell')),
  notional REAL,
  qty REAL,
  status TEXT NOT NULL DEFAULT 'new' CHECK(status IN ('new','filled','rejected','cancelled')),
  fill_price REAL,
  fill_date DATE,
  ext_ref TEXT,                                  -- broker-side order id
  note TEXT,
  UNIQUE(trade_id, account)                      -- one order per signal per account
);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status, created_at);

CREATE TABLE IF NOT EXISTS fx_rates (
  date DATE NOT NULL,
  pair TEXT NOT NULL,                            -- e.g. 'USDCAD'
  rate REAL NOT NULL,
  source TEXT,
  PRIMARY KEY(date, pair)
);
