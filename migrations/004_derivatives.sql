-- Derivatives & futures section (leveraged bets by powerful people).
--
-- Options ride in the existing `trades` table via the `instrument` column
-- ('call_option' | 'put_option' | 'option'), so no new trade schema is
-- needed — only an index to pull them fast. Futures are different: no named
-- individual discloses a futures position, so the honest proxy is the CFTC
-- Commitments of Traders report — aggregate positioning by trader CATEGORY
-- (commercials / large speculators / non-reportables) per market.

CREATE INDEX IF NOT EXISTS idx_trades_instrument ON trades(instrument);

CREATE TABLE IF NOT EXISTS cot_positioning (
  report_date DATE NOT NULL,          -- Tuesday of the report week
  market TEXT NOT NULL,               -- e.g. 'E-MINI S&P 500'
  market_code TEXT,                   -- CFTC contract market code
  noncomm_long REAL,                  -- large speculators (funds/CTAs) long
  noncomm_short REAL,
  comm_long REAL,                     -- commercials (hedgers)
  comm_short REAL,
  nonrept_long REAL,                  -- small traders
  nonrept_short REAL,
  open_interest REAL,
  source TEXT DEFAULT 'cftc_cot',
  PRIMARY KEY(report_date, market)
);
CREATE INDEX IF NOT EXISTS idx_cot_market ON cot_positioning(market, report_date DESC);
