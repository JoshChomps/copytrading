-- Ticker -> sector/industry map, populated from yfinance and cached in the DB
-- (persists across GitHub Actions runs via the data cache). Enables sector /
-- area attribution: what areas filers pursue and where their picks actually
-- worked. Unmapped tickers fall back to 'Unknown' in the analysis, never
-- guessed.
CREATE TABLE IF NOT EXISTS sector_map (
  ticker TEXT PRIMARY KEY,
  sector TEXT,
  industry TEXT,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
