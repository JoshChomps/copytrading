-- Company-news context store (alerts + dashboard, NOT a backtest factor —
-- see copytrade/ingest/news.py for why). Provenance per row.

CREATE TABLE IF NOT EXISTS news (
  uid TEXT PRIMARY KEY,               -- sha1(ticker|url)
  ticker TEXT NOT NULL,
  headline TEXT NOT NULL,
  url TEXT NOT NULL,
  published_at TIMESTAMP NOT NULL,
  source TEXT,                        -- outlet/domain
  provider TEXT NOT NULL,             -- finnhub | gdelt
  fetched_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_news_ticker ON news(ticker, published_at DESC);
