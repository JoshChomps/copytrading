-- Record sector-lookup ATTEMPTS, not just successes.
--
-- `sector_map` only ever held tickers yfinance could classify, so a ticker it
-- failed on stayed "unmapped" and was re-requested on every subsequent run.
-- The docstring claimed the map "is never re-fetched"; in practice a run that
-- mapped 521 of 800 spent the next run re-asking for the same 279 failures,
-- yfinance throttled, and every later pass reported `mapped: 0` — which reads
-- as "nothing left to map" when it means "the provider refused".
--
-- Failures are recorded here so a resumable run can skip a ticker it has
-- already asked about recently, while still allowing a retry later: a delisted
-- name is permanently unclassifiable, but a throttled one is not.

CREATE TABLE IF NOT EXISTS sector_attempts (
  ticker TEXT PRIMARY KEY,
  attempts INTEGER NOT NULL DEFAULT 0,
  last_attempt TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_status TEXT                    -- 'no_sector' | 'error' | 'throttled'
);
CREATE INDEX IF NOT EXISTS idx_sector_attempts_last ON sector_attempts(last_attempt);
