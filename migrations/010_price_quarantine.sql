-- Quarantine for price series that are positive but not credible.
--
-- The earlier fix (see store.db.upsert_prices) rejected adj_close <= 0 after
-- four foreign OTC ADRs produced a -587,200% return. That fix was real but
-- PARTIAL: it removed the negative values and left the rest of the same
-- corrupted series in place. The residue is still in the data --
--
--   AOZOF   min 3.7e-24, max 14.02          range 3.8e24
--   AEXAY   max 4.1e15                      range 4.1e18
--   DAIUF   min 5.1e-07, 8 sessions >+100%  range 8.8e7
--   ITC     oscillates 45 <-> 7,800, 312 sessions >+100%
--   BM      oscillates 0.07 <-> 7.2, 116 sessions >+100%
--   CHK     3.05 (2021-03-01) -> 81.46 (2024-10-04) read as one session
--
-- -- and it is not cosmetic: DAIUF was the single best-performing name on the
-- dashboard at +251% mean over 8 buys.
--
-- Rows are NOT deleted. `prices` is re-populated from providers, so a delete
-- would be undone by the next refresh, and deleting also destroys the evidence
-- that the provider is wrong. Instead the ticker is quarantined and excluded at
-- `load_price_matrix`, the one place every analysis reads prices from. Its
-- trades then fall into the existing "unpriceable" bucket, which is already
-- counted and disclosed in every report.

CREATE TABLE IF NOT EXISTS price_quarantine (
  ticker TEXT PRIMARY KEY,
  reason TEXT NOT NULL,           -- which screen fired
  detail TEXT,                    -- the measured numbers, so it can be re-judged
  detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  released INTEGER NOT NULL DEFAULT 0   -- manual override: 1 = trust it anyway
);
CREATE INDEX IF NOT EXISTS idx_price_quarantine_released ON price_quarantine(released);
