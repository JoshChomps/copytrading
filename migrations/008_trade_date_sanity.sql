-- Null out impossible transaction dates.
--
-- A PTR discloses a trade that has ALREADY happened, so a transaction_date
-- after its own filing_datetime is a parse error, never data. The House Clerk
-- PDFs carry a handful: 3031-04-30, 2220-04-07, 2202-09-19, 2035-09-01,
-- 2030-06-01 and 2027-05-31 all reached the DB this way, plus ~78 ordinary
-- day/month typos that land a few days past the filing.
--
-- The TRADE is still valid and still copyable: `filing_datetime` is the
-- point-in-time execution anchor and it is unaffected. Only the transaction
-- date is unusable, so it is set to NULL rather than deleting the row --
-- nulling costs the trade its entry in the disclosure-lag and zero-lag decay
-- statistics (both of which measure filing minus transaction) while keeping it
-- in every backtest, which is the correct trade-off. Deleting would silently
-- shrink the sample the headline results rest on.
--
-- Long POSITIVE lags are deliberately left alone: 603 trades are disclosed
-- more than two years late and those are genuine amended/late filings, not
-- errors. Lateness is the phenomenon this project measures, not a bug.

UPDATE trades
   SET transaction_date = NULL
 WHERE transaction_date IS NOT NULL
   AND filing_datetime IS NOT NULL
   AND date(transaction_date) > date(filing_datetime);
