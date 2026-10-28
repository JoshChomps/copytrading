-- 13F option positions: a manager can hold the SAME cusip as stock AND puts
-- AND calls in one filing (Burry's classic 'stock + protective puts'), so
-- put_call must be part of the holdings key. Recreate the table with it in
-- the PK (13F backfill has not populated this table in any run yet, so there
-- is nothing to migrate).

DROP TABLE IF EXISTS holdings_13f;
CREATE TABLE holdings_13f (
  filer_id INTEGER NOT NULL REFERENCES filers(filer_id),
  period DATE NOT NULL,
  cusip TEXT NOT NULL,
  put_call TEXT NOT NULL DEFAULT '',   -- '' | 'Put' | 'Call'
  value REAL,
  shares REAL,
  PRIMARY KEY(filer_id, period, cusip, put_call)
);
