-- Congressional committee assignments and jurisdictions.
--
-- Motivation: the one untested version of "does the disclosure carry
-- information" that is NOT already ruled out. The academic claim (Ziobrowski
-- et al.) is that members trade better inside their committee's jurisdiction —
-- a defence-committee member buying defence stocks. Everything else this
-- project has tested treats all filers alike.
--
-- IMPORTANT LIMITATION, ENCODED IN THE SCHEMA. The upstream source
-- (unitedstates/congress-legislators) publishes CURRENT membership only; there
-- is no historical membership file. Applying a 2026 committee roster to a 2016
-- trade is look-ahead of exactly the kind the project's first invariant
-- forbids. `as_of` therefore records the snapshot date on every row, and any
-- return test built on this MUST restrict itself to trades filed while the
-- snapshot was in force (i.e. the current Congress). The column exists so that
-- restriction is enforceable rather than remembered.

CREATE TABLE IF NOT EXISTS committees (
  thomas_id     TEXT PRIMARY KEY,      -- e.g. HSAG, SSAF; subcommittees are parent+code
  type          TEXT,                  -- house | senate | joint
  name          TEXT,
  jurisdiction  TEXT,                  -- upstream prose blurb, where published
  parent_id     TEXT,                  -- set for subcommittees
  as_of         DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS committee_members (
  thomas_id  TEXT NOT NULL,
  bioguide   TEXT NOT NULL,            -- joins to filers.meta ->> 'bioguide'
  name       TEXT,
  party      TEXT,                     -- majority | minority
  rank       INTEGER,                  -- 1 = chair / ranking member
  title      TEXT,                     -- Chairman, Ranking Member, ...
  as_of      DATE NOT NULL,
  PRIMARY KEY (thomas_id, bioguide, as_of)
);

CREATE INDEX IF NOT EXISTS idx_cm_bioguide ON committee_members(bioguide);
CREATE INDEX IF NOT EXISTS idx_cm_asof ON committee_members(as_of);
