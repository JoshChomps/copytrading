# Ingest parser audit: 2026-08-16

Closes the largest item in the project log §17.4: *"The other ingest sources
beyond the date-format grep ... The House parser has now yielded three bugs of
the same family; nothing says the others were luckier, only that nobody has
looked."*

**Method.** Read every module in `copytrade/ingest/` against the three shapes
that produced §7.1, §7.7 and §7.8:

1. **a layout/format assumption fitted to one vintage** and never checked
   against another (§7.7: `\d{2}` vs `05/5/2015`);
2. **a guard placed somewhere other than the choke point** (§2.2);
3. **a failure that reports as success**, a status reading "done" that means
   "the provider is refusing" (§7.4, §7.8, §7.10).

Every claim below was then tested against the live database or the real
upstream files rather than reasoned about. Claims I could not test are marked
**untested**.

**The framing that governs severity.** Only two sources are populated:

| source | rows | span |
|---|---|---|
| `house_clerk` | 33,464 | 2015-01-03 → 2026-08-12 |
| `senate_watcher` | 6,317 | 2014-01-31 → 2021-03-09 |

`form4`, `form345_bulk`, `thirteenf`, `thirteend`, `cftc`, `quiver` and `news`
have **zero rows**. Their defects are **latent**, real, but they cannot have
touched a published number. Ranking below is by *live impact*, not by how bad
the code looks.

---

## A. Live findings: these touch published numbers

### A1. The health panel published an impossible aggregate, FIXED

**The dashboard's first panel**, whose own docstring says *"every other number
is worthless if the underlying sample is stale or thin"*, reported:

```
traded_tickers 3335 · priced_tickers 2990 · missing_prices 631 · coverage 89.7% · verdict "ok"
```

`2,990 + 631 = 3,621` against a universe of `3,335`. §2.8, an impossible
aggregate is a data bug, never a finding.

**Cause.** `build_health` counted two different populations
(`web/analyses.py:235`):

```python
traded  = COUNT(DISTINCT ticker) FROM trades WHERE instrument='stock'   # 3,335
priced  = COUNT(DISTINCT ticker) FROM prices                            # 2,990  <-- whole table
missing = <traded universe> EXCEPT <prices>                             # 631    <-- correct
```

`prices` also holds the seven benchmarks and tickers only ever traded as
options: 286 names that are not in the `instrument='stock'` denominator. So
`priced` and `missing` answered different questions and could not reconcile.

**It was not cosmetic.** The inflated ratio cleared the `cov < 0.85` gate
immediately below it, so the panel reported **verdict "ok" with zero notes at a
coverage level it exists to flag**. That is §7.4's failure mode exactly. True
coverage is **81.1% (2,704 of 3,335)**, not 89.7%.

**Fix.** `priced` is now the INTERSECTION with the traded universe, so
`priced + missing == traded` by construction. After rebuild the panel reads
`verdict: degraded` and emits *"only 81% of traded tickers are priced"*.

**Test.** `test_health_price_coverage_counts_the_traded_universe_only`
(`tests/test_dashboard.py`). Checked by reintroducing the bug: fails with
`assert 4 == 2`. The 631 figure published alongside it was always correct, it
is the 2,990 and the 89.7% that were wrong, in the project log §9 and §17.5.

### A2. Amended House PTRs are stored as duplicate trades, NOT fixed

`CongressTrade.source_uid` hashes a key that **includes `filing_date`**
(`ingest/congress.py:46`). The `uid` override exists precisely to exclude it
its comment says *"amended PTRs re-report the same trade with a later receipt
date; the earliest wins"*, but **only `senate_watcher` passes `uid`**.
`house_clerk` does not, so an amended House PTR re-reporting a transaction
creates a second row.

Measured on the live DB, grouping on
`(filer, ticker, side, transaction_date, amount_low, amount_high)` and
requiring more than one distinct filing date:

| source | dup groups | rows involved | surplus rows |
|---|---|---|---|
| `house_clerk` | 323 | 656 | **333** |
| `senate_watcher` (passes `uid`) | 0 | - | **0** |

The senate row is the control: the one source that uses the override has
**zero**, which is what the mechanism predicts.

The pattern is unmistakably amendments, not genuine repeat trades, Patrick
Fallon has eight tickers all dated 2021-10-04, each filed three times on the
same three dates (2021-10-05, 2021-11-04, 2022-05-14).

**Impact is real but small: 90 surplus buys of 18,936 (0.48%)**, 186 surplus
sells. This will not flip a sign in §5 and is not a reason to distrust the
headline. It slightly overweights amended trades in the per-trade means and
adds a spurious later-dated entry for each.

**Why it is not fixed here.** The one-line fix (give `house_clerk` a
filing-date-free `uid`) only changes *ingest*; the 333 rows already stored come
out on a `--reparse`, which is a ~30-minute full House re-scrape plus a
dashboard rebuild, and it moves published numbers. Owner's call, like §7.5.

### A3. `refresh_prices` uses two different thresholds for "priced"

`ingest/market.py:301` treats a ticker as already-held only with **>50** price
rows, but `:321` computes `unpriceable` against **≥1** row. **43 traded tickers
sit in the gap**: re-requested from providers on every single run (they never
reach 50 rows) *and* excluded from the `unpriceable` count that reports the
coverage gap. Invisible in both directions, the §7.4 shape, one layer down.

Examples: `AMEH`, `BITA`, `BRCM`, `BRKB`, `COMM`, `GTLS`, `FDRXX`.

### A4. Two counters that do not mean what they say

- `store_congress_trades` counts a **nameless trade as `deduped`**
  (`ingest/congress.py:204`). A dropped row is reported as a successful
  de-duplication. Currently 0 rows affected, but the counter is the thing you
  would read to notice.
- `house_clerk` builds a `raw` dict (type code + asset text) for every
  transaction and **`store_congress_trades` never passes it to
  `insert_trade`**. Verified: **0 of 33,464** House rows have `raw_ref` or
  `tx_code` stored. A parse cannot be re-audited without re-fetching the PDF
  which is exactly what §7.1, §7.7 and §7.8 each required.

---

## B. Latent findings: real defects in unpopulated sources

Ordered by what would bite first if the source were switched on.

### B1. `thirteend.py`: §7.10's headline defect, unfixed

§7.10 fixed this *in `house_clerk` only*. `refresh_13d` still does it:

```python
if r.status_code != 200:  continue     # counted nowhere
except httpx.HTTPError:   continue     # counted nowhere
```

A blanket 403/429 across every quarter returns
`{"filings_seen": 0, "inserted": 0, ...}`, a tidy, successful-looking run that
read nothing. There is no `degraded` field and no failure counter. This is the
same defect, in the same words, that §7.10 describes as *"the exact outage it
exists for"*.

### B2. `quiver.py`: bypasses the ticker choke point

- Uses `(r.get("Ticker") or "").strip().upper()` instead of `clean_ticker`,
  which every other congress path uses. `clean_ticker`'s docstring exists
  because real filings carry option symbols like `SPY160219P00180000` that
  *"previously slipped through as raw HTML and polluted the price fetch"*.
  A §2.2 violation: the guard is at four of five call sites.
- Hardcodes `instrument="stock"`, so Quiver-sourced options would enter the
  equity sample as shares.
- `chamber = "Senate" if "senate" in (r.get("House") or "").lower() else
  "House"`: a **missing** field silently yields "House".

### B3. `thirteenf.py`: a scale change the docstring names and the code ignores

`Holding.value` is documented as *"dollars (modern filings report USD, older
ones $000s)"* and **nothing ever normalises it**. 13F value reporting changed
from thousands to whole dollars, so holdings either side of that boundary
differ by 1000×, and `quarterly_deltas` feeds the unconverted number straight
into `amount_low`/`amount_high`. This is §7.7's shape with the excuse removed:
the vintage difference is written down in the code and still not handled.

Also: `backfill_13f_cik` reads only `filings.recent` from the submissions JSON
(older filings live in `filings.files` and are invisible, silently capping
`quarters`); a mid-series XML fetch failure makes `zip(loaded, loaded[1:])`
compute a delta *across* the gap and attribute it to one quarter; and
`openfigi_cusip_to_ticker` `break`s on any non-200 while the caller reports
`cusips_newly_mapped: len(unmapped)`, the number **requested**, not mapped.

### B4. `form4.py` / `form345_bulk.py`: derivatives dropped, and a silent-zero path

- Both parse **only** the non-derivative table (`nonDerivativeTable` /
  `NONDERIV_TRANS.tsv`). `derivativeTable` / `DERIV_TRANS.tsv` are never read,
  so insider option activity is invisible, worth knowing alongside §17.5's
  "options never priced", since it is the same blind spot on a second channel.
- `ingest_zip` raises if `SUBMISSION.tsv` or `NONDERIV_TRANS.tsv` is missing but
  **tolerates a missing `REPORTINGOWNER.tsv`**, in which case `owner` is `None`
  for every row, every row is skipped, and the function returns a normal
  success dict with `inserted: 0`.
- `source_uid` falls back to the row counter `n` when `NONDERIV_TRANS_SK` is
  absent; `n` depends on `limit`, so two runs at different limits would produce
  different uids for the same row.

### B5. `senate_efd.py`: a dead parameter and an unpaginated cap

- `scrape_recent_ptrs(conn, settings, days: int = 14)` **never uses `days`**.
  It always queries from the 1st of the current month, so on the 1st the window
  is one day.
- `"start": "0", "length": "100"` with no pagination: filings past the first
  100 in a month are silently dropped.
- Electronic reports are never actually parsed (`res.skipped += 1`), which the
  module states honestly. `parse_efd_report_html` exists but is unreachable from
  the scrape path and its column positions are **untested** against a real
  document.

### B6. `cftc.py`: a two-digit year that assumes the 21st century

`_iso` maps a 6-digit `YYMMDD` to `f"20{s[0:2]}-..."`. The legacy COT archive
starts in **1986**, so pre-2000 rows would parse as 2086-2099. Mitigated in
practice because the ISO column (`rec[2]`) is tried first. `LATEST_CSV` also
hardcodes `deacot2026.zip`, which will 404 in 2027, loudly, via
`raise_for_status`, so that one is fine.

### B7. `market.py`: `provider_symbol` collisions (benign today)

`fetch_yfinance` builds `{provider_symbol(t): t for t in tickers}`. Where two
disclosed spellings normalise to one provider symbol, the later silently wins.
**Four real collisions in the traded universe**: `BRK.B`/`BRK-B`,
`RDS.A`/`RDS-A`, `RDS.B`/`RDS-B`, `LGF.B`/`LGF-B`.

Verified harmless *as currently called*: `BRK.B` and `BRK-B` both hold 3,157
rows, because `_fetch_in_chunks` retries what is still missing and the loser is
alone in the retry batch. The collision is masked by the retry loop, not
absent: a caller using `fetch_daily_closes` directly would lose one spelling.

---

## C. Corrections to the record

**§17.2's grep claim was incomplete.** It states: *"The only other `\d{2}` in
`ingest/` is an SEC accession number."* There is a second one
`senate_watcher.py:156`:

```python
m = re.search(r"for_(\d{2})_(\d{2})_(\d{4})", path.name)
```

It parses the date out of `transaction_report_for_MM_DD_YYYY.json` to sort files
chronologically, and chronological order is what makes the amended-PTR dedup
keep the **earliest** receipt. An unpadded filename falls back to
`("9999","99","99")` and sorts last, which would invert that.

**Tested, not assumed:** all **1,055** report filenames in the local clone are
uniformly padded (every name is 38 characters), so this has never fired, and the
source froze in March 2021 so it never will. **§17.2's conclusion holds; its
evidence did not.** The distinction matters because the grep was the *only*
thing standing behind "the regex gate was unique to `house_clerk`".

---

## D. Environment drift found while running the suite

The documented baseline, *"360 tests pass"*, **does not currently reproduce**.
Three `requirements.txt` entries are not installed:

| missing | consequence |
|---|---|
| **`pdfplumber==0.11.4`** | 7 `test_scrape.py` failures. Any `scrape-house` run raises, the §7.8 guard working as designed. |
| **`yfinance==1.5.1`** | **The primary price provider is absent.** `fetch_daily_closes` catches this broadly and records `yfinance_error` in provenance, so a `prices` run would *appear* to work while serving only Tiingo (hourly-throttled) and stooq (split-adjusted only). Directly relevant to the "re-run `prices`" work item. |
| **`matplotlib==3.11.0`** | `tests/test_report_readme.py` cannot be collected. |

353 pass + 7 pdfplumber failures = **exactly the documented 360**, so the suite
itself is intact; only the environment drifted.

**Separately, collection was failing for five modules** for an unrelated reason.
Four installed packages (`py_clob_client`, `py_order_utils`,
`py_builder_signing_sdk`, `pandas-datareader`) each ship a top-level
`tests/__init__.py` into site-packages. A **regular** package beats an implicit
namespace package no matter what sys.path order says, so
`from tests.conftest import FIXTURES` resolved to the Polymarket client's test
package. Fixed by adding `tests/__init__.py` to this repo, which pins
resolution here regardless of what else shares the interpreter.

---

## E. What this audit did NOT check

- **`parse_efd_report_html`, `parse_cot_csv`, `parse_13f_infotable` and the
  DERA TSV reader were never run against a real document.** Their column and
  field assumptions are read, not verified, the same standing that §7.1 had
  before someone checked a Clerk PDF.
- **`committees.py`, `legislators.py`, `sectors.py`, `canada_ciec.py`,
  `famafrench.py`** were not audited. `committees.py` (316 lines) is the
  largest unaudited ingest module and it feeds §6.
- **No fix was made to any latent finding in section B.** They are recorded,
  not repaired.
- The `raw`/`raw_ref` provenance gap (A4) was confirmed for `house_clerk`; the
  other sources' provenance handling was not surveyed.
- Whether the 333 duplicate rows in A2 change any *panel* number was not
  computed: only their share of the sample (0.48% of buys).
