# Scraping architecture: constantly harvesting disclosures

This is the "constantly scrape US + Canadian politicians and other
significant traders" engine, and the reasoning behind its shape. TL;DR: a
**registry of sources** driven by **one scheduler**, with **authoritative
source-of-record feeds as the backbone** and aggregators (Finnhub) as
enrichment: never the other way round.

## The architecture

```
copytrade/scrape/
  base.py        Source protocol + ScrapeResult (the contract every feed meets)
  sources.py     one class per feed, each wrapping an ingester
  registry.py    which sources are enabled ("auto" = prereqs met) + watchlist
  scheduler.py   the loop: per-source cadence, backoff, alerts, heartbeats
```

A **Source** does one thing: `run_once(conn, settings) -> ScrapeResult`,
returning only *new* normalized trades since it last ran. Everything
cross-cutting lives in the scheduler, once:

- **cadence** per source (Form 4 every 3 min; House/Senate every 6 h; Canada
  daily; Finnhub hourly);
- **dedup** at the DB layer (`UNIQUE(source, source_uid)`) so re-runs never
  double-insert: sources need no dedup logic;
- **isolation + backoff**: a source that throws is caught, backed off
  exponentially, and disabled after 8 straight failures, it can never take
  the loop down (proven in `tests/test_scrape.py`, and observed live: with
  every host blocked, each source logged its own 403 and the others kept
  going);
- **heartbeats**: `scrape:<name>:heartbeat` in `poll_state`, surfaced by
  `scrape-status` for liveness/health;
- **alerts**: new actionable trades by watchlisted (top-ranked) filers go to
  the webhook, enriched with cluster-buy flags and recent news.

Two run shapes:

- `poll`: long-lived process, one asyncio task per source at its cadence.
- `scrape-once`: every source once, then exit. **Preferred on a small
  Vancouver VPS**: drive it from cron (e.g. Form 4 every few minutes during
  market hours PT, the disclosure feeds a few times a day) so a crash can't
  wedge a daemon and the box does nothing between ticks.

Adding "other significant ones" = add a class to `sources.py` and list it in
`ALL_SOURCES`. That's the whole extension surface.

## Why authoritative feeds are the backbone (the Finnhub blind spot)

You asked to double-check that "Finnhub + house scrapers" is a good
architecture. It is: **with one correction that's baked into the code**:

> **Finnhub's congressional endpoint is symbol-keyed.** You query it
> per-ticker, so it can only return trades in symbols you already ask about.
> If you rely on it as the primary feed, any trade in a ticker outside your
> universe is *invisible*, a silent blind spot that quietly defines your
> universe by your watchlist.

So each source declares `authoritative: bool`:

| source | `authoritative` | coverage | why |
|---|---|---|---|
| `edgar_form4` | ✅ | every US insider filing | EDGAR Atom feed is the source of record |
| `house_clerk` | ✅ | every US House filer | the Clerk's `<YEAR>FD.xml` lists *all* filings |
| `senate_efd` | ✅ | every US Senator | efdsearch is the source of record |
| `canada_ciec` | ✅ | Canadian MPs (coarse) | the CIEC registry is the source of record |
| `finnhub_congress` | ❌ | only your watchlist symbols | symbol-keyed aggregator |

The scheduler prints the split on startup so it's never ambiguous which
feeds are actually giving you complete coverage. **Recommended posture**: run
the authoritative feeds for coverage; keep Finnhub on as a fast, normalized
cross-check on your high-conviction watchlist (and as the zero-scraping
option before you wire the House PDF parser). Both write to the same
`trades` table and dedup against each other.

## Source coverage & status

| source | reach here | off-sandbox status |
|---|---|---|
| `edgar_form4` (insiders) | ❌ 403 | **works**, no key, real acceptance timestamps |
| `house_clerk` (US House) | ❌ 403 | index parser exact & tested; **PDF text needs `pip install pdfplumber`**; scanned PTRs need OCR (flagged, not guessed) |
| `senate_efd` (US Senate live) | ❌ 403 | CSRF/agreement handshake implemented; electronic-table parse tested; **validate the handshake off-sandbox** (most likely to have drifted) |
| `senate_watcher` (US Senate history) | ✅ works | `ingest-senate`, real 2013-2021, the backtest backbone |
| `canada_ciec` (Canadian MPs) | ❌ 403 | see §Canada, coarse by law; point `canada_ciec_json_url` at a tracker export |
| `finnhub_congress` (US House+Senate) | ❌ 403 | works with a free key; symbol-keyed |
| `backfill-13f` (funds) | ❌ 403 | works, Berkshire/Scion/etc. via CIK |

Every "❌ 403" is this dev sandbox's egress policy (git + PyPI only), **not**
a code problem: the identical calls run from any normal machine. The pure
parsers are unit-tested against spec-accurate fixtures so the logic is
locked; the network glue is what needs one real run to confirm.

## §Canada: read before expecting US-style signals

Canada has **no STOCK Act**. Concretely:

- MPs file a Disclosure Summary and must report a **material change**
  (acquire/dispose > $10,000) within **60 days**, coarser and slower than a
  US PTR, with no exact amount and often no exact date.
- **Ministers and reporting public office holders are barred** from holding
  public-company shares (divest / blind-trust within 120 days), the
  highest-profile Canadians have nothing to copy by law.
- The registry is per-member HTML/PDF with no bulk API.

So `canada_ciec` stores declarations as `instrument='ca_material_change'` and
**excludes them from the priced backtest by default**, they lack the
precision the event study needs, and faking that precision is the trap this
codebase refuses everywhere else. Best Canadian data comes from third-party
trackers (e.g. integrityindex.ca) or media FOIs; point `canada_ciec_json_url`
at such an export to ingest it. Net: US insiders/Congress are the real
signal; Canadian political copying is a thin, lagged supplement, the
value of the Canadian angle in this project is the **tax/account/fee
structure** (`docs/CANADA.md`), not Canadian political trades.

## Minimizing disclosure lag (how fresh can signals actually be?)

The delay from a filer's trade to the moment you can act splits into three
parts; only the last is a software problem (`python -m copytrade lag`
measures the first, which is the biggest):

| lag component | who controls it | congress | insiders (Form 4) | our lever |
|---|---|---|---|---|
| **reporting** (trade → filing) | the filer, by law | up to **45 days** (STOCK Act); members choose | **2 business days** | none, but *prefer fast-disclosing filers* (measured by `lag`) |
| **publication** (filing → public) | the disclosure office | hours-days (House Clerk/eFD) | ~minutes (EDGAR) | source choice only |
| **pipeline** (public → we act) | us | our poll cadence | our poll cadence | **fully ours** |

Facts, [confirmed 2026](https://www.congress.gov/bill/112th-congress/senate-bill/2038):
there is **no real-time congressional feed**, the 45-day cap and the
member's discretion inside it dominate everything, and no pipeline can shrink
it. So "minimize lag" for congress means three things, in order of impact:

1. **Weight toward fast disclosers.** Some members file within days, others
   use the full 45. `python -m copytrade lag` ranks them; the
   `decay_lag_speed.png` chart shows fast filers (≤15 bd) retain materially
   more post-filing alpha. This is the only real lever, and it's large.
2. **Poll the authoritative feeds tightly.** House Clerk/eFD post as filed;
   `poll_form4_secs`/`poll_congress_secs` in config set our cadence. For a
   paid *push* (minutes faster than polling) QuiverQuant/Unusual Whales
   webhook into `subscribe`-style handlers, worth it only if you trade the
   fastest disclosers where minutes matter.
3. **Insiders are already near-optimal.** EDGAR's real-time Atom feed carries
   the acceptance timestamp within minutes; the `edgar_form4` source polls
   every 3 minutes by default and executes next-close. The 2-business-day
   legal cap makes Form 4 the structurally freshest signal in the system.

Bottom line: for insiders the lag is minutes-to-days and mostly solved; for
congress it's weeks and immovable by law, so the edge (if any) must survive
that: which is exactly what the filing-anchored backtest measures.

## Deploying the scraper in Vancouver

Cron on a small VPS (or a Pi), all times PT (market logic stays ET):

```cron
# Form 4 every 5 min during market hours (06:30-13:15 PT = 09:30-16:15 ET)
*/5 6-13 * * 1-5   cd /srv/copytrade && .venv/bin/python -m copytrade scrape-once
# disclosure feeds + report + allocation after close
30 13 * * 1-5      cd /srv/copytrade && .venv/bin/python -m copytrade report
```

`scrape_sources = ["auto"]` runs every feed whose prerequisites are met, so a
fresh box scrapes the free authoritative feeds immediately and lights up
Finnhub/Canada only once their keys/URLs are set. Check liveness with
`scrape-status`.
