# CopyTrade: disclosure-based copy-trading research pipeline

Full-stack implementation of [`COPYTRADE PROJECT PLAN.md`](COPYTRADE%20PROJECT%20PLAN.md):
ingest US politician (STOCK Act PTR), corporate insider (SEC Form 4) and hedge-fund (13F)
filings, backtest their **copyability** with strict point-in-time discipline, rank filers by
**lag-adjusted alpha net of real transaction costs**, compare the whole thing against
low-effort benchmarks (S&P buy-and-hold, DCA, 60/40), and serve it all through a web
dashboard. Built for a **Canadian resident (Vancouver, BC)**, costs, account structure
(TFSA vs margin) and platform recommendations are Canada-first; see
[`docs/CANADA.md`](docs/CANADA.md). Live polling with webhook alerts is included;
**nothing places live orders**: paper trading is the ceiling.

The core research question: *how much of a filer's edge survives the disclosure lag
and the fees: by the time you can legally copy it?*

## The two rules everything enforces

1. **Point-in-time discipline.** A trade is only actionable at its *filing* timestamp.
   Every backtested entry executes at the first market close **after** the filing
   never at the transaction date. The transaction-date anchor exists only to draw the
   hypothetical zero-lag decay curve; the vertical gap between the two curves is the
   alpha the lag destroys. Tests prove a pre-filing price pop cannot be captured
   (`tests/test_engine.py`), and the paper-order ledger obeys the same rule
   (`tests/test_paper.py`).
2. **Net of costs.** Every table and equity curve subtracts a configurable cost model:
   spread/slippage, commission (per-trade / per-share / %, with minimums), CAD↔USD FX
   conversion, and the SEC section 31 fee on sells. For a Canadian on a
   "zero-commission" platform the FX line item dominates, that is modelled, not
   hand-waved.

## Quickstart

```bash
make setup          # venv + pinned deps (Python 3.11+)
make demo           # offline synthetic dataset with KNOWN skill/lags (no network)
make report         # ranking.csv, decay plots, walk-forward vs benchmarks -> reports/
make serve          # dashboard + JSON API at http://127.0.0.1:8000
make test           # 50 tests incl. real-payload parsers and no-look-ahead proofs
```

### Real data (no API key needed for the first two)

```bash
make real           # senate PTRs (real disclosure dates) + USD/CAD FX + prices + report
# ... or step by step:
.venv/bin/python -m copytrade ingest-senate     # REAL Senate PTRs, 2013-2021, via GitHub
.venv/bin/python -m copytrade fx-refresh        # REAL daily USD/CAD (BoC, Fed fallback)
.venv/bin/python -m copytrade prices            # daily adjusted closes via yfinance
.venv/bin/python -m copytrade report
.venv/bin/python -m copytrade allocate          # TFSA vs margin sleeve plan
```

Live/broader sources (network + keys; see the obstacles table in
[`docs/CANADA.md`](docs/CANADA.md) if any of these are blocked where you run):

```bash
cp .env.example .env                         # add COPYTRADE_FINNHUB_API_KEY
.venv/bin/python -m copytrade backfill-form4 --days 30    # SEC EDGAR insider filings
.venv/bin/python -m copytrade backfill-congress --years 2 # Finnhub congress aggregator
.venv/bin/python -m copytrade poll           # live Form 4 + congress polling, webhook alerts
```

### Constant scraping (all sources, one scheduler, see `docs/SCRAPING.md`)

```bash
.venv/bin/python -m copytrade poll          # long-lived loop: every enabled feed at its cadence
.venv/bin/python -m copytrade scrape-once   # one pass of every source (cron-friendly)
.venv/bin/python -m copytrade scrape-status # per-source liveness/health heartbeats
.venv/bin/python -m copytrade scrape-house  # authoritative US House PTR scrape
```

Sources register behind one interface (US insiders, US House, US Senate, Canadian
MPs, funds); **authoritative source-of-record feeds are the backbone, Finnhub is
enrichment**: because Finnhub's congress endpoint is symbol-keyed and would
otherwise silently blind you to trades outside your watchlist. A failing source
backs off and never takes the loop down. Canada has no STOCK Act, so the Canadian
source is a coarse material-change supplement, documented as such.

### Mirror one filer, derivatives, and lag

```bash
.venv/bin/python -m copytrade mirror Pelosi      # copy ONE named filer vs SPY, net of costs
.venv/bin/python -m copytrade derivatives        # stock vs option vs futures framing (see below)
.venv/bin/python -m copytrade cftc-refresh        # CFTC futures positioning (large speculators)
.venv/bin/python -m copytrade lag                # disclosure-lag decomposition + fastest filers
```

`mirror` is the honest "copy one famous member" test, concentrated, no
significance gate, no pretence the filer was selectable in advance; options
are excluded (unpriceable), so a member whose edge is *leverage* looks weaker
here than the headlines, and that gap is the point.

`derivatives` is a **separate section for leveraged bets by powerful people**:
option trades (congress PTRs, 13F put/call holdings, insider derivative table)
shown as the *underlying's* signed directional move from filing, not option
P&L, since the leverage is unmeasurable from range disclosures, plus CFTC
Commitments-of-Traders large-speculator futures positioning. The value is the
framing: is a filer's edge stock-picking, or timing/leverage?

### Paper trading (Phase 8, human-gated)

```bash
.venv/bin/python -m copytrade orders-from-signals   # BUY filings by top filers -> paper orders
.venv/bin/python -m copytrade orders-fill           # fill at first close AFTER order (THE rule)
.venv/bin/python -m copytrade orders-push-alpaca    # mirror into an Alpaca PAPER account
.venv/bin/python -m copytrade orders-list
```

The Alpaca client **refuses non-paper endpoints** unless `COPYTRADE_ALLOW_LIVE=1` is
set on purpose. No live-order code exists for any Canadian broker yet, that is a
deliberate gate, not an oversight.

## What comes out

- `reports/ranking.csv`: per filer: n_trades, lag-adjusted 63d alpha, **alpha lost to
  lag**, month-clustered t-stat, hit rate, median filing lag, turnover class
  (short/medium/long), a `routine` flag (same-calendar-month filers à la
  Cohen-Frazzini-Malloy 2012: calendar-driven trades carry little information),
  and a `copyable` flag that demands t > 2 with n ≥ 30 (multiple-hypothesis
  discipline).
- `reports/track_records.csv`: **per-filer profits with NO significance gate**
  (`python -m copytrade profits`): realized position returns, win rate, $/pick,
  annualized-per-pick, holding period, and a `confidence` label by sample size.
  This is where a filer who trades a *few high-conviction names a year* surfaces
  they'll never clear the `copyable` bar but their track record is right here.
- `reports/decay_cluster_vs_solo.png`: **cluster buys** (≥2 distinct filers in the
  same ticker within 10 days, trailing-window so point-in-time safe) vs solo buys:
  does the consensus premium documented in the insider literature exist in this data?
  Live alerts flag new cluster buys with the same trailing logic.
- `reports/allocation.csv`: **TFSA vs margin sleeve per copyable filer** with the
  reason (CRA business-income risk gates frequent/short-hold filers out of the TFSA).
- `reports/decay_{congress,insider}.png` + per-filer plots, mean cumulative excess
  return in event time, transaction-anchored vs filing-anchored. **This is the key
  chart**: the gap is the alpha the disclosure lag eats.
- `reports/walkforward_equity.png`: "copy the top-10 filers" simulated walk-forward
  (ranked each year only on trades whose outcome had resolved before that year began)
  vs **SPY buy & hold**, **SPY DCA over 12 months**, **quarterly-rebalanced 60/40
  SPY/TLT**, a **copy-the-exit variant** (positions close when the filer files a
  sell), and a **"Copy ALL filers" baseline** (what the NANC/KRUZ ETFs do, if
  top-N can't beat it, selection adds nothing), all net of the same cost model,
  with a **bootstrap p-value** on the excess over the benchmark. If the strategy
  can't beat these, the honest answer is "buy the index".
- `reports/decay_party.png` / `decay_lag_speed.png`, cohort decays by party
  (via `unitedstates/congress-legislators` enrichment; 43/43 real senators
  matched) and by disclosure speed (fast ≤15 bd vs slow filers).
- `reports/sensitivity_amounts.png`: congress amounts are *ranges*; the run is repeated
  sizing positions at the low/mid/high of each range.
- `reports/taxlots.csv`: closed-lot ledger (entry/exit/P&L) for CRA capital-gains /
  ACB records.
- `reports/README.md`: the numbers with the caveats attached.

## Re-testing the null: timescale, selection, exits, leverage

The headline result is that copying disclosures produces **no statistically
significant alpha**. That verdict was reached at one holding horizon (63d), with
in-sample filer ranking, one exit rule, and no multiple-testing correction. Those
are four distinct ways a real effect could hide or a fake one could appear, so
each now has a command that searches its axis *and corrects for having searched*.

**Current state (rebuilt 2026-08-02, 36,093 trades):** of 44 {subset × horizon}
cells, **17 survive Benjamini-Hochberg FDR, every one negative and in the
short band. Zero positive cells survive.** Gross excess return at 63 days is
−0.31% and the cheapest Canadian round trip is 30bps, so the net result is very
nearly the cost itself. Beta-neutral alpha t = −0.97; Fama-French alpha
−1.5%/yr (t = −0.67) with R² 0.82 on the market factor alone.

```bash
python -m copytrade timescale      # {subset x horizon}, 1 day to 2 years
python -m copytrade who-to-copy    # is filer skill persistent, out of sample?
python -m copytrade exits          # does any exit rule beat "hold N days"?
python -m copytrade leverage       # how hard can this be pushed before ruin?
python -m copytrade greeks --cad   # the full battery, in the currency you spend
python -m copytrade holdings --filer "Cleo Fields"   # what a member is disclosed to be in
```

`holdings` is the honest version of the "portfolio" screen every consumer
tracker leads with. A PTR discloses **transactions, not holdings**, so positions
opened before the tracking window are invisible and every size is a range
it therefore reports a disclosed net **band** with a qualitative status, never a
share count or a market value.

- **`timescale`** sweeps horizons from 1 to 504 trading days, split into the two
  sleeves the strategy always implied, **SHORT (1-21d)**, event-driven
  post-filing drift, and **LONG (42-504d)**, durable tilt. A ~44-cell grid throws
  off ~2 "significant" cells from pure noise, so Benjamini-Hochberg FDR is applied
  across the whole grid and nothing is reported as a finding unless it survives.
  Long-horizon cells are additionally clustered on **non-overlapping blocks** as
  long as the holding period, because two 504-day positions opened a month apart
  share ~96% of their window and are not independent observations.
- **`who-to-copy`** asks whether *any* stable skill differential exists, rather
  than whether one particular selection rule works. Filers are ranked on a train
  window and measured on the **next** window. A null here is strongly actionable:
  it means stop hunting for whom to copy.
- **`exits`** tests trailing stops, take-profits and stop-losses against the
  fixed-horizon baseline. The column that matters is the **paired** difference
  a rule can look profitable purely by inheriting what the baseline already had.
- **`leverage`** models a **fixed loan** (the realistic retail case, where the
  margin ratio decays in a drawdown) rather than daily-rebalanced constant
  leverage (which resets the ratio every morning and badly understates risk of
  ruin). Sizing should key off the **bootstrapped margin-call probability**, not
  the realised drawdown: the realised path is one draw, and one already known to
  have ended well.
- **`greeks --cad`** reports in Canadian dollars. Holding a USD asset is also a
  long-USD/short-CAD position; that translation effect is separate from the FX
  *spread* in the cost model and does not net to zero over a sample.

## Cost models (`copytrade/backtest/costs.py`)

| preset | commission | FX (per side) | notes |
|---|---|---|---|
| `ibkr_ca` | $0.005/share, min $1 | 3bps | IBKR Canada Fixed, US stocks. Cheapest FX in Canada + real API. **Default.** |
| `questrade` | $0 | 150bps | $0 stock commissions; the CAD↔USD conversion is the real cost. |
| `questrade_gambit` | $0 | 4bps | Questrade + Norbert's gambit (DLR/DLR.U) for the conversion. |
| `wealthsimple` | $0 | 150bps | Core tier; no trading API, included so the fee drag is visible. |
| `alpaca` | $0 | - | US, API-first; slippage 10bps/side + SEC sell fee. Paper-trading target. |
| `default10bps` / `zero` | - | - | The plan's 10bps proxy / frictionless sensitivity check. |

Switch via `cost_model` in `config.toml` or `?cost_model=` on the API. Round-trip costs
of 30-300bps per position erase most slow-filing congress "edges", which is precisely
the point of modelling them. Platform details, TFSA/CRA rules, deployment-in-BC notes
and the live-data obstacle map live in [`docs/CANADA.md`](docs/CANADA.md).

## Keys & secrets (exact names)

Local: put them in `.env`. GitHub Actions: repo → Settings → Secrets and
variables → Actions → **New repository secret**, same names. Everything runs
with zero keys; each one upgrades a capability.

| secret name | tier | unlocks |
|---|---|---|
| `COPYTRADE_FINNHUB_API_KEY` | free | live congress trades (both chambers) + company news |
| `COPYTRADE_TIINGO_API_KEY` | free | price fallback with true adjusted closes when yfinance throttles |
| `COPYTRADE_WEBHOOK_URL` | free | Discord/Telegram alerts (cluster buys, watchlist filings) |
| `COPYTRADE_OPENFIGI_API_KEY` | free | higher CUSIP→ticker rate limits for 13F backfills |
| `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY` | free | paper-trading order mirroring |
| `COPYTRADE_QUIVER_API_KEY` | paid | bulk live congress feed, no symbol blind spot (`backfill-quiver`) |

## Data sources: what is real, what is demo

| source | command | filing anchor | status |
|---|---|---|---|
| Senate Stock Watcher (GitHub) | `ingest-senate` | real `date_recieved` per PTR | **real**, ~2013-2021 (repo stale since 2021-03 → backtest-grade) |
| USD/CAD daily FX | `fx-refresh` | - | **real & live** (BoC Valet; Fed H.10 mirror fallback) |
| SEC EDGAR Form 4 | `backfill-form4`, `poll` | real `ACCEPTANCE-DATETIME` | **real & live** (needs direct sec.gov access) |
| Finnhub congress | `backfill-congress`, `poll` | `filingDate` from API | **real & live** (needs API key) |
| SEC DERA bulk insider sets | `backfill-form345 2024q1` | real `FILING_DATE` (day precision) | **real**, quarterly ZIPs = years of Form 3/4/5 in one download (needs sec.gov; validate one quarter with `--limit` first) |
| 13F-HR fund holdings | `backfill-13f <cik>` | real `acceptanceDateTime` | **real**, QoQ deltas; CUSIP→ticker via OpenFIGI |
| Company news (Finnhub / GDELT) | `news-refresh` | - | **real**, context for alerts/dashboard only, never a backtest factor |
| Market prices | `prices` | - | **real** via yfinance adjusted closes |
| `demo` | `demo` | synthetic | clearly labeled; separate DB enforced (`demo` refuses a DB holding real trades) |

Records without a disclosure date are **counted and skipped, never backdated**, the
senate aggregate file (no `date_recieved`) is rejected in favour of the per-day report
files for exactly this reason.

## Full stack

```
FastAPI (copytrade/web/app.py)          SQLite (data/copytrade.db)
  GET  /api/summary                       filers / trades / prices / holdings_13f
  GET  /api/analytics?cost_model=...      orders (paper ledger) / fx_rates
  GET  /api/allocation?tfsa_room=...      UNIQUE(source, source_uid) dedup
  POST /api/refresh                     CLI (python -m copytrade ...)
  GET  /api/trades/recent                 migrate demo ingest-senate fx-refresh
  GET  /api/filers/{id}/trades            backfill-form4 backfill-congress prices
  /reports/* (PNGs)  /  (dashboard)       rank report allocate poll serve orders-*
```

The dashboard (`copytrade/web/static/index.html`) is a single self-contained page
no CDN dependencies: with light/dark theming, crosshair-tooltip canvas charts, and
sortable tables. It shows **22 panels** in four groups:

| group | panels |
|---|---|
| Findings | scorecard (USD/CAD, headline or every metric) · timescale grid + decay shape + band summary · net edge (absolute, beta-neutral, consensus CAR) · Fama-French factors · conviction · committee jurisdiction · persistence · exits · leverage (+ return-vs-ruin scatter) · broker-cost sensitivity |
| Evidence | walk-forward equity + capacity · signal decay · year by year · disclosure lag · sector attribution |
| Portfolio | filer ranking · track records · **disclosed positions** · TFSA/margin sleeves |
| Data | health · integrity (checks, price quarantine, review tier) · recent filings · caveats |

**Every panel is precomputed** by `python -m copytrade dashboard-build` and served
from `reports/dashboard/*.json` with its build timestamp attached. Nothing is
computed in a request handler: the timescale sweep, the leverage bootstrap and the
walk-forward each take minutes on the live sample, and when the walk-forward,
decay, ranking and sleeve sections *were* computed per request they never
returned, so those four sections of the dashboard sat permanently blank. A full
build is ~10 minutes; `--only <names>` rebuilds a subset.

`GET /api/analytics` still serves the legacy combined shape, now assembled from
those panels; pass `?live=true` to force a recompute.

## Repo layout

```
copytrade/
  ingest/     edgar.py form4.py form345_bulk.py congress.py senate_watcher.py
              senate_efd.py house_clerk.py canada_ciec.py legislators.py
              fx.py news.py thirteenf.py market.py
  scrape/     base.py (Source) registry.py scheduler.py sources.py
  store/      db.py (sqlite + migrations runner)
  backtest/   engine.py (point-in-time) costs.py benchmarks.py metrics.py
  rank/       rank.py signals.py (cluster/routine) profit.py (track records)
  accounts/   allocation.py (TFSA vs margin)
  broker/     paper.py         alerts/   webhook.py
  report/     report.py        web/      app.py + static/index.html
  demo/       synthetic.py     cli.py    poll.py  config.py
  store/      db.py (sqlite + migrations runner) price_qa.py (price-series screens)
migrations/   001_init … 008_trade_date_sanity 009_sector_attempts 010_price_quarantine
docs/         CANADA.md (platforms, fees, TFSA/CRA, deploy) SCRAPING.md (scrape architecture)
tests/        real-payload fixtures + no-look-ahead proofs
```

## Data integrity

Three screens run over the stored data, because every result inherits whatever
is wrong in it and two bugs of exactly this shape once drove the whole-sample
mean below −100% for a long-only book.

```bash
python -m copytrade prices-qa            # screen price series; quarantine the impossible
python -m copytrade dashboard-build --only quality   # the report
```

- **Impossible transaction dates.** A PTR discloses a trade that already
  happened, so a transaction date after its own filing is a parse error. The
  House Clerk PDFs carried 3031-04-30, 2220-04-07 and ~80 quieter typos. The
  date is nulled and the **trade is kept**, `filing_datetime` is the execution
  anchor and is unaffected, so the trade stays in every backtest and only drops
  out of the disclosure-lag statistics. Blocked at ingest in `insert_trade`.
- **Price series that are positive but not credible.** The earlier
  `adj_close <= 0` guard removed the negative ticks of four broken OTC ADRs and
  left the rest of those series in place; DAIUF still ran 5.1e-07 → 45.31 and
  reached the dashboard as the best-performing name in the sample. `prices-qa`
  screens for sub-penny artifacts, unadjusted reverse splits, series
  oscillating between two scales, and multi-year gaps read as one session.
  Offenders are **quarantined, not deleted** (providers would just re-supply
  them) and excluded at `load_price_matrix`, the single place every analysis
  reads prices. Real events survive on purpose, DJT +357%, GME +135%.
- **Sector-lookup failures are cached.** `sectors-refresh` used to store only
  successes, so it re-requested every failure on the next run, tripped
  yfinance's rate limiter, and then reported `mapped: 0`, which reads as
  "nothing left to map" when it means "the provider refused". Attempts are now
  recorded and a throttled run says so.

## Honest limitations

- Congress dollar amounts are ranges; every number derived from them is approximate
  (see the sensitivity run).
- The free senate dataset ends March 2021, it powers the backtest, not live signals.
  Live congress data needs Finnhub/Quiver/etc. or direct efdsearch scraping
  (see `docs/CANADA.md`).
- Unpriceable/delisted tickers are excluded and counted; exits with no real print that
  day are counted as `stale_exits`, survivorship bias is disclosed in every report,
  not hidden.
- 13F cloning sees quarter-end longs only; restrict to low-turnover concentrated
  managers (plan §7) before trusting it.
- A ranking of hundreds of filers **will** contain lucky streaks; trust the walk-forward
  and the clustered t-stats, not the leaderboard.
- TFSA/CRA notes are educational summaries with sources, not tax advice.
- Research tooling, not investment advice.
