# CopyTrade Pipeline: Project Plan / Context Document

> Purpose: Build a disclosure-based copy-trading research pipeline that ingests US politician,
> insider, and hedge-fund filings, backtests their copyability with strict point-in-time
> discipline, ranks filers by lag-adjusted alpha, and (optionally) paper-trades the top signals.
> This document is the single source of truth. Read fully before writing code.

---

## 0. Core thesis and constraints (do not violate)

1. **There is no live feed to intercept.** All signals come from mandatory public disclosures:
   - Congress: STOCK Act Periodic Transaction Reports (PTRs). Filed ≤30 days after awareness,
     ≤45 days after execution. Dollar amounts are RANGES, not exact values.
   - Corporate insiders: SEC Form 4. Filed within 2 business days. Freshest signal.
   - Hedge funds: SEC 13F. Quarterly, up to 45 days after quarter end. Longs only.
2. **Point-in-time discipline is non-negotiable.** A trade is only actionable at
   `filing_date` (actually: first public availability), never `transaction_date`.
   Every backtest executes at the first market close AFTER the filing timestamp.
   Any code path that peeks at transaction_date for execution is a bug.
3. **The research question:** for each filer, how much of their excess return survives the
   disclosure lag? Output = per-filer signal decay curves and lag-adjusted alpha.
4. **Respect data sources:** SEC EDGAR fair-access rules (≤10 req/s, declared User-Agent
   with contact email, back off on 403/429). No scraping of Autopilot or any brokerage app's
   private API. Aggregator APIs used within their ToS/free tiers.
5. **Money safety:** Phases 1-4 touch no brokerage. Phase 5 is paper trading only unless the
   human explicitly changes this. Never store brokerage credentials in the repo; use env vars.

## 1. Tech stack

- Python 3.11+, `uv` or `pip` with pinned requirements.
- Storage: SQLite (single file `data/copytrade.db`), accessed via `sqlite3`/SQLAlchemy core.
  FTS not needed. Keep schema migrations as numbered SQL files in `migrations/`.
- HTTP: `httpx` (async) with retry/backoff wrapper. All pollers are asyncio tasks.
- Data/backtest: `pandas`, `numpy`. Plots: `matplotlib` (PNG artifacts to `reports/`).
- Config: `config.toml` + `.env` (API keys). `pydantic-settings` for loading.
- Alerts: single webhook poster (Discord or Telegram), configurable, optional.
- Tests: `pytest`. Every parser gets golden-file tests with real captured payloads in
  `tests/fixtures/`.
- CLI entrypoint: `python -m copytrade <command>` using `typer`.

Repo layout:
```
copytrade/
  ingest/        # per-source pollers + parsers
  normalize/     # mapping raw payloads -> unified schema
  store/         # db access layer, migrations runner
  backtest/      # point-in-time engine, metrics
  rank/          # filer scoring
  alerts/        # webhook notifier
  cli.py
config.toml
migrations/
tests/
reports/         # generated: equity curves, decay plots, ranking tables
data/            # sqlite db + raw payload cache (gitignored)
```

## 2. Data sources

### 2.1 SEC EDGAR (free, primary, build first)
- Base: `https://www.sec.gov` and `https://data.sec.gov`.
- Headers: `User-Agent: <name> <email>` on every request. Global rate limiter ≤8 req/s.
- Discovery of new filings: EDGAR full-text/daily index or
  `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany...` and the
  "recent filings" JSON at `https://data.sec.gov/submissions/CIK##########.json`.
  For near-real-time: poll `https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4`
  Atom feed every 2-5 min during market hours.
- **Form 4 (insiders):** parse the XML (`form4.xml` inside the filing). Extract: issuer CIK/
  ticker, reporting owner + relationship (officer/director/10% owner), transaction code
  (P=open-market purchase, S=sale, P is the high-signal one), shares, price, transaction
  date; filing accepted timestamp = point-in-time anchor.
- **13F-HR:** quarterly XML info tables. Extract manager CIK, period, CUSIP→ticker mapping
  (maintain a CUSIP table; fall back to OpenFIGI free API), value, shares. Compute
  quarter-over-quarter position deltas as the "trades".
- Cache every raw payload to `data/raw/<accession>.xml` before parsing (reprocessable).

### 2.2 Congressional trading (aggregator first, own scraper later)
- MVP: one aggregator, behind an interface so it's swappable:
  - Finnhub `/stock/congressional-trading` (key already available in .env from FinSignal), or
  - Quiver Quantitative / Unusual Whales if a key is provided.
- Fields needed: member name, chamber, party, ticker, side, transaction_date, filing_date
  (a.k.a. disclosure date: REQUIRED; skip records without it), amount range low/high,
  asset type (stock / option / other). Options records: keep but flag `instrument='option'`;
  MVP backtests underlying-only.
- Phase 6 (later): direct scrapers for House Clerk (disclosures-clerk.house.gov, PDFs, needs
  OCR fallback, defer) and Senate eFD (efdsearch.senate.gov, HTML, session/CSRF dance).

### 2.3 Market data (for backtesting and benchmarks)
- Daily adjusted closes via `yfinance` for tickers + SPY (total-return proxy: use adjusted
  close). Cache to parquet in `data/prices/`. Nightly refresh command.
- Delisted-ticker handling: if price history missing after filing date, mark trade
  `unpriceable` and report the count (survivorship-bias disclosure in every report).

## 3. Unified schema (migration 001)

```sql
CREATE TABLE filers (
  filer_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN ('congress','insider','fund')),
  meta JSON            -- chamber/party/committees, or officer title, or fund CIK
);
CREATE TABLE trades (
  trade_id INTEGER PRIMARY KEY,
  filer_id INTEGER NOT NULL REFERENCES filers,
  source TEXT NOT NULL,             -- 'edgar_form4','edgar_13f','finnhub_congress',...
  source_uid TEXT NOT NULL,         -- accession no / aggregator id ; UNIQUE(source, source_uid)
  ticker TEXT, cusip TEXT,
  side TEXT CHECK(side IN ('buy','sell')),
  instrument TEXT DEFAULT 'stock',  -- stock|option|other
  transaction_date DATE,
  filing_datetime TIMESTAMP NOT NULL,   -- point-in-time anchor
  amount_low REAL, amount_high REAL,    -- ranges (congress) or exact*1 (form4: shares*price)
  shares REAL, price REAL,              -- when exact (form4)
  raw_ref TEXT                          -- path to cached raw payload
);
CREATE INDEX idx_trades_filer ON trades(filer_id, filing_datetime);
CREATE TABLE prices (ticker TEXT, date DATE, adj_close REAL, PRIMARY KEY(ticker,date));
```
Dedup rule: `UNIQUE(source, source_uid)`; cross-source dedup (same congress trade from two
aggregators) by (filer, ticker, side, transaction_date, amount_low), keep earliest
filing_datetime.

## 4. Backtest engine (the heart, build with tests first)

### 4.1 Execution model
- Signal time: `filing_datetime`. Execution: next trading day's close after filing
  (conservative; also support same-day close if filed before 15:30 ET as a sensitivity toggle).
- Position size: amount-range midpoint (congress) or shares*price (Form 4), normalized to
  portfolio weights; cap any single trade at configurable max weight (default 5%).
- Costs: configurable, default 10 bps per side (spread+slippage proxy). No shorting in MVP
  (sells only close existing copied longs / are used as exit signals).
- Exit rules (configurable per experiment):
  a) hold fixed horizon H ∈ {5, 21, 63, 126, 252} trading days (for decay curves);
  b) mirror-exit: close when the filer's corresponding sell is *filed*;
  c) portfolio mode: maintain weights mirroring filer's disclosed book (13F cloning).
- Benchmark: SPY adjusted close, same execution dates. All alpha = strategy − SPY,
  matched-horizon.

### 4.2 Core outputs (per filer and pooled per cohort)
- **Decay curve:** mean cumulative excess return over event time, computed twice
  anchored at transaction_date vs anchored at filing_date. The vertical gap = alpha eaten
  by the lag. This is the single most important chart in the project.
- Hit rate, mean/median excess return at each horizon, t-stats (cluster by calendar month),
  buys and sells SEPARATELY (expect buys ≫ sells in information content).
- Turnover class: median holding period inferred from matched buy→sell pairs; classify
  filer as short-term (<63 td), medium (63-252), long (>252). This is the user's
  long/short-term filter.
- Portfolio simulation: equal-weight top-N filers by trailing lag-adjusted alpha,
  walk-forward (rank on years t-2..t-1, trade year t) to avoid selection look-ahead.

### 4.3 Report generation
`python -m copytrade report` writes to `reports/`:
- `ranking.csv` (filer, kind, n_trades, lag_alpha_63d, tstat, hit_rate, turnover_class)
- decay plots per top-20 filer, pooled cohort plots, walk-forward equity curve vs SPY,
- a `README.md` summarizing methodology caveats (ranges not exact sizes, options excluded,
  survivorship notes, price-index vs total-return handling).

## 5. Live pipeline

- `python -m copytrade poll` runs asyncio pollers: Form 4 feed (2-5 min), congress
  aggregator (30-60 min), 13F (daily check during filing windows).
- On new trade insert: evaluate against current top-N filer list; if actionable, send
  webhook alert: filer, ticker, side, size range, lag (filing − transaction), filer's
  trailing lag-alpha stats. Alerts are informational; no auto-execution in MVP.
- Systemd/user cron friendly; graceful resume from last accession/id (state table).

## 6. Phase plan with acceptance criteria

- **P1 Ingest-EDGAR-Form4**: poller + parser + storage. AC: 30 days of recent Form 4s
  ingested; golden-file tests pass; rate limiter verified; rerun is idempotent.
- **P2 Ingest-Congress (aggregator)**: AC: ≥2 years of congress trades stored with filing
  dates; dedup works; records missing filing_date are counted and excluded.
- **P3 Prices**: AC: all tickers appearing in trades have cached daily history or are
  flagged unpriceable; SPY benchmark cached.
- **P4 Backtester**: AC: unit tests prove no look-ahead (synthetic fixture where using
  transaction_date would change results); decay curves generate; costs applied.
- **P5 Ranking + report**: AC: ranking.csv + plots reproduce deterministically from db.
- **P6 Live poll + alerts**, AC: end-to-end new Form 4 → alert < 10 min.
- **P7 13F cloning module**: AC: quarterly rebalance sim for ≥5 low-turnover managers.
- **P8 (optional, human-gated) Alpaca paper trading**, AC: alerts can place paper orders;
  daily reconciliation report. Real money is explicitly out of scope for this repo.

## 7. Known pitfalls (encode as tests/checks where possible)

- Congress amounts are ranges; never treat midpoint as exact, report sensitivity
  (low/mid/high) in at least one experiment.
- House data historically messy (name variants, spouse trades, PDFs). Normalize names via a
  manual alias table `filers_aliases`.
- Form 4 transaction codes: only P (and optionally A) count as informative buys; S as sells;
  ignore F, M, G, etc. in MVP but store them.
- 13F shows quarter-end longs only: no shorts, no timing, window-dressing risk. Restrict
  cloning to managers with turnover < 25%/quarter and concentrated books (top-10 ≥ 40%).
- Ticker changes/mergers: map via CUSIP where available; log unresolved.
- Multiple-hypothesis risk in ranking: use walk-forward, report deflated Sharpe or at least
  demand t-stat > 2 with n_trades ≥ 30 before a filer is "copyable".
- UK user context: eventual real trading via IBKR; CGT implications of high turnover
  out of code scope, but keep per-trade P&L export (`taxlots.csv`) to make this analyzable.

## 8. Definition of done (MVP)

One command bootstrap (`make setup && make backfill && make report`) produces, from public
data only: a ranked table of congress members and insiders by lag-adjusted 63-day alpha,
decay-curve plots showing how much alpha the disclosure lag destroys per cohort, and a
walk-forward equity curve of "copy the top 10" vs SPY, with an honest caveats section.
