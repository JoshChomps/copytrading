# Canada guide: platforms, fees, accounts, deployment, live data

Everything Canada-specific in one place, written for a resident of Vancouver, BC.
Educational summaries with sources, verify tax specifics with CRA/an accountant
before acting.

## 1. Platform recommendation (in order)

### Interactive Brokers Canada: the execution platform (`ibkr_ca`, default)

- **Why**: the only major Canadian-resident brokerage with a real trading API
  (Client Portal Web API / TWS API, `ib_insync`), tiered/fixed commissions
  (USD 0.005/share, min $1, max 1% of trade value), and near-interbank FX
  (spot + 0.5bp, min USD ~2 per conversion). Registered accounts (TFSA, RRSP)
  are supported alongside margin.
- **For this pipeline**: paper trading via the IBKR paper account works with the
  same API. Live order code is intentionally not implemented yet.
- **Costs modelled**: `ibkr_ca` = 10bps slippage + $0.005/sh (min $1) + 3bps FX
  + SEC sell fee.

### Questrade: the TFSA-friendly runner-up (`questrade`, `questrade_gambit`)

- $0 stock commissions (since Feb 2025), full registered-account lineup, has an
  API (REST, OAuth) though order endpoints are more restricted than IBKR's.
- The catch: CAD↔USD conversion around **1.5% each way** unless you use
  **Norbert's gambit** (buy DLR.TO with CAD, journal, sell DLR.U for USD) which
  collapses FX to roughly the ETF spread, modelled as `questrade_gambit` (4bps).
- Good home for the **TFSA sleeve** (slow-turnover copying / index ETF).

### Wealthsimple: visibility only (`wealthsimple`)

- $0 commissions but **1.5% FX each way** on USD trades from a CAD account (core
  tier) and **no public trading API**, unusable for automation. The preset exists
  so the report can show what the fee drag would be.

Sources: [Questrade vs IBKR 2026 (MDJ)](https://milliondollarjourney.com/questrade-vs-interactive-brokers.htm),
[Savvy New Canadians comparison](https://www.savvynewcanadians.com/questrade-vs-interactive-brokers/),
[LifeMoney 2026 comparison](https://lifemoney.ca/blog/questrade-vs-interactive-brokers-2026).

### Paper trading today: Alpaca (`orders-push-alpaca`)

Free paper API, no residency requirement for paper accounts, notional market
orders: the fastest way to shake out the signal→order plumbing with live
quotes. Live Alpaca accounts for Canadians are limited; live execution belongs
on IBKR Canada once (and only if) paper results justify it.

## 2. TFSA vs non-registered split (the `allocate` command)

The strategy's income is split across two sleeves to maximize the TFSA while
keeping CRA risk out of it:

| | TFSA sleeve | Margin (non-registered) sleeve |
|---|---|---|
| what goes in | copyable filers with **medium/long** turnover and ≤ ~24 implied orders/yr; otherwise broad-index ETF | all frequent / short-hold copying |
| tax on gains | none, **if** CRA doesn't deem it a business | capital gains (50% inclusion) or business income, facts-dependent |
| key risk | **S3-F9-C1 business-income reassessment**: frequency, short holds, options, finance knowledge, time spent. Retroactive, with interest. No safe trade-count bright line. | none specific; losses are at least usable |
| US dividends | 15% treaty withholding, **unrecoverable** in a TFSA | withheld but creditable against Canadian tax |
| contribution room 2026 | $7,000 new; $109,000 cumulative since 2009 if always eligible, **verify in CRA My Account** (over-contribution = 1%/month) | unlimited |

Practical rules encoded in `copytrade/accounts/allocation.py`:

1. Only `copyable` filers (clustered t>2, n≥30) get capital at all.
2. Short-turnover or high-frequency filers → margin sleeve, with the reason
   printed per filer in `allocation.csv`.
3. If nothing qualifies for the TFSA, the room goes to the benchmark (index
   ETF) rather than sitting idle, tax-free beta beats no beta.
4. High-yield US names belong in an RRSP (treaty-exempt), not the TFSA, the
   RRSP is out of scope for the copy book but flagged in the notes.

Sources: [Questrade on TFSA day-trading rules](https://www.questrade.com/learning/tfsa-day-trading-cra-rules),
[Jamie Golombek on CRA trade-frequency audits](https://www.jamiegolombek.com/articledetail.php?article_id=2126),
[Wealthsimple on withholding](https://help.wealthsimple.com/hc/en-ca/articles/360056584994),
[TD: taxes for Canadians holding US equities (PDF)](https://advisors.td.com/dao.le/mediahandler/media/311624/Tax_Implication_cdn_investing_US_equities.pdf).

## 3. Running it from Vancouver

- **Timezones**: market logic is Eastern (filing cutoff `15:30 ET` = **12:30 PT**;
  market close 16:00 ET = 13:00 PT). `display_tz = "America/Vancouver"` in
  `config.toml` is for reporting; schedule crons in PT accordingly, e.g. the
  daily post-close report run at **13:30 PT**, the Form 4 poller from
  **06:00-17:15 PT** (EDGAR acceptance window 06:00-22:00 ET plus the 17:30 ET
  after-hours batch = 14:30 PT).
- **Hosting in Canada (west)**: the closest full cloud region is
  **AWS ca-west-1 (Calgary)**: single-digit-ms from Vancouver, keeps data in
  Canada. Alternatives: AWS ca-central-1 / Azure Canada Central (both
  Toronto-area), OVHcloud Beauharnois (QC) for cheap bare metal. A $5 VPS or a
  Raspberry Pi at home is genuinely sufficient: the pipeline is SQLite + a few
  MB/day of polling; nothing here is latency-sensitive (execution is
  next-day-close by design).
- **Latency caveat**: don't pay for proximity to exchanges. The strategy's edge
  horizon is measured in *days* (disclosure lag), not microseconds.

## 4. Live data: obstacle map (measured, not guessed)

Tested 2026-07 from a restricted sandbox (git+PyPI egress only), the left
column is what this repo does about it. From an unrestricted machine
(home/VPS), everything below works directly.

| source | sandbox status | how to get it |
|---|---|---|
| Senate Stock Watcher (GitHub) | ✅ works (git clone) | `ingest-senate`; **real disclosure dates**; frozen at 2021-03 |
| USD/CAD FX (Fed H.10 mirror on GitHub) | ✅ works, **live** (updated weekly) | `fx-refresh` fallback path |
| Bank of Canada Valet API | ❌ 403 via proxy | `fx-refresh` primary path, works off-sandbox |
| SEC EDGAR (Form 4, acceptance datetimes) | ❌ 403 via proxy | `backfill-form4` / `poll` off-sandbox; fair-access UA + ≤8 rps built in |
| Finnhub congressional API | ❌ 403 via proxy | `backfill-congress` with free API key off-sandbox |
| yfinance / stooq price history | ❌ 403 via proxy | `prices` off-sandbox (yfinance, adjusted closes) |
| House Stock Watcher S3 | ❌ 403 **everywhere** (bucket dead since ~2025) | House data needs Finnhub/Quiver/FMP or the Capitol Trades/Lambda APIs |
| SEC DERA insider bulk ZIPs | ❌ 403 via proxy | `backfill-form345 2024q1` off-sandbox, years of Form 3/4/5 per download; run `--limit 100` first (parser follows the published spec but was written without live-archive access) |
| GDELT DOC 2.0 (news, keyless) | ❌ 403 via proxy | `news-refresh` off-sandbox; full-text news index since 2017 |
| OpenInsider (Form 4 screener) | ❌ 403 via proxy | redundant with EDGAR/DERA paths, not implemented |
| efdsearch.senate.gov / house clerk direct | untested here | the plan's Phase 6: direct scraping, needed for live senate data post-2021 |

**Bottom line on "live"**: real-time *insider* signals (Form 4, minutes-level
acceptance timestamps) work today off-sandbox with zero keys. Real-time
*congress* signals need one free API key (Finnhub) or a paid aggregator
(QuiverQuant, Unusual Whales); the free GitHub dataset is history-only. Nothing
in this repo fabricates data to paper over that: missing disclosure dates are
skipped and counted, staleness is warned about at ingest time, and FX rows
carry their source.

## 5. Research context: what similar strategies actually found

Read before expecting miracles; each finding shaped a feature in this repo:

- **Pre-STOCK Act senators beat the market by ~12pp/yr** (Ziobrowski et al.,
  2004) and copyable abnormal returns (~4.9%/quarter) existed, the origin of
  the whole idea.
- **Post-STOCK Act (2012), the edge is far weaker**: a 2022 Dartmouth study
  found "no evidence of stock-picking prowess" in congressional trades. The
  significance gates (clustered t>2, walk-forward, bootstrap p-value) exist
  because the null hypothesis is now the smart default.
- **The NANC/KRUZ ETFs** (live products copying Democrat/Republican trades
  since 2023) have NOT outperformed risk-adjusted, but the party cohorts hold
  wildly different books (NANC tech-heavy ~27%/yr vs KRUZ industrials
  ~13%/yr in their first years) → this repo's party enrichment
  (`enrich-legislators`, runs automatically after `ingest-senate`) and the
  `decay_party.png` cohort chart.
- **"Copy everyone" is what the ETFs do** → the `Copy ALL filers` baseline
  curve: if top-N selection can't beat it, the ranking layer adds nothing.
- **Insiders' exits carry information too** → `exit_on_filer_sell`
  (copy-the-exit) variant in every report.
- **Disclosure speed is the whole game** → `decay_lag_speed.png` splits fast
  (≤15 bd) vs slow filers.

Sources: [NANC/KRUZ study (Economics Letters 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0165176525001004),
[ETF.com NANC vs KRUZ](https://www.etf.com/sections/etf-basics/nanc-vs-kruz-battle-congress-stock-trackers),
[Morningstar on the two ETFs](https://www.morningstar.com/funds/2-etfs-that-track-congressional-stock-trades),
[US News overview](https://money.usnews.com/investing/articles/track-congressional-stock-picks-with-these-new-etfs).

## 6. Fee arithmetic worth internalizing

A $10k USD position round trip:

| | ibkr_ca | questrade | questrade_gambit | wealthsimple |
|---|---|---|---|---|
| commissions | ~$2 | $0 | $0 | $0 |
| FX (both legs) | ~$6 | ~$300 | ~$8 | ~$300 |
| slippage (10bps/side) | $20 | $20 | $20 | $20 |
| SEC sell fee | ~$0.28 | ~$0.28 | ~$0.28 | ~$0.28 |
| **total** | **~$28 (0.28%)** | **~$320 (3.2%)** | **~$28 (0.28%)** | **~$320 (3.2%)** |

A congress-copy edge of 1-2% per position does not survive 3.2% round-trip
costs. It can survive 0.28%. Platform choice **is** the strategy decision for a
Canadian: hence `ibkr_ca` as default and the gambit preset for Questrade users.
(If you convert once and keep a USD balance, per-trade FX drops toward zero
the presets model the conservative convert-every-trade case.)
