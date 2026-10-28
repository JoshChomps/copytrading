"""Settings loading: config.toml provides project defaults, COPYTRADE_* env
vars (or .env) override them. API keys live only in the environment."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="COPYTRADE_", env_file=".env", extra="ignore"
    )

    # storage
    db_path: str = "data/copytrade.db"
    raw_dir: str = "data/raw"
    reports_dir: str = "reports"

    # SEC EDGAR fair access: declared UA with contact email, <=8 req/s
    edgar_user_agent: str = "copytrade-research you@example.com"
    edgar_rps: float = 8.0

    # aggregator/data keys (env only: never commit). Everything works with
    # all of these empty; each key upgrades one capability (see README table).
    finnhub_api_key: str = ""
    tiingo_api_key: str = ""      # price fallback with true adjusted closes
    quiver_api_key: str = ""      # bulk live congress (both chambers, paid)
    openfigi_api_key: str = ""    # raises CUSIP-mapping rate limits (13F)

    # alerts (Discord/Telegram-compatible webhook; empty = disabled)
    webhook_url: str = ""
    webhook_format: str = "discord"  # discord | generic

    # execution model (Canadian resident defaults; see docs/CANADA.md)
    cost_model: str = "ibkr_ca"
    display_tz: str = "America/Vancouver"  # reporting only; market logic stays ET
    trade_notional: float = 10_000.0  # per-trade stake used for share-based fees
    max_weight: float = 0.05          # cap on any single position weight
    same_day_close: bool = False      # execute same-day close if filed before cutoff
    same_day_cutoff: str = "15:30"    # ET
    amount_basis: str = "mid"         # low | mid | high (congress ranges)

    # ranking / walk-forward
    horizon: int = 63
    horizons: list[int] = [5, 21, 63, 126, 252]
    top_n: int = 10
    min_trades_rank: int = 30         # trades needed before a filer is "copyable"
    min_trades_walkforward: int = 5   # trades needed in the training window
    benchmark: str = "SPY"

    # polling cadence (seconds)
    poll_form4_secs: int = 180
    poll_congress_secs: int = 3600

    # congress universe for symbol-keyed aggregators (path to newline tickers)
    congress_universe_file: str = "config/universe.txt"

    # scraping engine: which sources to run. ["auto"] = every source whose
    # prerequisites are met (keys/universe present). Or an explicit list, e.g.
    # ["edgar_form4", "house_clerk", "finnhub_congress"]. See docs/SCRAPING.md.
    scrape_sources: list[str] = ["auto"]
    # optional Canadian material-change feed (third-party tracker export/URL);
    # Canada has no STOCK Act, so this is coarse, see docs/SCRAPING.md §Canada
    canada_ciec_json_url: str = ""


# Credentials that deliberately carry NO `COPYTRADE_` prefix because they are
# the vendor's own conventional names. pydantic-settings only reads prefixed
# keys out of .env, so without an explicit bridge these sit in the file and are
# invisible to `os.environ`: which is where broker/paper.py and the CLI read
# them. Symptom: keys present in .env, app reports them missing.
UNPREFIXED_ENV_KEYS = (
    "ALPACA_API_KEY_ID",
    "ALPACA_API_SECRET_KEY",
    "ALPACA_PAPER_BASE_URL",
)


def load_dotenv_unprefixed(env_path: str | Path = ".env") -> list[str]:
    """Export the non-`COPYTRADE_` credentials from .env into os.environ.

    A REAL environment variable always wins, this only fills gaps, so CI
    secrets and shell exports are never clobbered by a stale local file.
    Returns the names actually set, so callers can report what was picked up
    without ever touching the values.
    """
    p = Path(env_path)
    if not p.exists():
        return []
    loaded: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key in UNPREFIXED_ENV_KEYS and val and not os.environ.get(key):
            os.environ[key] = val
            loaded.append(key)
    return loaded


def load_settings(config_path: str | Path = "config.toml") -> Settings:
    """config.toml values become init kwargs, except where an env var is set
    so precedence is: env var > config.toml > code default."""
    load_dotenv_unprefixed()
    overrides: dict = {}
    p = Path(config_path)
    if p.exists():
        with open(p, "rb") as f:
            data = tomllib.load(f)
        flat: dict = {}
        for key, val in data.items():
            if isinstance(val, dict):
                flat.update(val)
            else:
                flat[key] = val
        for key, val in flat.items():
            if key in Settings.model_fields and f"COPYTRADE_{key.upper()}" not in os.environ:
                overrides[key] = val
    return Settings(**overrides)
