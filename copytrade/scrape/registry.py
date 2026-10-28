"""Which sources are enabled, and the watchlist the scheduler alerts against.

Enable/disable via config (`scrape_sources` = list of source names, or the
sentinel "auto"). "auto" turns on every source whose prerequisites are met
(keys present, universe file exists, etc.) so a fresh clone scrapes the
free authoritative feeds out of the box and lights up aggregators only once
their keys are configured.
"""

from __future__ import annotations

from copytrade.scrape.base import Source
from copytrade.scrape.sources import (
    CanadaCIECSource,
    FinnhubCongressSource,
    Form4Source,
    HouseClerkSource,
    SenateEFDSource,
)


def all_sources() -> dict[str, type]:
    # QuiverCongressSource imported lazily: ingest.quiver itself imports
    # scrape.base, so a module-level import here would be circular
    from copytrade.ingest.quiver import QuiverCongressSource

    return {
        s.name: s for s in (
            Form4Source, HouseClerkSource, SenateEFDSource,
            CanadaCIECSource, FinnhubCongressSource, QuiverCongressSource,
        )
    }


def _prereqs_met(name: str, settings) -> bool:
    if name == "finnhub_congress":
        return bool(settings.finnhub_api_key)
    if name == "quiver_congress":
        return bool(getattr(settings, "quiver_api_key", ""))
    return True  # authoritative feeds need only network + a contact UA


def build_sources(settings) -> list[Source]:
    registry = all_sources()
    requested = getattr(settings, "scrape_sources", ["auto"]) or ["auto"]
    if requested == ["auto"] or requested == "auto":
        names = [n for n in registry if _prereqs_met(n, settings)]
    else:
        names = [n for n in requested if n in registry]
    return [registry[n]() for n in names]


def build_watchlist(settings) -> dict:
    """Top-ranked filers to alert on (from the last generated ranking)."""
    from copytrade.poll import top_filer_stats

    return top_filer_stats(settings)
