"""Source abstraction for the scraping engine.

Every disclosure feed: US insiders, US House, US Senate, Canadian MPs,
funds: implements one small interface so the scheduler can treat them
uniformly. A source is responsible only for producing NEW normalized trades
since it last ran; dedup (DB UNIQUE constraints), alerting, cadence, backoff
and resume-state are the scheduler's job, not each source's.

Design rules the interface enforces:
  * run_once must be idempotent: re-running never double-inserts (dedup is
    at the DB layer, on source+source_uid).
  * a source declares its natural cadence but the scheduler owns timing.
  * a source classifies itself: `authoritative` (source-of-record, complete
    coverage: the backbone) vs `aggregator` (convenient but possibly
    incomplete, e.g. symbol-keyed). The scheduler prefers authoritative feeds
    and treats aggregators as cross-checks so an aggregator's blind spots
    never silently define the universe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ScrapeResult:
    source: str
    inserted: int = 0
    deduped: int = 0
    skipped: int = 0
    alertable: list[dict] = field(default_factory=list)  # new trades worth alerting on
    note: str | None = None
    error: str | None = None
    #: The run WORKED but is incomplete, e.g. most PTRs came back unreadable.
    #: Deliberately NOT `error`: the scheduler reads `error` as "this run
    #: failed", which suppresses alert dispatch and counts toward disabling the
    #: source after MAX_CONSECUTIVE_FAILURES. A run that ingested real trades
    #: must still alert on them and must not push its source toward being
    #: switched off, while still saying loudly that its counts are a floor.
    degraded: str | None = None

    def __add__(self, other: "ScrapeResult") -> "ScrapeResult":
        return ScrapeResult(
            source=self.source,
            inserted=self.inserted + other.inserted,
            deduped=self.deduped + other.deduped,
            skipped=self.skipped + other.skipped,
            alertable=self.alertable + other.alertable,
            note=other.note or self.note,
            error=other.error or self.error,
            degraded=other.degraded or self.degraded,
        )


@runtime_checkable
class Source(Protocol):
    name: str
    kind: str            # congress | insider | fund | congress_ca
    cadence_secs: int
    authoritative: bool  # True = source of record (backbone); False = aggregator

    async def run_once(self, conn, settings) -> ScrapeResult:
        """Fetch new disclosures since last run, store them, return counts +
        anything alert-worthy. Must be safe to call repeatedly."""
        ...


class BaseSource:
    """Convenience base: sane defaults + a state-key helper for resume."""

    name: str = "base"
    kind: str = "congress"
    cadence_secs: int = 3600
    authoritative: bool = False

    def state_key(self, suffix: str = "cursor") -> str:
        return f"scrape:{self.name}:{suffix}"

    async def run_once(self, conn, settings) -> ScrapeResult:  # pragma: no cover
        raise NotImplementedError
