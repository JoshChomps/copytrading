"""Constant-scraping engine: a registry of disclosure sources driven by one
scheduler. See docs/SCRAPING.md for the architecture and the reasoning behind
'authoritative feeds are the backbone, aggregators are enrichment'."""

from copytrade.scrape.base import ScrapeResult, Source
from copytrade.scrape.registry import build_sources
from copytrade.scrape.scheduler import run_scheduler

__all__ = ["Source", "ScrapeResult", "build_sources", "run_scheduler"]
