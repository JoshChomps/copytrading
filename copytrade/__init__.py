"""CopyTrade: disclosure-based copy-trading research pipeline.

Signals come from mandatory public disclosures (Congress PTRs, SEC Form 4,
13F). Every backtest executes at the first market close AFTER the filing
timestamp: point-in-time discipline is non-negotiable.
"""

__version__ = "0.1.0"
