"""Webhook alert poster (Discord-compatible by default, or generic JSON).

Alerts are informational only, no auto-execution in the MVP (plan §5).
"""

from __future__ import annotations

import httpx


def format_alert(trade: dict, filer_stats: dict | None = None) -> str:
    lag = trade.get("lag_days")
    lines = [
        f"**{trade.get('filer_name', '?')}** ({trade.get('kind', '?')}) "
        f"{(trade.get('side') or '?').upper()} **{trade.get('ticker', '?')}**",
        f"filed: {trade.get('filing_datetime')}  |  transacted: {trade.get('transaction_date')}"
        + (f"  |  lag: {lag}d" if lag is not None else ""),
    ]
    lo, hi = trade.get("amount_low"), trade.get("amount_high")
    if lo or hi:
        lines.append(f"size: ${lo:,.0f}–${hi:,.0f}" if lo != hi else f"size: ${lo:,.0f}")
    if filer_stats:
        lines.append(
            "trailing lag-alpha: {a:+.2%} (t={t:.1f}, n={n}, hit {h:.0%})".format(
                a=filer_stats.get("lag_alpha", 0), t=filer_stats.get("tstat", 0),
                n=filer_stats.get("n_trades", 0), h=filer_stats.get("hit_rate", 0),
            )
        )
    return "\n".join(lines)


async def send_alert(webhook_url: str, text: str, fmt: str = "discord") -> bool:
    if not webhook_url:
        return False
    payload = {"content": text} if fmt == "discord" else {"text": text}
    async with httpx.AsyncClient(timeout=15) as client:
        for backoff in (2, 4, 8, None):
            resp = await client.post(webhook_url, json=payload)
            if resp.status_code in (200, 204):
                return True
            if resp.status_code == 429 and backoff:
                import asyncio

                await asyncio.sleep(backoff)
                continue
            return False
    return False
