"""Form 4 (insider) ownershipDocument XML parser.

Signal mapping (plan §7): transaction code P = open-market purchase (the
high-signal buy), A = award/acquisition (optional buy), S = sale. All other
codes (F, M, G, C, ...) are stored with side=NULL so they never enter a
backtest but remain queryable.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

BUY_CODES = {"P"}
OPTIONAL_BUY_CODES = {"A"}
SELL_CODES = {"S"}


@dataclass
class Form4Tx:
    code: str
    shares: float | None
    price: float | None
    date: str | None                 # transaction date, YYYY-MM-DD
    acquired_disposed: str | None    # A or D
    security_title: str | None = None

    @property
    def side(self) -> str | None:
        if self.code in BUY_CODES:
            return "buy"
        if self.code in SELL_CODES:
            return "sell"
        return None


@dataclass
class Form4Filing:
    issuer_cik: str | None
    issuer_symbol: str | None
    issuer_name: str | None
    owner_name: str | None
    owner_cik: str | None
    relationship: dict = field(default_factory=dict)
    transactions: list[Form4Tx] = field(default_factory=list)


def _text(node: ET.Element | None, path: str) -> str | None:
    if node is None:
        return None
    found = node.find(path)
    if found is None or found.text is None:
        return None
    return found.text.strip()


def _value(node: ET.Element | None, path: str) -> str | None:
    return _text(node, f"{path}/value") or _text(node, path)


def _num(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_form4(xml_text: str) -> Form4Filing:
    root = ET.fromstring(xml_text)

    issuer = root.find("issuer")
    owner = root.find("reportingOwner")
    rel = owner.find("reportingOwnerRelationship") if owner is not None else None

    relationship = {}
    if rel is not None:
        for tag, key in (
            ("isDirector", "director"),
            ("isOfficer", "officer"),
            ("isTenPercentOwner", "ten_percent_owner"),
        ):
            v = _text(rel, tag)
            relationship[key] = v in ("1", "true")
        title = _text(rel, "officerTitle")
        if title:
            relationship["officer_title"] = title

    filing = Form4Filing(
        issuer_cik=_text(issuer, "issuerCik"),
        issuer_symbol=(_text(issuer, "issuerTradingSymbol") or "").upper() or None,
        issuer_name=_text(issuer, "issuerName"),
        owner_name=_text(owner, "reportingOwnerId/rptOwnerName") if owner is not None else None,
        owner_cik=_text(owner, "reportingOwnerId/rptOwnerCik") if owner is not None else None,
        relationship=relationship,
    )

    table = root.find("nonDerivativeTable")
    if table is not None:
        for tx in table.findall("nonDerivativeTransaction"):
            amounts = tx.find("transactionAmounts")
            filing.transactions.append(
                Form4Tx(
                    code=(_text(tx, "transactionCoding/transactionCode") or "").upper(),
                    shares=_num(_value(amounts, "transactionShares")),
                    price=_num(_value(amounts, "transactionPricePerShare")),
                    date=_value(tx, "transactionDate"),
                    acquired_disposed=_value(amounts, "transactionAcquiredDisposedCode"),
                    security_title=_value(tx, "securityTitle"),
                )
            )
    return filing


def store_form4(conn, filing: Form4Filing, accession: str, filing_datetime: str,
                raw_ref: str | None = None) -> int:
    """Normalize a parsed Form 4 into filers/trades. Returns rows inserted."""
    from copytrade.store import db as store

    if not filing.owner_name or not filing.issuer_symbol:
        return 0
    meta = {"issuer": filing.issuer_name, "issuer_cik": filing.issuer_cik,
            "owner_cik": filing.owner_cik, **filing.relationship}
    filer_id = store.upsert_filer(conn, filing.owner_name, "insider", meta)

    inserted = 0
    for i, tx in enumerate(filing.transactions):
        if not tx.date:
            continue
        new = store.insert_trade(
            conn,
            filer_id=filer_id,
            source="edgar_form4",
            source_uid=f"{accession}:{i}",
            ticker=filing.issuer_symbol,
            side=tx.side,
            tx_code=tx.code,
            instrument="stock",
            transaction_date=tx.date,
            filing_datetime=filing_datetime,
            shares=tx.shares,
            price=tx.price,
            amount_low=(tx.shares or 0) * (tx.price or 0) or None,
            amount_high=(tx.shares or 0) * (tx.price or 0) or None,
            raw_ref=raw_ref,
        )
        inserted += int(new)
    conn.commit()
    return inserted
