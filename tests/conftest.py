from pathlib import Path

import pytest

from copytrade.store import db as store

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def conn():
    c = store.connect(":memory:")
    store.run_migrations(c, ROOT / "migrations")
    yield c
    c.close()
