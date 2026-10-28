PY := .venv/bin/python

.PHONY: setup demo real backfill prices report rank allocate serve poll test clean

setup:
	python3 -m venv .venv || uv venv .venv
	.venv/bin/pip install -r requirements.txt || uv pip install --python $(PY) -r requirements.txt

# offline end-to-end demo: synthetic market + filers with known skill/lags.
# uses a SEPARATE db so real ingested data is never mixed with demo data
demo:
	COPYTRADE_DB_PATH=data/demo.db COPYTRADE_REPORTS_DIR=reports_demo $(PY) -m copytrade demo
	COPYTRADE_DB_PATH=data/demo.db COPYTRADE_REPORTS_DIR=reports_demo $(PY) -m copytrade report

# REAL pipeline: senate PTRs (real disclosure dates) + USD/CAD FX + prices + report
# needs: git egress for ingest-senate/fx, general egress for yfinance prices
real:
	$(PY) -m copytrade ingest-senate
	$(PY) -m copytrade fx-refresh
	$(PY) -m copytrade prices --start 2012-01-01
	$(PY) -m copytrade report
	$(PY) -m copytrade allocate

# additional real sources (EDGAR needs direct sec.gov; congress needs Finnhub key)
backfill:
	$(PY) -m copytrade backfill-form4 --days 30
	$(PY) -m copytrade backfill-congress --years 2

prices:
	$(PY) -m copytrade prices

report:
	$(PY) -m copytrade report

rank:
	$(PY) -m copytrade rank

allocate:
	$(PY) -m copytrade allocate

serve:
	$(PY) -m copytrade serve --host 127.0.0.1 --port 8000

poll:
	$(PY) -m copytrade poll

test:
	$(PY) -m pytest -q

clean:
	rm -rf data reports_demo reports/*.png reports/*.csv reports/README.md
