"""Makes `tests` a REGULAR package, which is load-bearing on this machine.

Five test modules do `from tests.conftest import FIXTURES`. Without this file
`tests` is only an implicit namespace package, and Python resolves a namespace
portion ONLY when no regular package of that name is found anywhere on
sys.path: path order does not save it. Four unrelated installs
(`py_clob_client`, `py_order_utils`, `py_builder_signing_sdk`,
`pandas-datareader`) each ship their own top-level `tests/__init__.py` into
site-packages, so `import tests.conftest` resolved to the Polymarket client's
test package and the whole suite died at collection:

    ModuleNotFoundError: No module named 'tests.conftest'
    ERROR tests/{test_parsers,test_pipeline,test_report_readme,
                 test_senate_watcher,test_legislators}.py

An empty regular package here wins on sys.path order (cwd is first) and pins
the resolution to this repo regardless of what else is installed alongside it.
"""
