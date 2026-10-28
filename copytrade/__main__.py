import sys

from copytrade.cli import app

# Reports use →, ≥, ² and em-dashes. A Windows console defaults to cp1252, which
# renders those as mojibake (or raises on redirect to a file). Python 3.7+ can
# retarget the stream in place; guarded because a stream may already be wrapped
# or replaced (pytest's capture, a pipe) and need not support reconfigure().
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

app()
