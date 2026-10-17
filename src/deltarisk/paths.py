"""Where futurecast keeps its runtime files.

Everything the app reads or writes at runtime (config overrides, tracked coins, the candle
cache) lives under one home folder: ~/.futurecast, or $FUTURECAST_HOME
when set. Relative paths given to futurecast are taken relative to that folder.
"""

from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    """The futurecast home folder ($FUTURECAST_HOME, default ~/.futurecast)."""
    return Path(os.environ.get("FUTURECAST_HOME") or "~/.futurecast").expanduser()


def resolve(path: str | Path) -> Path:
    """`path` itself when absolute, otherwise the same path under the home folder."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else home() / p
