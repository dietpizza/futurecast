"""Where deltarisk keeps its runtime files.

Everything the app reads or writes at runtime (config overrides, tracked coins, the candle,
funding and model caches) lives under one home folder: ~/.deltarisk, or $DELTARISK_HOME when set.
Relative paths given to deltarisk are taken relative to that folder.

The app was called FutureCast until 2026-10-18: $FUTURECAST_HOME is still honoured, and an old
~/.futurecast folder is moved to ~/.deltarisk the first time the new default is needed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_HOME = "~/.deltarisk"
LEGACY_HOME = "~/.futurecast"  # before the rename


def home() -> Path:
    """The deltarisk home folder ($DELTARISK_HOME, then the legacy $FUTURECAST_HOME, default
    ~/.deltarisk, which takes over a legacy ~/.futurecast on first use)."""
    configured = os.environ.get("DELTARISK_HOME") or os.environ.get("FUTURECAST_HOME")
    if configured:
        return Path(configured).expanduser()
    default = Path(DEFAULT_HOME).expanduser()
    _adopt_legacy_home(default, Path(LEGACY_HOME).expanduser())
    return default


def _adopt_legacy_home(default: Path, legacy: Path) -> None:
    """Move the pre-rename home folder to the new default, once, if only the old one exists."""
    if default.exists() or not legacy.is_dir():
        return
    try:
        os.replace(legacy, default)
        logger.info(f"Moved {legacy} to {default} (the app is now called DeltaRisk)")
    except OSError as err:
        logger.warning(f"Could not move {legacy} to {default}: {err}")


def resolve(path: str | Path) -> Path:
    """`path` itself when absolute, otherwise the same path under the home folder."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else home() / p
