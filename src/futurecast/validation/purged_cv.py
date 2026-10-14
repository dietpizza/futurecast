"""Purged and Embargoed Walk-Forward Cross-Validation Splitter."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Generator, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitInfo:
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_indices: np.ndarray
    test_indices: np.ndarray


class PurgedWalkForwardCV:
    """Purged & Embargoed Walk-Forward Time-Series Splitter.
    
    Prevents both forward-target leakage (via Purging) and autoregressive
    correlation leakage (via Embargoing).
    """

    def __init__(
        self,
        train_window_days: int = 120,
        test_window_days: int = 30,
        step_window_days: int = 30,
        purge_window_minutes: int = 60,
        embargo_window_minutes: int = 1440,
    ) -> None:
        self.train_window = timedelta(days=train_window_days)
        self.test_window = timedelta(days=test_window_days)
        self.step_window = timedelta(days=step_window_days)
        self.purge_window = timedelta(minutes=purge_window_minutes)
        self.embargo_window = timedelta(minutes=embargo_window_minutes)

    def split(self, df: pd.DataFrame) -> Generator[SplitInfo, None, None]:
        """Yield (train_indices, test_indices, split_info) for each walk-forward fold."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")

        if len(df) == 0:
            return

        timestamps = df.index
        total_start = timestamps.min()
        total_end = timestamps.max()

        if pd.isna(total_start) or pd.isna(total_end) or total_start >= total_end:
            return

        fold = 1
        curr_train_start = total_start

        while True:
            if pd.isna(curr_train_start) or curr_train_start >= total_end:
                break

            raw_train_end = curr_train_start + self.train_window
            # Purge the boundary right before the test set
            purged_train_end = raw_train_end - self.purge_window

            test_start = raw_train_end
            test_end = test_start + self.test_window

            if test_end > total_end:
                # If test_start exceeds total_end or remaining test slice is too small (< half of test window), stop
                if test_start >= total_end or (total_end - test_start) < (self.test_window / 2):
                    break
                test_end = total_end

            # Train indices: [curr_train_start, purged_train_end]
            train_mask = (timestamps >= curr_train_start) & (timestamps <= purged_train_end)
            # Test indices: [test_start, test_end]
            test_mask = (timestamps >= test_start) & (timestamps <= test_end)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield SplitInfo(
                    fold=fold,
                    train_start=curr_train_start,
                    train_end=purged_train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_indices=train_idx,
                    test_indices=test_idx,
                )
                fold += 1

            # Step forward
            curr_train_start += self.step_window

            # If we've reached the end
            if curr_train_start + self.train_window >= total_end:
                break
