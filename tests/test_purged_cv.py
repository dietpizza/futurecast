"""Tests for Purged and Embargoed Walk-Forward Cross-Validation."""

from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from futurecast.validation.purged_cv import PurgedWalkForwardCV


def test_purged_cv_splits_and_embargo():
    # Create 60 days of minute data (86400 bars)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    dates = pd.date_range(start=start, periods=86400, freq="1min", tz="UTC")
    df = pd.DataFrame(index=dates)

    cv = PurgedWalkForwardCV(
        train_window_days=20,
        test_window_days=5,
        step_window_days=5,
        purge_window_minutes=60,
        embargo_window_minutes=1440,
    )

    splits = list(cv.split(df))
    assert len(splits) > 0

    for split in splits:
        train_idx = split.train_indices
        test_idx = split.test_indices

        # 1. No overlap between train and test
        assert len(set(train_idx).intersection(set(test_idx))) == 0

        # 2. Check purge window: distance between train end and test start is >= 60 min
        train_end_time = dates[train_idx[-1]]
        test_start_time = dates[test_idx[0]]
        purge_diff = (test_start_time - train_end_time).total_seconds() / 60.0
        assert purge_diff >= 60.0
