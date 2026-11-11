import numpy as np
import pandas as pd
import pytest

from coupling import clean
from coupling import config as C


def local_hourly(start, end, tz="Europe/Berlin", seed=0):
    """Hourly series indexed in local time, the shape entsoe-py returns before the 15-min MTU."""
    idx = pd.date_range(pd.Timestamp(start, tz=tz), pd.Timestamp(end, tz=tz), freq="h", inclusive="left")
    return pd.Series(np.random.default_rng(seed).normal(80, 20, len(idx)), index=idx)


@pytest.mark.parametrize("day, n_hours", [("2024-10-27", 25), ("2025-03-30", 23)])
def test_dst_days_have_correct_length_and_no_gaps(day, n_hours):
    start = pd.Timestamp(day, tz=C.MARKET_TZ)
    end = start + pd.DateOffset(days=1)  # calendar day; Timedelta(days=1) would be 24 absolute hours
    grid = clean.hourly_grid(start, end)
    assert len(grid) == n_hours
    s = local_hourly(day, (pd.Timestamp(day) + pd.Timedelta(days=1)).date())
    hourly, stats = clean.prices_to_hourly(s, grid)
    assert stats["n_missing"] == 0
    check = clean.day_length_check(hourly.to_frame("X"))
    assert check.loc["X", "days_incomplete"] == 0


def test_quarter_hours_average_to_hour_and_incomplete_hours_are_dropped():
    t0 = C.MTU15_GOLIVE
    idx = pd.date_range(t0, t0 + pd.Timedelta(hours=3), freq="15min", inclusive="left")
    s = pd.Series([10, 20, 30, 40, 1, 1, 1, 1, 5, 5, 5, 5], index=idx, dtype=float)
    s = s.drop(idx[5])  # hour 2 loses one quarter -> incomplete
    grid = clean.hourly_grid(t0, t0 + pd.Timedelta(hours=3))
    hourly, stats = clean.prices_to_hourly(s, grid)
    assert hourly.iloc[0] == pytest.approx(25.0)
    assert np.isnan(hourly.iloc[1])
    assert hourly.iloc[2] == pytest.approx(5.0)
    assert stats["n_incomplete_hours"] == 1


def test_conflicting_duplicate_voids_hour_but_exact_duplicate_does_not():
    t0 = C.MTU15_GOLIVE
    idx = pd.date_range(t0, t0 + pd.Timedelta(hours=2), freq="15min", inclusive="left")
    s = pd.Series(np.arange(8, dtype=float), index=idx)
    s = pd.concat([s, pd.Series([0.0], index=[idx[0]]),   # exact duplicate of quarter 0
                   pd.Series([99.0], index=[idx[4]])])    # conflicting value for quarter 4
    grid = clean.hourly_grid(t0, t0 + pd.Timedelta(hours=2))
    hourly, stats = clean.prices_to_hourly(s, grid)
    assert hourly.iloc[0] == pytest.approx(1.5)
    assert np.isnan(hourly.iloc[1])
    assert stats["n_exact_duplicates"] == 1
    assert stats["n_conflicting_stamps"] == 1


def test_switch_day_mixes_hourly_and_quarter_hour_resolution():
    before = pd.Series([50.0], index=[C.MTU15_GOLIVE - pd.Timedelta(hours=1)])
    after = pd.Series([60.0, 62.0, 64.0, 66.0],
                      index=pd.date_range(C.MTU15_GOLIVE, periods=4, freq="15min"))
    grid = clean.hourly_grid(C.MTU15_GOLIVE - pd.Timedelta(hours=1), C.MTU15_GOLIVE + pd.Timedelta(hours=1))
    hourly, _ = clean.prices_to_hourly(pd.concat([before, after]), grid)
    assert hourly.tolist() == [50.0, 63.0]


def test_negative_prices_are_kept_and_counted():
    s = local_hourly("2024-05-01", "2024-05-02")
    s.iloc[10:14] = -5.0
    grid = clean.hourly_grid(pd.Timestamp("2024-05-01", tz=C.MARKET_TZ), pd.Timestamp("2024-05-02", tz=C.MARKET_TZ))
    hourly, stats = clean.prices_to_hourly(s, grid)
    assert stats["n_negative"] == 4
    assert (hourly < 0).sum() == 4
