"""Convergence metrics, aligned with ACER's market-monitoring definition.

ACER counts an hour as fully converged when the price difference is below 1 EUR/MWh. We use the same
definition (TOL_ACER) and add an exact-convergence check (TOL_EXACT) as a sensitivity test.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from . import config as C


def converged(spread: pd.Series | pd.DataFrame, tol: float = C.TOL_ACER):
    """1.0 if |spread| < tol, 0.0 otherwise, NaN where the spread is missing."""
    return (spread.abs() < tol).astype(float).where(spread.notna())


def border_start(flows: pd.DataFrame, min_abs_mw: float = 1.0) -> pd.Series:
    """First market-time month in which the border carries physical flow.

    Interconnectors commissioned during the sample (ALEGrO, COBRA, NordLink) are therefore not
    analysed as borders before they existed.
    """
    starts = {}
    for name in flows.columns:
        active = flows[name].abs() > min_abs_mw
        if not active.any():
            starts[name] = pd.NaT
            continue
        first = active[active].index[0].tz_convert(C.MARKET_TZ)
        starts[name] = pd.Timestamp(year=first.year, month=first.month, day=1, tz=C.MARKET_TZ)
    return pd.Series(starts, name="border_start")


def market_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_convert(C.MARKET_TZ)
    f = pd.DataFrame(index=index)
    f["date"] = local.normalize().tz_localize(None)
    f["hour"] = local.hour
    f["dow"] = local.dayofweek
    f["month"] = local.month
    f["year"] = local.year
    f["quarter"] = local.tz_localize(None).to_period("Q")
    f["crisis"] = (index >= C.CRISIS_START) & (index < C.CRISIS_END)
    f["winter_peak"] = local.month.isin(C.WINTER_MONTHS) & local.hour.isin(C.PEAK_HOURS) & (local.dayofweek < 5)
    f["post_mtu15"] = index >= C.MTU15_GOLIVE
    return f


def border_panel(prices: pd.DataFrame, borders=C.BORDERS, starts: pd.Series | None = None) -> pd.DataFrame:
    """Long border-hour panel: spread (a - b), |spread|, ACER and exact convergence flags."""
    frames = []
    for b in borders:
        spread = prices[b.a] - prices[b.b]
        if starts is not None and pd.notna(starts.get(b.name)):
            spread = spread.where(spread.index >= starts[b.name])
        frames.append(pd.DataFrame({
            "border": b.name,
            "spread": spread,
            "abs_spread": spread.abs(),
            "conv": converged(spread, C.TOL_ACER),
            "conv_exact": converged(spread, C.TOL_EXACT),
        }))
    panel = pd.concat(frames)
    panel.index.name = "ts_utc"
    return panel.dropna(subset=["spread"])


def share_by(panel: pd.DataFrame, by: list[str], col: str = "conv") -> pd.DataFrame:
    """Convergence share (and n) per group. `by` may include columns of market_time_features."""
    feats = market_time_features(panel.index)
    df = pd.concat([panel.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    return df.groupby(by, observed=True)[col].agg(share="mean", n="size").reset_index()


def pairwise_matrix(prices: pd.DataFrame, tol: float = C.TOL_ACER, mask: pd.Series | None = None) -> pd.DataFrame:
    """Share of hours with |p_i - p_j| < tol for every zone pair. Both prices must be valid."""
    p = prices if mask is None else prices[mask]
    zones = list(p.columns)
    m = pd.DataFrame(np.eye(len(zones)), index=zones, columns=zones)
    for a, b in combinations(zones, 2):
        m.loc[a, b] = m.loc[b, a] = converged(p[a] - p[b], tol).mean()
    return m


def region_convergence(prices: pd.DataFrame, zones: list[str], tol: float = C.TOL_ACER) -> pd.Series:
    """ACER regional metric: max - min price across `zones` below tol. NaN if any zone is missing."""
    p = prices[zones]
    rng = p.max(axis=1) - p.min(axis=1)
    return (rng < tol).astype(float).where(p.notna().all(axis=1))


def res_share(fundamentals: pd.DataFrame, zone: str) -> pd.Series:
    f = fundamentals[zone]
    return (f["wind_fc"].fillna(0) + f["solar_fc"].fillna(0)) / f["load_fc"]


def high_res_flag(fundamentals: pd.DataFrame, a: str, b: str, q: float = 0.9) -> pd.Series:
    """Hours in the top decile of combined forecast RES share for zones a and b, computed per calendar
    year so that capacity growth doesn't shift every 'high-RES' hour into recent years."""
    fa, fb = fundamentals[a], fundamentals[b]
    share = (fa["wind_fc"].fillna(0) + fa["solar_fc"].fillna(0) + fb["wind_fc"].fillna(0)
             + fb["solar_fc"].fillna(0)) / (fa["load_fc"] + fb["load_fc"])
    year = share.index.tz_convert(C.MARKET_TZ).year
    thresh = share.groupby(year).transform(lambda x: x.quantile(q))
    return (share >= thresh).where(share.notna())


def quarter_hour_convergence(prices_15min: pd.DataFrame, a: str, b: str, tol: float = C.TOL_ACER) -> pd.DataFrame:
    """For each delivery hour after go-live, compare convergence of the hourly mean with
    convergence in all four quarter-hours. The hour is NaN unless all 8 prices are present."""
    q = prices_15min[[a, b]]
    hour = q.index.floor("h")
    complete = q.notna().all(axis=1).groupby(hour).sum() == 4
    spread_q = q[a] - q[b]
    hourly_spread = spread_q.groupby(hour).mean()
    out = pd.DataFrame({
        "conv_hourly_mean": converged(hourly_spread, tol),
        "conv_all_quarters": (spread_q.abs() < tol).groupby(hour).all().astype(float),
        "n_quarters_conv": (spread_q.abs() < tol).groupby(hour).sum(),
        "intrahour_spread_range": spread_q.groupby(hour).max() - spread_q.groupby(hour).min(),
    })
    return out[complete]
