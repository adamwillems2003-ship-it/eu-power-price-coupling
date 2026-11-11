"""Turn cached raw ENTSO-E data into analysis-ready hourly UTC panels.

Rules
-----
* Time: everything is converted to UTC, which removes DST ambiguity. Market time (Europe/Brussels) is
  derived only for features such as hour-of-day and delivery day.
* Prices: an hour is valid only when its delivery period is complete. That means 1 hourly price before
  the 15-minute MTU go-live and 4 quarter-hour prices after it. The hourly price after go-live is the
  plain mean of the 4 quarter-hours (each quarter carries equal energy weight in an hour).
* Duplicates: exact duplicate stamps are dropped. Stamps with conflicting values set the hour to NaN
  and are counted in the quality report.
* Fundamentals (load, wind, solar, flows, NTC) are averaged over whatever sub-hourly values exist,
  because their native resolution differs by TSO and over time.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger(__name__)

MTU15_UTC = C.MTU15_GOLIVE.tz_convert("UTC")


def read_raw(dataset: str, key: str) -> pd.DataFrame:
    files = sorted((C.RAW_DIR / dataset).glob(f"{key}_*.parquet"))
    parts = [df for df in (pd.read_parquet(f) for f in files) if len(df)]
    if not parts:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="ts_utc"))
    df = pd.concat(parts)
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    return df.sort_index()


def hourly_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left",
                         name="ts_utc")


def dedupe(s: pd.Series) -> tuple[pd.Series, int, int]:
    """Drop exact duplicates. Stamps with conflicting values become NaN.

    Returns (series, n_exact_dupes, n_conflicting_stamps).
    """
    s = s.sort_index()
    dup_mask = s.index.duplicated(keep=False)
    if not dup_mask.any():
        return s, 0, 0
    frame = s.rename("v").reset_index()
    ts_col = frame.columns[0]
    n_exact = int(frame.duplicated().sum())
    frame = frame.drop_duplicates()
    counts = frame.groupby(ts_col)["v"].transform("size")
    n_conflict = int(frame.loc[counts > 1, ts_col].nunique())
    frame.loc[counts > 1, "v"] = np.nan
    frame = frame.drop_duplicates(subset=ts_col)
    out = frame.set_index(ts_col)["v"].rename(s.name)
    return out, n_exact, n_conflict


def prices_to_hourly(s: pd.Series, grid: pd.DatetimeIndex) -> tuple[pd.Series, dict]:
    """Strict aggregation of one zone's day-ahead prices onto the hourly UTC grid."""
    s = s.astype(float)
    s.index = pd.DatetimeIndex(s.index).tz_convert("UTC")
    s, n_exact, n_conflict = dedupe(s)
    s = s[(s.index >= grid[0]) & (s.index < grid[-1] + pd.Timedelta(hours=1))]
    hour = s.index.floor("h")
    g = s.groupby(hour)
    mean = g.mean()
    n_valid = g.count()
    n_stamps = g.size()
    expected = pd.Series(np.where(mean.index >= MTU15_UTC, 4, 1), index=mean.index)
    complete = (n_valid == expected) & (n_stamps == expected)
    hourly = mean.where(complete).reindex(grid)
    stats = dict(
        n_hours=len(grid),
        n_missing=int(hourly.isna().sum()),
        n_incomplete_hours=int((~complete).sum()),
        n_exact_duplicates=n_exact,
        n_conflicting_stamps=n_conflict,
        n_negative=int((hourly < 0).sum()),
        n_zero=int((hourly == 0).sum()),
        min=float(hourly.min()),
        max=float(hourly.max()),
    )
    return hourly, stats


def quarter_hour_prices(s: pd.Series, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.Series:
    """15-minute prices after go-live, on a complete 15-minute UTC grid (gaps are NaN)."""
    s = s.astype(float)
    s.index = pd.DatetimeIndex(s.index).tz_convert("UTC")
    s, _, _ = dedupe(s)
    grid = pd.date_range(start_utc, end_utc, freq="15min", inclusive="left", name="ts_utc")
    return s.reindex(grid)


def mean_to_hourly(df: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Loose aggregation for fundamentals: hourly mean of the available sub-hourly values."""
    if df.empty:
        return pd.DataFrame(index=grid)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[~df.index.duplicated(keep="first")]
    return df.groupby(df.index.floor("h")).mean().reindex(grid)


def day_length_check(hourly: pd.DataFrame) -> pd.DataFrame:
    """Valid hours per market-time delivery day and zone, against the expected 23/24/25 (DST days)."""
    local = hourly.index.tz_convert(C.MARKET_TZ)
    day = local.normalize()
    expected = pd.Series(1, index=hourly.index).groupby(day).size()
    valid = hourly.notna().groupby(day).sum()
    bad = valid.ne(expected, axis=0)
    out = bad.sum().rename("days_incomplete").to_frame()
    out["days_total"] = len(expected)
    out["dst_days_23h"] = int((expected == 23).sum())
    out["dst_days_25h"] = int((expected == 25).sum())
    return out


def build(start: pd.Timestamp = C.SAMPLE_START, end: pd.Timestamp | None = None) -> None:
    from .fetch import default_end

    end = end or default_end()
    grid = hourly_grid(start, end)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Prices
    hourly, qh, quality = {}, {}, {}
    for zone in C.ZONES:
        raw = read_raw("prices", zone)
        s = raw.iloc[:, 0] if len(raw.columns) else pd.Series(dtype=float, index=raw.index)
        hourly[zone], quality[zone] = prices_to_hourly(s, grid)
        if end > C.MTU15_GOLIVE:
            qh[zone] = quarter_hour_prices(s, max(start, C.MTU15_GOLIVE).tz_convert("UTC"),
                                           end.tz_convert("UTC"))
    prices = pd.DataFrame(hourly)
    prices.to_parquet(C.PROCESSED_DIR / "prices_hourly.parquet")
    if qh:
        pd.DataFrame(qh).to_parquet(C.PROCESSED_DIR / "prices_15min.parquet")
    q = pd.DataFrame(quality).T.join(day_length_check(prices))
    q.index.name = "zone"
    q.to_csv(C.PROCESSED_DIR / "price_quality.csv")

    # Fundamentals: day-ahead load forecast, wind and solar forecast
    fund = {}
    for zone in C.ZONES:
        load = mean_to_hourly(read_raw("load_forecast", zone), grid)
        res = mean_to_hourly(read_raw("res_forecast", zone), grid)
        fund[(zone, "load_fc")] = load.iloc[:, 0] if load.shape[1] else pd.Series(np.nan, index=grid)
        wind_cols = [c for c in res.columns if "Wind" in c]
        solar_cols = [c for c in res.columns if "Solar" in c]
        fund[(zone, "wind_fc")] = res[wind_cols].sum(axis=1, min_count=1) if wind_cols else np.nan
        fund[(zone, "solar_fc")] = res[solar_cols].sum(axis=1, min_count=1) if solar_cols else np.nan
    fundamentals = pd.DataFrame(fund, index=grid)
    fundamentals.columns = pd.MultiIndex.from_tuples(fundamentals.columns, names=["zone", "var"])
    fundamentals.to_parquet(C.PROCESSED_DIR / "fundamentals_hourly.parquet")

    # Flows and NTC per border. Net flow is signed a->b.
    flows, ntc = {}, {}
    for border in C.BORDERS:
        ab = mean_to_hourly(read_raw("flows", f"{border.a}__{border.b}"), grid)
        ba = mean_to_hourly(read_raw("flows", f"{border.b}__{border.a}"), grid)
        ab_s = ab.iloc[:, 0] if ab.shape[1] else pd.Series(np.nan, index=grid)
        ba_s = ba.iloc[:, 0] if ba.shape[1] else pd.Series(np.nan, index=grid)
        flows[border.name] = ab_s.sub(ba_s, fill_value=0).where(ab_s.notna() | ba_s.notna())
        for a, b in ((border.a, border.b), (border.b, border.a)):
            n = mean_to_hourly(read_raw("ntc_da", f"{a}__{b}"), grid)
            ntc[f"{a}->{b}"] = n.iloc[:, 0] if n.shape[1] else pd.Series(np.nan, index=grid)
    pd.DataFrame(flows, index=grid).to_parquet(C.PROCESSED_DIR / "net_flows_hourly.parquet")
    pd.DataFrame(ntc, index=grid).to_parquet(C.PROCESSED_DIR / "ntc_hourly.parquet")
    log.info("built processed panels: %d hours x %d zones", len(grid), len(C.ZONES))


def load_processed() -> dict[str, pd.DataFrame]:
    d = C.PROCESSED_DIR
    out = {
        "prices": pd.read_parquet(d / "prices_hourly.parquet"),
        "fundamentals": pd.read_parquet(d / "fundamentals_hourly.parquet"),
        "flows": pd.read_parquet(d / "net_flows_hourly.parquet"),
        "ntc": pd.read_parquet(d / "ntc_hourly.parquet"),
        "quality": pd.read_csv(d / "price_quality.csv", index_col=0),
    }
    if (d / "prices_15min.parquet").exists():
        out["prices_15min"] = pd.read_parquet(d / "prices_15min.parquet")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build()
