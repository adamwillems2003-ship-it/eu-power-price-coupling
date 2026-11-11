"""Download raw data from the ENTSO-E Transparency Platform (via entsoe-py) and gas prices from FRED.

Each (dataset, key, calendar year) is cached as parquet under data/raw/. Re-runs only download
missing years and the current, incomplete year.

The API key is read from the ENTSOE_API_KEY environment variable, or failing that from ~/.entsoe_key.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Callable

import pandas as pd
from entsoe import EntsoePandasClient, EntsoeRawClient
from entsoe.exceptions import NoMatchingDataError

from . import config as C

log = logging.getLogger(__name__)
KEY_FILE = Path.home() / ".entsoe_key"


def api_key() -> str:
    key = os.environ.get("ENTSOE_API_KEY")
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
    if not key:
        raise SystemExit(
            "No ENTSO-E API key: set ENTSOE_API_KEY or write the token to ~/.entsoe_key "
            "(request one at transparency.entsoe.eu -> My account settings, then email "
            "transparency@entsoe.eu with subject 'Restful API access')."
        )
    return key


def default_end() -> pd.Timestamp:
    """First day of the current month, so the sample ends on the last complete month."""
    now = pd.Timestamp.now(tz=C.MARKET_TZ)
    return pd.Timestamp(year=now.year, month=now.month, day=1, tz=C.MARKET_TZ)


def year_chunks(start: pd.Timestamp, end: pd.Timestamp):
    """Yield [s, e) chunks aligned to calendar years in market time."""
    for year in range(start.year, end.year + 1):
        s = max(start, pd.Timestamp(year=year, month=1, day=1, tz=C.MARKET_TZ))
        e = min(end, pd.Timestamp(year=year + 1, month=1, day=1, tz=C.MARKET_TZ))
        if s < e:
            yield year, s, e


def _to_frame(obj: pd.Series | pd.DataFrame, name: str) -> pd.DataFrame:
    df = obj.to_frame(name) if isinstance(obj, pd.Series) else obj.copy()
    df.index = df.index.tz_convert("UTC")
    df.index.name = "ts_utc"
    df.columns = [str(c) if not isinstance(c, tuple) else "|".join(map(str, c)) for c in df.columns]
    return df


def cached_yearly(dataset: str, key: str, start: pd.Timestamp, end: pd.Timestamp,
                  query: Callable[[pd.Timestamp, pd.Timestamp], pd.Series | pd.DataFrame],
                  force: bool = False) -> pd.DataFrame:
    """Run `query` once per calendar year and cache each year's result. Empty results are cached too."""
    out_dir = C.RAW_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    current_year = pd.Timestamp.now(tz=C.MARKET_TZ).year
    parts = []
    for year, s, e in year_chunks(start, end):
        path = out_dir / f"{key}_{year}.parquet"
        if path.exists() and not force and year != current_year:
            parts.append(pd.read_parquet(path))
            continue
        try:
            df = _to_frame(query(s, e), dataset)
        except NoMatchingDataError:
            log.warning("no data: %s %s %s", dataset, key, year)
            df = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="ts_utc"))
        df.to_parquet(path)
        parts.append(df)
        log.info("fetched %s %s %s rows=%d", dataset, key, year, len(df))
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="ts_utc"))
    return pd.concat(parts).sort_index()


def fetch_prices(client: EntsoePandasClient, zone: str, start, end, force=False) -> pd.DataFrame:
    return cached_yearly("prices", zone, start, end,
                         lambda s, e: client.query_day_ahead_prices(zone, start=s, end=e), force)


def fetch_load_forecast(client: EntsoePandasClient, zone: str, start, end, force=False) -> pd.DataFrame:
    return cached_yearly("load_forecast", zone, start, end,
                         lambda s, e: client.query_load_forecast(zone, start=s, end=e), force)


def fetch_res_forecast(client: EntsoePandasClient, zone: str, start, end, force=False) -> pd.DataFrame:
    """Day-ahead wind and solar forecast (process type A01)."""
    return cached_yearly("res_forecast", zone, start, end,
                         lambda s, e: client.query_wind_and_solar_forecast(zone, start=s, end=e), force)


def fetch_flows(client: EntsoePandasClient, a: str, b: str, start, end, force=False) -> pd.DataFrame:
    """Physical flows a->b (A11). Both directions are fetched by the caller."""
    return cached_yearly("flows", f"{a}__{b}", start, end,
                         lambda s, e: client.query_crossborder_flows(a, b, start=s, end=e), force)


def fetch_ntc(client: EntsoePandasClient, a: str, b: str, start, end, force=False) -> pd.DataFrame:
    """Day-ahead forecast transfer capacity a->b. Flow-based borders have none."""
    return cached_yearly("ntc_da", f"{a}__{b}", start, end,
                         lambda s, e: client.query_net_transfer_capacity_dayahead(a, b, start=s, end=e), force)


_CURRENCY_RE = re.compile(r"<currency_Unit\.name>(\w+)</currency_Unit\.name>")
_UNIT_RE = re.compile(r"<price_Measure_Unit\.name>(\w+)</price_Measure_Unit\.name>")


def fetch_currency_audit(raw: EntsoeRawClient, start, end) -> pd.DataFrame:
    """Read the currency and price unit straight from the A44 XML for one sample day per zone-year.

    entsoe-py's parser drops these fields, so this is the only way to confirm every series is EUR/MWh.
    """
    path = C.RAW_DIR / "currency_audit.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for zone in C.ZONES:
        for year, s, _ in year_chunks(start, end):
            day = s + pd.Timedelta(days=14)
            try:
                xml = raw.query_day_ahead_prices(zone, start=day, end=day + pd.Timedelta(days=1),
                                                 sequence=1 if zone == "DE_LU" else None)
            except NoMatchingDataError:
                rows.append(dict(zone=zone, year=year, currency=None, unit=None))
                continue
            rows.append(dict(zone=zone, year=year,
                             currency=",".join(sorted(set(_CURRENCY_RE.findall(xml)))),
                             unit=",".join(sorted(set(_UNIT_RE.findall(xml))))))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def fetch_gas(force: bool = False) -> pd.Series:
    """Monthly TTF (IMF, USD/MMBtu) from FRED. Indexed by month start in market time."""
    path = C.RAW_DIR / "gas_ttf_monthly.csv"
    if force or not path.exists():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={C.FRED_GAS_SERIES}"
        pd.read_csv(url).to_csv(path, index=False)
    df = pd.read_csv(path, parse_dates=[0])
    s = df.set_index(df.columns[0]).iloc[:, 0].astype(float)
    s.index = s.index.tz_localize(C.MARKET_TZ)
    s.name = "ttf_usd_mmbtu"
    return s


def fetch_all(start: pd.Timestamp = C.SAMPLE_START, end: pd.Timestamp | None = None,
              force: bool = False) -> None:
    end = end or default_end()
    key = api_key()
    client = EntsoePandasClient(api_key=key, retry_count=5, retry_delay=15)
    raw = EntsoeRawClient(api_key=key, retry_count=5, retry_delay=15)
    for zone in C.ZONES:
        fetch_prices(client, zone, start, end, force)
        fetch_load_forecast(client, zone, start, end, force)
        fetch_res_forecast(client, zone, start, end, force)
    for border in C.BORDERS:
        for a, b in ((border.a, border.b), (border.b, border.a)):
            fetch_flows(client, a, b, start, end, force)
            fetch_ntc(client, a, b, start, end, force)
    fetch_currency_audit(raw, start, end)
    fetch_gas(force)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default=str(C.SAMPLE_START.date()))
    p.add_argument("--end", default=None, help="exclusive, market time; default: first day of this month")
    p.add_argument("--force", action="store_true", help="re-download cached years")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start = pd.Timestamp(args.start, tz=C.MARKET_TZ)
    end = pd.Timestamp(args.end, tz=C.MARKET_TZ) if args.end else None
    fetch_all(start, end, args.force)


if __name__ == "__main__":
    main()
