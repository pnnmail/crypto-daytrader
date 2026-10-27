"""
Coinbase Exchange REST — a public API, no key. Binance.com is geo-blocked
(checked: HTTP 451), Coinbase covers all coins of the starting list and returns
candles/order book/trades without authorisation.

The data is continuous (24/7) — unlike stocks, no fallbacks for
weekend/holiday gaps are needed here (see the spec, item 4).
"""
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from config import (
    COINBASE_BASE, COINBASE_QUOTE, CANDLE_PAGE_PAUSE_SEC, CANDLE_MAX_PER_REQUEST,
    KRAKEN_FUTURES_BASE, KRAKEN_FUTURES_SYMBOL,
)

GRANULARITY_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400}

_session = requests.Session()

# Coinbase does not publish an explicit limit in the headers of public endpoints, so
# the backoff is a fixed conservative set of attempts, the same pattern as RATE_LIMIT_BACKOFF
# in the original project (data/fetcher.py in stock_agent2).
RATE_LIMIT_BACKOFF = [5, 10, 20]


def product_id(coin: str) -> str:
    return f"{coin.upper()}-{COINBASE_QUOTE}"


def _get(path: str, params: dict = None) -> object:
    url = f"{COINBASE_BASE}{path}"
    last_exc = None
    for attempt, backoff in enumerate([0] + RATE_LIMIT_BACKOFF):
        if backoff:
            time.sleep(backoff)
        try:
            resp = _session.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                last_exc = RuntimeError(f"rate limited: {url}")
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            last_exc = e
            continue
    raise RuntimeError(f"Coinbase API unavailable after {len(RATE_LIMIT_BACKOFF)+1} attempts: {url} ({last_exc})")


def fetch_candles_page(pid: str, granularity: int, start: datetime = None, end: datetime = None) -> list:
    params = {"granularity": granularity}
    if start is not None:
        params["start"] = start.replace(microsecond=0).isoformat()
    if end is not None:
        params["end"] = end.replace(microsecond=0).isoformat()
    data = _get(f"/products/{pid}/candles", params)
    return data or []


def _rows_to_df(rows: list) -> pd.DataFrame:
    """rows: [time, low, high, open, close, volume], arbitrary order, duplicate pages possible."""
    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    dedup = {r[0]: r for r in rows}
    ordered = sorted(dedup.values(), key=lambda r: r[0])
    idx = pd.to_datetime([r[0] for r in ordered], unit="s", utc=True)
    df = pd.DataFrame({
        "Low": [r[1] for r in ordered],
        "High": [r[2] for r in ordered],
        "Open": [r[3] for r in ordered],
        "Close": [r[4] for r in ordered],
        "Volume": [r[5] for r in ordered],
    }, index=idx)
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def fetch_candles(coin: str, granularity_label: str = "1h", days_back: float = 5) -> pd.DataFrame:
    """
    Candles for the last `days_back` days, with pagination (Coinbase returns at most
    ~300 candles per request — a year of STUMPY history needs dozens of requests).
    Returns a DataFrame in ascending time order, the index is a UTC timestamp.
    """
    granularity = GRANULARITY_SECONDS[granularity_label]
    pid = product_id(coin)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    max_span = timedelta(seconds=granularity * (CANDLE_MAX_PER_REQUEST - 1))

    all_rows = []
    page_end = end
    while page_end > start:
        page_start = max(start, page_end - max_span)
        rows = fetch_candles_page(pid, granularity, page_start, page_end)
        if not rows:
            break
        all_rows.extend(rows)
        earliest_ts = min(r[0] for r in rows)
        new_page_end = datetime.fromtimestamp(earliest_ts, tz=timezone.utc) - timedelta(seconds=granularity)
        if new_page_end >= page_end:
            break
        page_end = new_page_end
        if page_end > start:
            time.sleep(CANDLE_PAGE_PAUSE_SEC)

    df = _rows_to_df(all_rows)
    if df.empty:
        return df
    return df[df.index >= pd.Timestamp(start)]


def fetch_ticker(coin: str) -> dict:
    pid = product_id(coin)
    data = _get(f"/products/{pid}/ticker")
    return {
        "price": float(data.get("price", 0) or 0),
        "bid": float(data.get("bid", 0) or 0),
        "ask": float(data.get("ask", 0) or 0),
        "volume_24h": float(data.get("volume", 0) or 0),
        "time": data.get("time"),
    }


def fetch_order_book(coin: str, level: int = 2) -> dict:
    pid = product_id(coin)
    data = _get(f"/products/{pid}/book", params={"level": level})
    bids = [(float(p), float(s)) for p, s, *_ in data.get("bids", [])]
    asks = [(float(p), float(s)) for p, s, *_ in data.get("asks", [])]
    return {"bids": bids, "asks": asks}


def fetch_trades(coin: str, limit: int = 1000) -> pd.DataFrame:
    """
    The latest trades with the taker side (buy/sell) — the real order flow,
    more precise than the Close>Open heuristic on hourly bars used by the version
    for stocks (there is no trade tape there, only OHLCV).
    Coinbase returns up to 1000 trades per page without a cursor.
    """
    pid = product_id(coin)
    data = _get(f"/products/{pid}/trades", params={"limit": min(limit, 1000)})
    if not data:
        return pd.DataFrame(columns=["time", "side", "size", "price"])
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["size"] = df["size"].astype(float)
    df["price"] = df["price"].astype(float)
    return df.sort_values("time").reset_index(drop=True)[["time", "side", "size", "price"]]


def fetch_funding_rate(coin: str) -> dict:
    """
    Funding rate from Kraken Futures (PF_* perpetuals): positive -> longs
    pay shorts (skew towards the bulls), negative -> the other way round.
    Extreme values often precede squeezes (spec, item 5). Coinbase
    spot has nothing like this — there are no perpetuals there at all.

    Kraken returns fundingRate in ABSOLUTE units of the quote currency per
    contract, not as a fraction of price (checked empirically: for XBT fundingRate=0.229
    with markPrice=64931 — as a fraction that would be 23%, absurd for funding;
    as $ per contract it is 0.00035% of price, which matches the scale of
    the other coins). So we divide by markPrice ourselves.
    """
    coin = coin.upper()
    symbol = KRAKEN_FUTURES_SYMBOL.get(coin)
    if not symbol:
        return {"error": f"no Kraken Futures perpetual for {coin}"}

    try:
        resp = requests.get(f"{KRAKEN_FUTURES_BASE}/tickers", timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        return {"error": f"Kraken Futures unavailable: {e}"}

    for t in data.get("tickers", []):
        if t.get("symbol") == symbol:
            funding_rate, mark_price = t.get("fundingRate"), t.get("markPrice")
            if funding_rate is None or not mark_price:
                return {"error": "no funding rate in the Kraken response"}
            return {
                "symbol": symbol,
                "funding_rate_pct": round(funding_rate / mark_price * 100, 5),
                "funding_rate_raw": funding_rate,
                "mark_price": mark_price,
            }

    return {"error": f"{symbol} not found in Kraken Futures tickers"}


def check_data_quality(df_hourly: pd.DataFrame) -> tuple:
    """
    Unlike stocks, crypto trades continuously — there are no "trading halts" in
    the usual sense. The only real check is that data arrived at all
    and the last candle is not badly stale (e.g. a hung connector).
    """
    if df_hourly is None or df_hourly.empty:
        return False, "no data"
    last_ts = df_hourly.index[-1]
    age = datetime.now(timezone.utc) - last_ts.to_pydatetime()
    if age > timedelta(hours=3):
        return False, f"last candle from {last_ts.isoformat()} — data is stale ({age})"
    return True, ""
