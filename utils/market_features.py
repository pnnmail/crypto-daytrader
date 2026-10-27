# -*- coding: utf-8 -*-
"""Market state features at the moment of decision.

WHAT THIS IS AND HOW IT DIFFERS FROM SCORING. There are NO votes, points or rules here.
It is only a description of the state the market was in, so that later, after the fact,
one can ask "what happened AFTER such states". Not a single number
from here takes part in the decision — `utils/signals.py` decides, and this work
must not touch it.

WHY A COIN CANNOT BE VIEWED IN A VACUUM. An altcoin can be "oversold"
for as long as BTC keeps falling. So next to the coin's own features
the state of BTC and ETH is written, plus RELATIVE strength — the coin's return minus
BTC's return over the same window. Without this "oversold" and "falling together
with the whole market" are indistinguishable in the data.

LOOK-AHEAD BIAS. All rolling statistics are computed on CLOSED bars:
the last, still forming bar is dropped. Its close is known and contains
no look-ahead, but its High/Low/Volume are incomplete, and a feature like
RVOL on a partial bar is systematically understated. So that a feature computed
live matches the feature computed on history, there is one rule for both.

CACHE. BTC and ETH are the same for all coins of a wave, and there are 25 coins. Without a cache that is 50
extra exchange requests per cycle. The TTL is short — features must
describe the current moment, not one half an hour old.
"""
import time

import numpy as np
import pandas as pd

# Windows in HOURS for regime returns. The set is declared in advance.
REGIME_LOOKBACKS_H = (1, 4, 24)

_CACHE = {}
_CACHE_TTL_SEC = 300


def _closed(df):
    """Drops the last, still forming bar."""
    return df.iloc[:-1] if df is not None and len(df) > 1 else df


def _cached_candles(coin, label, days_back):
    key = (coin, label, round(days_back, 2))
    hit = _CACHE.get(key)
    now = time.time()
    if hit and now - hit[0] < _CACHE_TTL_SEC:
        return hit[1]
    from data.fetcher import fetch_candles
    df = fetch_candles(coin, label, days_back=days_back)
    _CACHE[key] = (now, df)
    return df


def _ret_pct(df, hours):
    """Return over the last `hours` hours on closed hourly bars."""
    d = _closed(df)
    if d is None or len(d) <= hours:
        return None
    try:
        a = float(d["Close"].iloc[-hours - 1])
        b = float(d["Close"].iloc[-1])
    except (TypeError, ValueError, KeyError, IndexError):
        return None
    if not a:
        return None
    return round((b / a - 1) * 100, 4)


def _regime_label(r4, r24):
    """A rough regime label. The thresholds were NOT tuned to the result — it is just
    sign and magnitude; any fine tuning of thresholds here would be overfitting."""
    if r4 is None or r24 is None:
        return None
    if r24 > 2 and r4 > 0:
        return "BULL"
    if r24 < -2 and r4 < 0:
        return "BEAR"
    if r24 < -5:
        return "PANIC"
    return "NEUTRAL"


def market_regime():
    """The state of BTC and ETH. The same for all coins of a wave, hence cached."""
    out = {}
    for sym, prefix in (("BTC", "btc"), ("ETH", "eth")):
        try:
            df = _cached_candles(sym, "1h", days_back=6)
        except Exception:
            continue
        for h in REGIME_LOOKBACKS_H:
            out[f"{prefix}_ret_{h}h"] = _ret_pct(df, h)
    out["btc_regime"] = _regime_label(out.get("btc_ret_4h"), out.get("btc_ret_24h"))
    return out


def coin_features(coin, df_1h=None):
    """The coin's own features: trend, volatility, volume.

    df_1h — hourly bars, if they were already loaded higher up the stack. Otherwise
    they are fetched here (with a cache).
    """
    try:
        df = df_1h if df_1h is not None else _cached_candles(coin, "1h", days_back=14)
    except Exception:
        return {}
    d = _closed(df)
    if d is None or len(d) < 30:
        return {}
    out = {}
    try:
        close = d["Close"].astype(float)
        high = d["High"].astype(float)
        low = d["Low"].astype(float)
        vol = d["Volume"].astype(float)
    except (KeyError, TypeError, ValueError):
        return {}
    px = float(close.iloc[-1])

    # ── trend: distances to moving averages and their slope ──────────────
    for n in (20, 50, 200):
        if len(close) >= n:
            ema = close.ewm(span=n, adjust=False).mean()
            out[f"dist_ema{n}_pct"] = round((px / float(ema.iloc[-1]) - 1) * 100, 4)
            if len(ema) >= n + 5:
                # slope in percent over 5 bars — a unitless quantity,
                # comparable between coins of different price
                prev = float(ema.iloc[-6])
                if prev:
                    out[f"slope_ema{n}_pct"] = round((float(ema.iloc[-1]) / prev - 1) * 100, 4)

    # ── volatility ───────────────────────────────────────────────────────
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    if atr.notna().any() and px:
        out["atr_pct"] = round(float(atr.iloc[-1]) / px * 100, 4)
    rv = close.pct_change().rolling(24).std()
    if rv.notna().any():
        out["realized_vol_24h_pct"] = round(float(rv.iloc[-1]) * 100, 4)
        # The percentile of current volatility within its own history: it
        # answers "is it quiet or noisy right now FOR THIS coin",
        # whereas the absolute value is incomparable between coins.
        hist = rv.dropna()
        if len(hist) >= 50:
            out["vol_percentile"] = round(
                float((hist < hist.iloc[-1]).mean()) * 100, 1)

    # ── volume ───────────────────────────────────────────────────────────
    for n in (20, 50):
        if len(vol) >= n:
            avg = float(vol.iloc[-n:].mean())
            if avg:
                out[f"rvol{n}"] = round(float(vol.iloc[-1]) / avg, 4)
    if len(vol) >= 50:
        window = vol.iloc[-50:]
        sd = float(window.std())
        if sd:
            out["volume_zscore"] = round(
                (float(vol.iloc[-1]) - float(window.mean())) / sd, 4)
        out["volume_percentile"] = round(
            float((window < float(vol.iloc[-1])).mean()) * 100, 1)
    return out


def relative_strength(coin_ret_by_h, regime):
    """The coin's return MINUS the return of BTC/ETH over the same window."""
    out = {}
    for h in REGIME_LOOKBACKS_H:
        cr = coin_ret_by_h.get(h)
        if cr is None:
            continue
        for sym, prefix in (("btc", "vs_btc"), ("eth", "vs_eth")):
            mr = regime.get(f"{sym}_ret_{h}h")
            if mr is not None:
                out[f"{prefix}_{h}h"] = round(cr - mr, 4)
    return out


def build_market_features(coin, df_hourly=None):
    """The full feature block for one coin. Errors are not propagated:
    the features are a description, the loop must not crash because of them.

    df_hourly — hourly bars, if the caller already loaded them (build_snapshot
    does). Then there will be no repeated exchange request.
    """
    feats = {}
    df = df_hourly
    if df is None:
        try:
            df = _cached_candles(coin, "1h", days_back=14)
        except Exception:
            df = None
    try:
        feats.update(coin_features(coin, df))
    except Exception:
        pass
    try:
        regime = market_regime()
        feats.update(regime)
        coin_ret = {h: _ret_pct(df, h) for h in REGIME_LOOKBACKS_H}
        feats.update({f"ret_{h}h": v for h, v in coin_ret.items() if v is not None})
        feats.update(relative_strength(coin_ret, regime))
    except Exception:
        pass
    return feats
