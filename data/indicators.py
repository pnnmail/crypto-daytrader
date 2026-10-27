"""
Indicators for crypto. RSI/MACD/Bollinger/ATR — the same logic as in the version
for stocks (data/indicators.py in stock_agent2), 1:1, only the OHLCV source
changes to Coinbase candles. Pivot Points and VWAP are rewritten for the
00:00 UTC boundary (see utils/time_utils.py) instead of the "previous trading session" —
crypto has no close, so the old notion of a session does not apply.
"""
import pandas as pd
import numpy as np

from utils.time_utils import utc_day_start, now_utc

# Absorption detection thresholds — the same values as in stock_agent2 (checked
# empirically there): volume noticeably above average with range noticeably below.
ABSORPTION_VOLUME_MULT = 1.5
ABSORPTION_RANGE_MULT = 0.6

# If the current UTC day's open differs from the previous day's Pivot by more
# than this percent — R1/R2/S1/S2 extrapolate a range the market no longer
# recognises (see the same gate in stock_agent2, along with the observed case).
PIVOT_GAP_THRESHOLD_PCT = 5.0


def compute_indicators(df_hourly: pd.DataFrame) -> dict:
    """RSI(14, Wilder), MACD(12/26/9), Bollinger(20,2σ). Volume — from hourly
    candles: in crypto, unlike stocks, there are no bars with volume=0 outside a session,
    so there is no need to filter "bars without volume"."""
    if df_hourly.empty or len(df_hourly) < 2:
        return {"price": 0, "price_change": 0, "rsi": 50.0, "macd": 0, "macd_signal": 0,
                "bb_upper": 0, "bb_lower": 0, "bb_position_pct": 50.0, "volume": 0}

    close = df_hourly["Close"].astype(float)
    last_c = float(close.iloc[-1])
    prev_c = float(close.iloc[-2]) if len(close) > 1 else last_c
    chg = (last_c - prev_c) / prev_c * 100 if prev_c else 0

    last_volume = float(df_hourly["Volume"].iloc[-1])

    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    for i in range(14, len(avg_gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * 13 + gain.iloc[i]) / 14
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * 13 + loss.iloc[i]) / 14
    rsi = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, 1))
    rsi_v = float(rsi.iloc[-1])
    if pd.isna(rsi_v) or rsi_v <= 0 or rsi_v >= 100:
        rsi_v = 50.0

    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    bb_mid = close.rolling(20, min_periods=1).mean()
    bb_std = close.rolling(20, min_periods=1).std()
    bb_upper = float(bb_mid.iloc[-1] + 2 * bb_std.iloc[-1])
    bb_lower = float(bb_mid.iloc[-1] - 2 * bb_std.iloc[-1])
    bb_position_pct = (last_c - bb_lower) / (bb_upper - bb_lower) * 100 if bb_upper != bb_lower else 50.0

    return {
        "price": round(last_c, 6),
        "price_change": round(chg, 2),
        "rsi": round(rsi_v, 1),
        "macd": round(float(macd.iloc[-1]), 6),
        "macd_signal": round(float(macd_signal.iloc[-1]), 6),
        "bb_upper": round(bb_upper, 6),
        "bb_lower": round(bb_lower, 6),
        "bb_position_pct": round(bb_position_pct, 1),
        "volume": last_volume,
    }


def compute_atr(df: pd.DataFrame, period: int = 14):
    if df.empty or len(df) < period:
        return None
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period, min_periods=1).mean().iloc[-1])


def compute_moving_averages(df_daily: pd.DataFrame) -> dict:
    if df_daily.empty:
        return {}
    close = df_daily["Close"]
    result = {"sma20": float(close.rolling(20, min_periods=1).mean().iloc[-1]),
              "sma50": float(close.rolling(50, min_periods=1).mean().iloc[-1])}
    result["sma200"] = float(close.rolling(200, min_periods=1).mean().iloc[-1]) if len(close) >= 200 else None
    return result


def compute_daily_levels(df_hourly: pd.DataFrame) -> dict:
    """
    Pivot Points (floor-trader) and VWAP from the UTC day boundary.

    In crypto the market is never closed, so there is no "session open/
    closed -> pivot base previous/last" fork from the stock version — the base
    is always "the previous full UTC day", and we always plan the current day.
    """
    if df_hourly.empty:
        return {}

    today_start = utc_day_start()
    prev_day_start = today_start - pd.Timedelta(days=1)

    prev_day_bars = df_hourly[(df_hourly.index >= prev_day_start) & (df_hourly.index < today_start)]
    if prev_day_bars.empty:
        # Not enough history for a full previous day (e.g. the bot was
        # just started) — fall back to the whole loaded range.
        h, l, c = float(df_hourly['High'].max()), float(df_hourly['Low'].min()), float(df_hourly['Close'].iloc[-1])
    else:
        h = float(prev_day_bars['High'].max())
        l = float(prev_day_bars['Low'].min())
        c = float(prev_day_bars['Close'].iloc[-1])

    pp = (h + l + c) / 3
    r1, s1 = 2 * pp - l, 2 * pp - h
    r2, s2 = pp + (h - l), pp - (h - l)
    r3, s3 = h + 2 * (pp - l), l - 2 * (h - pp)

    today_bars = df_hourly[df_hourly.index >= today_start]
    today_open = float(today_bars['Open'].iloc[0]) if not today_bars.empty else float(df_hourly['Close'].iloc[-1])
    pivot_gap_pct = abs(today_open - pp) / pp * 100 if pp else 0
    pivot_reliable = pivot_gap_pct <= PIVOT_GAP_THRESHOLD_PCT

    # VWAP: resets every UTC day. If the day has only just begun (few bars),
    # that is expected — unlike stocks, there are no "days without volume" here
    # (the market is continuous), no fallback to the previous session is needed.
    session = today_bars if not today_bars.empty else df_hourly
    vol_sum = float(session['Volume'].sum())
    vwap = (float((session['Close'] * session['Volume']).sum()) / vol_sum
            if vol_sum > 0 else float(df_hourly['Close'].iloc[-1]))

    atr_h = compute_atr(df_hourly, period=14)
    price_now = float(df_hourly['Close'].iloc[-1])

    prev_bar = df_hourly.iloc[-2] if len(df_hourly) > 1 else None
    prev_high = float(prev_bar['High']) if prev_bar is not None else h
    prev_low = float(prev_bar['Low']) if prev_bar is not None else l
    prev_close = float(prev_bar['Close']) if prev_bar is not None else c

    distance_to_vwap = (price_now - vwap) / vwap * 100 if vwap else 0
    distance_to_r1 = (r1 - price_now) / price_now * 100 if price_now else 0
    distance_to_s1 = (s1 - price_now) / price_now * 100 if price_now else 0

    if price_now > vwap * 1.001:
        vwap_position = "above"
    elif price_now < vwap * 0.999:
        vwap_position = "below"
    else:
        vwap_position = "at"

    def calc_trend(df_slice):
        if len(df_slice) < 2:
            return "neutral"
        first_close, last_close = float(df_slice['Close'].iloc[0]), float(df_slice['Close'].iloc[-1])
        change = (last_close - first_close) / first_close * 100
        if change > 0.2:
            return "bullish"
        if change < -0.2:
            return "bearish"
        return "neutral"

    trend_5h = calc_trend(df_hourly.tail(5))
    trend_15h = calc_trend(df_hourly.tail(15))

    return {
        'pivot': round(pp, 6), 'r1': round(r1, 6), 'r2': round(r2, 6), 'r3': round(r3, 6),
        's1': round(s1, 6), 's2': round(s2, 6), 's3': round(s3, 6),
        'pivot_reliable': pivot_reliable, 'pivot_gap_pct': round(pivot_gap_pct, 2),
        'vwap': round(vwap, 6), 'price_above_vwap': price_now > vwap,
        'atr_hourly': round(atr_h, 6) if atr_h else None,
        'prev_high': round(prev_high, 6), 'prev_low': round(prev_low, 6), 'prev_close': round(prev_close, 6),
        'vwap_position': vwap_position,
        'distance_to_vwap': round(distance_to_vwap, 2),
        'distance_to_r1': round(distance_to_r1, 2),
        'distance_to_s1': round(distance_to_s1, 2),
        'trend_5h': trend_5h, 'trend_15h': trend_15h,
        'trend_divergence': {trend_5h, trend_15h} == {"bullish", "bearish"},
        'day_boundary': 'UTC 00:00',
    }


def compute_volume_profile(df_hourly: pd.DataFrame) -> dict:
    if df_hourly.empty:
        return {}
    price_range = float(df_hourly['High'].max()) - float(df_hourly['Low'].min())
    if price_range == 0:
        return {'poc': float(df_hourly['Close'].iloc[-1]), 'value_area_high': None, 'value_area_low': None}
    step = price_range / 20
    low = float(df_hourly['Low'].min())
    levels = {}
    for i in range(20):
        level_low, level_high = low + i * step, low + (i + 1) * step
        mask = (df_hourly['Close'] >= level_low) & (df_hourly['Close'] < level_high)
        levels[round((level_low + level_high) / 2, 6)] = float(df_hourly.loc[mask, 'Volume'].sum())
    if not levels:
        return {'poc': float(df_hourly['Close'].iloc[-1]), 'value_area_high': None, 'value_area_low': None}
    poc = max(levels, key=levels.get)
    total_vol = sum(levels.values())
    sorted_levels = sorted(levels.items(), key=lambda x: abs(x[0] - poc))
    cum_vol, vah, val = 0, poc, poc
    for price, vol in sorted_levels:
        cum_vol += vol
        if cum_vol >= total_vol * 0.35:
            break
        vah, val = max(vah, price), min(val, price)
    return {'poc': round(poc, 6), 'value_area_high': round(vah, 6), 'value_area_low': round(val, 6),
            'total_volume': total_vol}


def detect_liquidity_levels(df_hourly: pd.DataFrame) -> dict:
    if df_hourly.empty:
        return {}
    highs, lows = df_hourly['High'].values, df_hourly['Low'].values
    equal_highs = [round(float(highs[i]), 6) for i in range(len(highs) - 1)
                   if highs[i] and abs(highs[i] - highs[i + 1]) / highs[i] < 0.001]
    equal_lows = [round(float(lows[i]), 6) for i in range(len(lows) - 1)
                  if lows[i] and abs(lows[i] - lows[i + 1]) / lows[i] < 0.001]
    return {'stops_above': list(set(equal_highs))[:3], 'stops_below': list(set(equal_lows))[:3]}


def compute_volume_ratio(df_daily: pd.DataFrame, lookback_weeks: int = 8) -> dict:
    """
    Volume of the last full day relative to the average for THE SAME weekday over
    lookback_weeks weeks — not relative to the average over all days.

    Despite the 24/7 regime, weekend volume is noticeably lower (see the spec, item 4).
    Comparing with the average over all days would make Saturday-Sunday forever
    "abnormally low volume", although that is normal for that weekday.
    """
    if df_daily.empty or len(df_daily) < 2:
        return {"volume_ratio": 1.0, "basis": "not enough history"}

    # The last candle of the daily series is usually the CURRENT, not yet closed UTC
    # day (it may have only minutes of volume). Comparing it with full
    # historical days is meaningless — the last COMPLETED day is needed.
    completed = df_daily[df_daily.index < utc_day_start()]
    if completed.empty or len(completed) < 2:
        return {"volume_ratio": 1.0, "basis": "not enough completed days"}

    last_row = completed.iloc[-1]
    last_weekday = completed.index[-1].weekday()
    last_volume = float(last_row['Volume'])

    same_weekday = completed[completed.index.weekday == last_weekday]
    history = same_weekday.iloc[:-1].tail(lookback_weeks)
    if history.empty:
        return {"volume_ratio": 1.0, "basis": "no history for this weekday", "last_volume": last_volume}

    avg_volume = float(history['Volume'].mean())
    ratio = last_volume / avg_volume if avg_volume > 0 else 1.0
    weekday_dative_plural = ['Mondays', 'Tuesdays', 'Wednesdays', 'Thursdays', 'Fridays', 'Saturdays', 'Sundays']
    return {
        "volume_ratio": round(ratio, 2),
        "basis": f"vs the average for {weekday_dative_plural[last_weekday]} over {len(history)} wks",
        "avg_volume_same_weekday": round(avg_volume, 2),
        "last_volume": round(last_volume, 2),
        "is_weekend": last_weekday >= 5,
    }


# Below this imbalance magnitude the flow counts as noise, not direction.
# The same number is already used as the threshold for delta_divergence below —
# we keep one value so that "a significant skew" means the same thing.
#
# Previously this boundary was drawn by the LLM, and it drew it ASYMMETRICALLY: a measurement on 26
# coins (2026-08-10) showed that it called an imbalance of -0.032 "bearish"
# and +0.041 "neutral", i.e. noise around zero was systematically read as
# a move down. With a factor weight of ±2 and an entry threshold of ±3 this shifted the whole
# system towards SHORT. A deterministic symmetric threshold removes the
# skew and also saves a whole LLM call per coin.
ORDERFLOW_BIAS_THRESHOLD = 0.1


def _bias_from_imbalance(imbalance: float) -> str:
    if imbalance >= ORDERFLOW_BIAS_THRESHOLD:
        return "bullish"
    if imbalance <= -ORDERFLOW_BIAS_THRESHOLD:
        return "bearish"
    return "neutral"


def compute_orderflow_metrics(trades_df: pd.DataFrame, df_hourly: pd.DataFrame, indicators: dict, daily_levels: dict) -> dict:
    """
    Python computes the taker imbalance from the real trade tape (Coinbase
    /trades returns the buy/sell side) — more precise than the Close>Open heuristic on
    hourly bars that the stock version is forced to use (there is no
    trade tape there, only OHLCV).

    The imbalance is a SHARE (buy-sell)/(buy+sell) in [-1,1], not absolute dollars
    as in stocks: there the 20M/50M thresholds are calibrated for one ticker, while here
    the turnover of BTC and DOGE differs by orders of magnitude — a fixed threshold would be
    meaningless for small coins and would never fire for large ones.
    """
    if trades_df is not None and not trades_df.empty:
        notional = trades_df['size'] * trades_df['price']
        buy_notional = float(notional[trades_df['side'] == 'buy'].sum())
        sell_notional = float(notional[trades_df['side'] == 'sell'].sum())
        total_notional = buy_notional + sell_notional
        imbalance = (buy_notional - sell_notional) / total_notional if total_notional > 0 else 0.0
    else:
        buy_notional = sell_notional = total_notional = 0.0
        imbalance = 0.0

    result = {
        "taker_buy_notional": round(buy_notional, 2),
        "taker_sell_notional": round(sell_notional, 2),
        "taker_imbalance": round(imbalance, 3),
        "large_player_active": abs(imbalance) > 0.35 and total_notional > 0,
        "orderflow_bias": _bias_from_imbalance(imbalance),
    }

    price = indicators.get('price', 0)
    vwap = daily_levels.get('vwap', price)
    if imbalance > ORDERFLOW_BIAS_THRESHOLD and price < vwap * 0.999:
        result['delta_divergence'] = True
    elif imbalance < -ORDERFLOW_BIAS_THRESHOLD and price > vwap * 1.001:
        result['delta_divergence'] = True
    else:
        result['delta_divergence'] = False

    if not df_hourly.empty and len(df_hourly) >= 5:
        avg_volume = float(df_hourly['Volume'].tail(20).mean())
        last_volume = float(df_hourly['Volume'].iloc[-1])
        volume_ratio = last_volume / avg_volume if avg_volume > 0 else 0
        result['volume_confirms_trend'] = volume_ratio > 1.2

        bar_ranges = df_hourly['High'] - df_hourly['Low']
        avg_range = float(bar_ranges.tail(20).mean())
        last_range = float(bar_ranges.iloc[-1])
        result['absorption_detected'] = bool(
            avg_range > 0 and avg_volume > 0
            and last_volume > avg_volume * ABSORPTION_VOLUME_MULT
            and last_range < avg_range * ABSORPTION_RANGE_MULT
        )
    else:
        result['volume_confirms_trend'] = False
        result['absorption_detected'] = False

    return result
