"""
Rolling correlation with BTC (spec, item 5) — most altcoins follow
bitcoin; without this the system would find "ten independent signals"
that are in fact one and the same BTC move.

Computed on DAILY RETURNS (not price levels) — the price correlation of two
rising assets is almost always artificially high, returns show
the real co-movement.
"""
import pandas as pd

from data.fetcher import fetch_candles

CORRELATION_WINDOW_DAYS = 30
HIGH_CORRELATION_THRESHOLD = 0.8


def compute_btc_correlation(coin: str, window_days: int = CORRELATION_WINDOW_DAYS,
                             df_coin_daily: pd.DataFrame = None, df_btc_daily: pd.DataFrame = None) -> dict:
    coin = coin.upper()
    if coin == "BTC":
        return {"coin": coin, "correlation": 1.0, "window_days": window_days, "high_correlation": False,
                "note": "this is BTC itself"}

    days_needed = window_days + 5
    df_coin = df_coin_daily if df_coin_daily is not None else fetch_candles(coin, "1d", days_back=days_needed)
    df_btc = df_btc_daily if df_btc_daily is not None else fetch_candles("BTC", "1d", days_back=days_needed)

    if df_coin.empty or df_btc.empty:
        return {"coin": coin, "correlation": None, "note": "no data"}

    merged = pd.DataFrame({"coin": df_coin["Close"], "btc": df_btc["Close"]}).dropna()
    if len(merged) < 5:
        return {"coin": coin, "correlation": None, "note": "not enough overlapping days"}

    returns = merged.pct_change().dropna().tail(window_days)
    if len(returns) < 5:
        return {"coin": coin, "correlation": None, "note": "not enough returns"}

    corr = returns["coin"].corr(returns["btc"])
    if pd.isna(corr):
        return {"coin": coin, "correlation": None, "note": "could not compute (zero variance?)"}

    corr = round(float(corr), 3)
    return {
        "coin": coin, "correlation": corr, "window_days": len(returns),
        "high_correlation": abs(corr) > HIGH_CORRELATION_THRESHOLD,
    }


def format_correlation_text(result: dict) -> str:
    if not result or result.get("correlation") is None:
        return ""
    coin, corr = result["coin"], result["correlation"]
    if coin == "BTC":
        return ""
    lines = [f"🔗 *Correlation with BTC* (returns, {result.get('window_days')} d.): {corr:+.2f}"]
    if result.get("high_correlation"):
        lines.append("⚠️ High correlation — the move probably follows BTC, not an independent signal.")
    return "\n".join(lines)
