"""
Donchian channels / Turtle as a BASELINE in the experiment.

Why it is here. Our system is complex: scoring from five factors, seven
variants of the LLM decider, books, briefings. The baseline answers a question
that nobody else would ask — does all this complexity give anything beyond
three lines of pandas? Without such a comparison "mean per point -0.08%" has nothing
to be compared with.

THE RULE (classic Turtle, LONG only — there are no shorts on spot):
  entry — a close above the maximum of the N previous DAYS;
  exit  — a close below the minimum of the M previous days, M < N.
The rule is STATEFUL: at any moment it is either in a position or out of it.
So the decision at a point = LONG if currently in a position, otherwise NO_TRADE.

shift(1) is mandatory: without it today's candle takes part in computing
its own channel, and look-ahead appears.

AN HONEST CAVEAT, without which the comparison would mislead.
Donchian is designed for holding for WEEKS (median trade in the backtest ~19
days), while the experiment measures the outcome after 4 HOURS. I.e. we ask
a weekly rule for its opinion on direction and check it four hours later.
That is a legitimate question ("does the direction match"), but NOT a test of the Donchian
strategy: its profit accrues over long holds, which are absent here.
This row must not be read as "Donchian is worse/better than our system as a strategy"
— it only speaks for the directional signal at the common horizon.

A historical measurement (scratchpad/donchian_wide.py) separately showed that on
the FULL Coinbase universe the rule loses money (-30.6% median versus -78.8% for
buy-and-hold), and looked positive only on surviving coins.
Here it is applied to the current list — i.e. precisely to the survivors, and
this bias must be kept in mind.
"""
import numpy as np
import pandas as pd

# Classic Turtle values, NOT tuned by us. On history the ordering by N
# was monotonic (20 better than 30 better than 55 better than 100), which is more reliable than a single
# peak, but one should still take the canonical value, not the best one.
N_ENTRY = 20
N_EXIT = 10
DAYS_BACK = 120          # with a margin for the channel's warm-up


def donchian_state(coin: str, df_daily: pd.DataFrame = None) -> dict:
    """The current state of the rule for a coin.

    Returns {"direction": "LONG"|"NO_TRADE", "in_position": bool, ...}
    or {"_error": True} if there was not enough data. An error is an ERROR, not
    NO_TRADE: otherwise load failures would look like the rule's caution and
    inflate its statistics (the same logic as for the LLM deciders).
    """
    try:
        if df_daily is None or df_daily.empty:
            from data.fetcher import fetch_candles
            df_daily = fetch_candles(coin, "1d", days_back=DAYS_BACK)
        if df_daily is None or len(df_daily) < N_ENTRY + 5:
            return {"_error": True, "_error_message": "too few daily candles"}

        high, low, close = df_daily["High"], df_daily["Low"], df_daily["Close"]
        upper = high.rolling(N_ENTRY).max().shift(1)
        lower = low.rolling(N_EXIT).min().shift(1)
        c, up, lo = close.values, upper.values, lower.values

        # The state is restored by a PASS over the history, not by a single
        # comparison: the rule is stateful, and "price above the channel today" is not
        # the same as "we are in a position" — the entry could have happened a week ago.
        in_pos, entry_px, entry_i = False, None, None
        for i in range(len(df_daily)):
            if not np.isfinite(up[i]) or not np.isfinite(lo[i]):
                continue
            if not in_pos and c[i] > up[i]:
                in_pos, entry_px, entry_i = True, float(c[i]), i
            elif in_pos and c[i] < lo[i]:
                in_pos, entry_px, entry_i = False, None, None

        res = {
            "direction": "LONG" if in_pos else "NO_TRADE",
            "in_position": in_pos,
            "n_entry": N_ENTRY,
            "n_exit": N_EXIT,
            "upper": round(float(up[-1]), 8) if np.isfinite(up[-1]) else None,
            "lower": round(float(lo[-1]), 8) if np.isfinite(lo[-1]) else None,
        }
        if in_pos and entry_px:
            res["entry_price"] = entry_px
            res["days_in_position"] = int(len(df_daily) - 1 - entry_i)
            res["open_pnl_pct"] = round((float(c[-1]) - entry_px) / entry_px * 100, 2)
        return res
    except Exception as e:
        return {"_error": True, "_error_message": str(e)[:150]}
