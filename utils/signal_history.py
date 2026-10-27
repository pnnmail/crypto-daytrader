"""
Signal history + outcome checks.

The resolution rule is "intraday": one position, a forced close after
OUTCOME_HORIZON_HOURS hours, statistics from a clean slate under this rule
(see resolution_rule in each record). Ported from the stock version
and heavily simplified, taking advantage of the fact that
crypto trades continuously: there is no notion of a "session", an "opening gap" or
an "out-of-session signal" — entry_price is always executable right now, and
candles come without gaps. So there is no reanchor/gap_invalidated/
gap_no_edge from the stock version — there it solved a specific problem of
market closure that does not exist in crypto.
"""
import json
import statistics
from datetime import datetime, timedelta, timezone

from config import (
    SIGNAL_HISTORY_FILE, NO_TRADE_HISTORY_FILE, LONG_THRESHOLD,
    OUTCOME_HORIZON_HOURS, CHECKPOINT_HOURS_LIST, SPREAD_ROUND_TRIP_PCT,
)
from utils.jsonstore import load_json, save_json
from utils.time_utils import session_key as make_session_key, now_utc

TRADE_DIRECTIONS = {"LONG", "SHORT", "STRONG_LONG", "STRONG_SHORT"}

# The SCORING FORMULA version, written into every observation. Observations with
# different versions must not be added into one statistic: the definition of the score
# differs, and the sum of such records means nothing (the same logic as
# resolution_rule for the resolution horizon).
#   v1 — orderflow_bias was decided by the LLM (before 2026-08-10)
#   v2 — orderflow_bias is computed deterministically in Python with the
#        ORDERFLOW_BIAS_THRESHOLD (see data/indicators.py): the LLM's boundary
#        was asymmetric and skewed the system towards SHORT.
#   v3 — following the 2026-08-11 measurement (10 coins, 400 days, ~95k points):
#        MACD and VWAP flipped to contrarian (they were mirror-wrong),
#        STUMPY compared with the coin's base rate rather than 50%.
#        Spread set to zero (research mode), costs shown as
#        a warning. Details in the utils/signals.py header.
#   v4 — the EXIT rule from the minute-bar backtest (in-sample 1436 trades
#        + out-of-sample 967 on an independent window): a 2.5*ATR stop instead of
#        a structural one, the MIN_ENTRY_RR filter off. These two parameters must not be changed
#        separately — the table and rationale are in the utils/signals.py header.
#        NOTE: in the first hours of v4 (before 2026-08-14 02:06 UTC) the
#        MIN_TARGET_ATR cut-off was missing, and with the price right at VWAP the target sat at
#        the price itself — a trade with R:R 1:0.0 that closed "at target" for +0.005%
#        and inflated the win rate. Records from that period were removed by hand (archive
#        *_prefix_bug_20260814.json), the version label was not changed: no corrupted
#        observations remain in the files.
# v5: three -1 votes without a measured basis removed (RSI>70, MACD bullish,
#     BB upper) — a measurement on all 25 coins over 400 days with separated windows
#     showed that overbought does NOT predict a fall, while
#     oversold does predict a rise. As a side effect the negative range
#     shrank to -6, so confidence and position size are normalised to
#     the maximum of their own side. v4 and v5 observations are NOT COMPARABLE — the share of shorts
#     changes by construction, not because of the market.
# v6: the STUMPY ±2 vote removed — measured on 1088 points, r=+0.00 at our
#     horizon, and the "factor is silent" group beat both of its opinionated
#     groups. The range became +7/-4, so the short threshold was recomputed from -3 to -2
#     by the same proportion rule recorded in config.py: otherwise the bar
#     would have been raised by arithmetic, not by measurement.
SCORING_VERSION = "v6_no_stumpy"
MAX_RECORDS_PER_COIN = 50
# Lower than for trades: NO_TRADE is written on EVERY run of every coin — without
# a stricter cap the control group would grow disproportionately faster.
#
NO_TRADE_MAX_RECORDS_PER_COIN = 30


# Both histories go through utils/jsonstore: atomic writes and failing loudly instead of
# silent emptiness. The same bare except Exception -> {} was here too;
# these files survived 2026-09-06 only because they are smaller and the write window
# is shorter — i.e. by luck, not by protection.
def load_signal_history():
    return load_json(SIGNAL_HISTORY_FILE)


def save_signal_history(history):
    save_json(SIGNAL_HISTORY_FILE, history)


def load_no_trade_history():
    return load_json(NO_TRADE_HISTORY_FILE)


def save_no_trade_history(history):
    save_json(NO_TRADE_HISTORY_FILE, history)


def _evidence_fields(trader_plan):
    return {
        "total_score": trader_plan.get("direction_score"),
        "scores": trader_plan.get("scores"),
        "reasons": trader_plan.get("reasons"),
        "features": trader_plan.get("features"),
    }


def append_signal(coin, trader_plan):
    direction = trader_plan.get("direction")
    if direction not in TRADE_DIRECTIONS:
        append_no_trade(coin, trader_plan)
        return

    key = make_session_key(coin, direction)
    now_iso = now_utc().isoformat(timespec="seconds")

    history = load_signal_history()
    coin_history = history.get(coin, [])

    # Deduplication: one UTC day + one direction = ONE observation.
    # /autoscan polls every AUTOSCAN_INTERVAL_MIN minutes — without this one
    # and the same signal would count as dozens of independent observations.
    for existing in coin_history:
        if existing.get("session_key") == key:
            existing["repeat_count"] = existing.get("repeat_count", 1) + 1
            existing["last_seen"] = now_iso
            save_signal_history(history)
            return

    record = {
        "timestamp": now_iso, "session_key": key, "repeat_count": 1, "direction": direction,
        "entry_price": trader_plan.get("entry_price"), "stop_loss": trader_plan.get("stop_loss"),
        "target_1": trader_plan.get("target_1"), "target_2": trader_plan.get("target_2"),
        "risk_reward_1": trader_plan.get("risk_reward_1"), "risk_reward_2": trader_plan.get("risk_reward_2"),
        "spread_cost_pct": trader_plan.get("spread_cost_pct"),
        "confidence": trader_plan.get("confidence"), "position_size_pct": trader_plan.get("position_size_pct"),
        # The horizon and the spread rate are fixed IN THE RECORD at creation time rather than
        # taken from the current config during resolution — otherwise a settings change would
        # silently redefine the meaning of already accumulated observations retroactively.
        "horizon_hours": OUTCOME_HORIZON_HOURS,
        "spread_assumption_pct": SPREAD_ROUND_TRIP_PCT,
        "scoring_version": SCORING_VERSION,
        **_evidence_fields(trader_plan),
    }
    coin_history.append(record)
    history[coin] = coin_history[-MAX_RECORDS_PER_COIN:]
    save_signal_history(history)


def append_no_trade(coin, trader_plan):
    total_score = trader_plan.get("direction_score")
    passed_threshold = total_score is not None and abs(total_score) >= LONG_THRESHOLD
    key = make_session_key(coin, "NO_TRADE")
    now_iso = now_utc().isoformat(timespec="seconds")

    history = load_no_trade_history()
    coin_history = history.get(coin, [])
    for existing in coin_history:
        if existing.get("session_key") == key:
            existing["repeat_count"] = existing.get("repeat_count", 1) + 1
            existing["last_seen"] = now_iso
            save_no_trade_history(history)
            return

    record = {
        "timestamp": now_iso, "session_key": key, "repeat_count": 1, "direction": "NO_TRADE",
        "no_trade_kind": "entry_filter" if passed_threshold else "below_threshold",
        "entry_filter_reason": trader_plan.get("entry_filter_reason"),
        "scoring_version": SCORING_VERSION,
        **_evidence_fields(trader_plan),
    }
    coin_history.append(record)
    history[coin] = coin_history[-NO_TRADE_MAX_RECORDS_PER_COIN:]
    save_no_trade_history(history)


def _dedupe_records(records):
    grouped = {}
    for r in sorted(records, key=lambda x: x.get("timestamp", "")):
        key = r.get("session_key") or r.get("timestamp", "")
        if key in grouped:
            grouped[key]["repeat_count"] = grouped[key].get("repeat_count", 1) + r.get("repeat_count", 1)
        else:
            grouped[key] = r
    return list(grouped.values())


def _pending_checkpoints(record):
    have = set((record.get("checkpoints") or {}).keys())
    return [h for h in CHECKPOINT_HOURS_LIST if f"{h}h" not in have]


def _resolve_signal_outcome(coin, record):
    """
    Resolves a signal's outcome from REAL minute candles after the signal moment.

    Minute granularity is mandatory: on hourly bars a 4-hour window
    has only 4 bars, and the assumption "stop before target when both hit in one bar"
    would decide the outcome of almost every trade rather than a rare edge case.

    The main outcome (stop/target_1/target_2/close) is decided once within
    horizon_hours FROM THE RECORD. Checkpoints are independent information reported
    separately: the raw return (without a stop/target) at 1/2/4/8/24 hours,
    filled in as that time actually passes — they may be
    computed over SEVERAL passes after the main outcome has been decided.

    Returns a dict with any combination of {"outcome","price","time"} (the main
    outcome, if not yet decided) and {"checkpoints": {...}} (new checkpoints), or
    {"unresolved_reason": "..."} if resolution is impossible, or None if
    there is nothing to resolve right now (the horizon has not expired and there are no new checkpoints).
    """
    from data.fetcher import fetch_candles

    try:
        signal_dt = datetime.fromisoformat(record["timestamp"])
        if signal_dt.tzinfo is None:
            signal_dt = signal_dt.replace(tzinfo=timezone.utc)
    except (ValueError, KeyError, TypeError):
        return {"unresolved_reason": "could not parse the signal timestamp"}

    direction = record.get("direction", "")
    is_long, is_short = direction in ("LONG", "STRONG_LONG"), direction in ("SHORT", "STRONG_SHORT")
    entry = record.get("entry_price")
    if not (is_long or is_short) or entry is None:
        return None

    horizon_hours = record.get("horizon_hours", OUTCOME_HORIZON_HOURS)
    stop, t1, t2 = record.get("stop_loss"), record.get("target_1"), record.get("target_2")

    already_resolved = record.get("outcome") not in (None, "unresolved")
    pending_cp = _pending_checkpoints(record)
    if already_resolved and not pending_cp:
        return None

    now = now_utc()
    elapsed_hours = (now - signal_dt).total_seconds() / 3600
    if elapsed_hours <= 0:
        return None

    days_back = elapsed_hours / 24 + 0.05  # a small buffer for rounding inaccuracy
    df = fetch_candles(coin, "1m", days_back=max(days_back, 0.05))
    if df.empty:
        return {"unresolved_reason": "no candles from the exchange"} if not already_resolved else None

    future_bars = df[df.index > signal_dt]
    if future_bars.empty:
        return {"unresolved_reason": "no candles after the signal moment"} if not already_resolved else None

    result = {}

    # ─── main outcome ───
    if not already_resolved:
        horizon_end = signal_dt + timedelta(hours=horizon_hours)
        window_bars = future_bars[future_bars.index <= horizon_end]

        hit = None
        for ts, row in window_bars.iterrows():
            try:
                high, low = float(row["High"]), float(row["Low"])
            except (TypeError, ValueError):
                continue
            if is_long:
                stop_hit = stop is not None and low <= stop
                t2_hit = t2 is not None and high >= t2
                t1_hit = t1 is not None and high >= t1
            else:
                stop_hit = stop is not None and high >= stop
                t2_hit = t2 is not None and low <= t2
                t1_hit = t1 is not None and low <= t1
            # If both hit in one bar, the stop is considered to have fired first
            # (conservative — biases the result AGAINST the system, the only
            # safe direction of error when evaluating your own strategy).
            if stop_hit:
                hit = {"outcome": "stop", "price": stop, "time": ts.isoformat()}
                break
            if t2_hit:
                hit = {"outcome": "target_2", "price": t2, "time": ts.isoformat()}
                break
            if t1_hit:
                hit = {"outcome": "target_1", "price": t1, "time": ts.isoformat()}
                break

        if hit:
            result.update(hit)
        elif now < horizon_end:
            pass  # the horizon has not expired yet — the main outcome will wait (checkpoints below are not blocked)
        elif window_bars.empty:
            result["unresolved_reason"] = "no candles within the horizon"
        else:
            # The horizon expired, neither the stop nor the targets were touched -> the position is CLOSED
            # by time. This is not a data gap but a real trade outcome.
            result["outcome"] = "close"
            result["price"] = float(window_bars["Close"].iloc[-1])
            result["time"] = window_bars.index[-1].isoformat()

    # ─── checkpoints (independent of the main outcome, may be computed later) ───
    new_checkpoints = {}
    for h in pending_cp:
        cp_time = signal_dt + timedelta(hours=h)
        if now < cp_time:
            continue
        cp_bars = future_bars[future_bars.index <= cp_time]
        if cp_bars.empty:
            continue
        price_at_cp = float(cp_bars["Close"].iloc[-1])
        ret = ((price_at_cp - entry) / entry * 100) if is_long else ((entry - price_at_cp) / entry * 100)
        new_checkpoints[f"{h}h"] = round(ret, 2)
    if new_checkpoints:
        result["checkpoints"] = new_checkpoints

    return result if result else None


def _get_or_resolve_outcome(coin, record):
    """
    Updates the record in place. Returns (changed, info):
      changed — whether the history needs saving to disk;
      info — {"newly_closed": bool, "net": float|None, "newly_unresolved": bool,
              "checkpoint_added": bool} for the resolve_all_outcomes summary.
    """
    info = {"newly_closed": False, "net": None, "newly_unresolved": False, "checkpoint_added": False}

    resolved = _resolve_signal_outcome(coin, record)
    if resolved is None:
        return False, info

    changed = False
    already_resolved = record.get("outcome") not in (None, "unresolved")

    if resolved.get("unresolved_reason") and not already_resolved:
        record["outcome"] = "unresolved"
        record["unresolved_reason"] = resolved["unresolved_reason"]
        changed = True
        info["newly_unresolved"] = True
    elif resolved.get("outcome") and not already_resolved:
        outcome_price = resolved["price"]
        entry = record.get("entry_price")
        direction = record.get("direction")
        is_long = direction in ("LONG", "STRONG_LONG")
        gross = ((outcome_price - entry) / entry * 100) if is_long else ((entry - outcome_price) / entry * 100)
        spread_pct = record.get("spread_assumption_pct", SPREAD_ROUND_TRIP_PCT)
        net = gross - spread_pct

        record["outcome"] = resolved["outcome"]
        record["outcome_price"] = outcome_price
        record["outcome_time"] = resolved.get("time")
        record["return_pct_gross"] = round(gross, 3)
        record["return_pct_net"] = round(net, 3)
        horizon_hours = record.get("horizon_hours", OUTCOME_HORIZON_HOURS)
        record["resolution_rule"] = f"intraday_{horizon_hours}h"
        changed = True
        info["newly_closed"] = True
        info["net"] = net

    if resolved.get("checkpoints"):
        record.setdefault("checkpoints", {}).update(resolved["checkpoints"])
        changed = True
        info["checkpoint_added"] = True

    return changed, info


def resolve_all_outcomes():
    """Runs through the whole signal history and computes outcomes/checkpoints that
    are not cached yet. Returns a summary of this pass (for the Telegram
    message and for logs)."""
    history = load_signal_history()
    summary = {"updated": 0, "closed": 0, "wins": 0, "losses": 0, "unresolved": 0, "checkpoints_added": 0}
    for coin, records in history.items():
        for record in records:
            try:
                changed, info = _get_or_resolve_outcome(coin, record)
            except Exception as e:
                print(f"   ⚠️ Could not resolve the outcome for {coin}: {e}")
                continue
            if not changed:
                continue
            summary["updated"] += 1
            if info["newly_closed"]:
                summary["closed"] += 1
                if (info["net"] or 0) > 0:
                    summary["wins"] += 1
                else:
                    summary["losses"] += 1
            if info["newly_unresolved"]:
                summary["unresolved"] += 1
            if info["checkpoint_added"]:
                summary["checkpoints_added"] += 1
    if summary["updated"]:
        save_signal_history(history)
    return summary


def compute_performance_stats(coin=None):
    """
    Win/loss is determined by the SIGN of return_pct_net (the return after
    the spread), uniformly for all outcomes — stop is always net<0 by construction,
    target_1/target_2 always net>0 (calculate_entry_levels in utils/signals.py
    already rejects targets that do not cover the spread — check_entry_filters),
    while close (closing by time) can be anything, and this is exactly where
    the check "would the move have covered the spread" really works.

    unresolved goes into neither the numerator nor the denominator.
    """
    history = load_signal_history()
    if coin:
        tagged = [(coin, r) for r in _dedupe_records(history.get(coin, []))]
    else:
        tagged = [(c, r) for c, recs in history.items() for r in _dedupe_records(recs)]

    returns, wins, losses, unresolved_count = [], 0, 0, 0
    for c, r in tagged:
        if r.get("outcome") == "unresolved":
            unresolved_count += 1
            continue
        net = r.get("return_pct_net")
        if net is None:
            continue  # not resolved yet
        returns.append((c, net))
        if net > 0:
            wins += 1
        else:
            losses += 1

    total_signals, resolved = len(tagged), len(returns)
    if resolved == 0:
        return {"coin": coin, "total_signals": total_signals, "resolved": 0, "unresolved": unresolved_count}

    values = [v for _, v in returns]
    avg_return = statistics.mean(values)
    std_return = statistics.stdev(values) if resolved > 1 else 0.0
    sharpe_like = (avg_return / std_return) if std_return > 0 else None
    gains = sum(v for v in values if v > 0)
    losses_sum = abs(sum(v for v in values if v < 0))
    profit_factor = (gains / losses_sum) if losses_sum > 0 else None
    best_coin, best_return = max(returns, key=lambda x: x[1])
    worst_coin, worst_return = min(returns, key=lambda x: x[1])

    return {
        "coin": coin, "total_signals": total_signals, "resolved": resolved, "unresolved": unresolved_count,
        "wins": wins, "losses": losses,
        "win_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) else None,
        "avg_return_pct": round(avg_return, 2), "std_return_pct": round(std_return, 2),
        "sharpe_like": round(sharpe_like, 2) if sharpe_like is not None else None,
        "profit_factor": round(profit_factor, 2) if profit_factor is not None else None,
        "best_trade_pct": round(best_return, 2), "best_trade_coin": best_coin,
        "worst_trade_pct": round(worst_return, 2), "worst_trade_coin": worst_coin,
    }
