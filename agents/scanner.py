"""
/scan — runs the full pipeline over several coins sequentially and
builds a single ranked report. A port of agents/scanner.py from stock_agent2
(scan_tickers/build_scan_report), with no overlapping LLM calls — the same
build_trade_report as /now, just one after another.
"""
import time

from config import LONG_THRESHOLD
from data.correlation import compute_btc_correlation
from utils.helpers import safe_format_float

SCAN_PAUSE_SEC = 1.0


def _no_trade_reason(plan: dict) -> str:
    """check_entry_filters always returns a reason for NO_TRADE (see
    utils/signals.py) — either "No signal" (the score did not reach the threshold)
    or a specific filter rejection reason (spread/R:R/volume/volatility).
    "No signal" by itself is uninformative without a number — we add the score."""
    reason = plan.get("entry_filter_reason", "")
    score = plan.get("direction_score", 0)
    if reason == "No signal" or not reason:
        return f"score {score:+d}, did not reach ±{LONG_THRESHOLD}"
    return reason


def scan_coins(coins: list) -> list:
    """Returns a list of (coin, trader_plan, error) for each coin."""
    from main import build_trade_report  # lazy import, see main.py

    results = []
    for i, coin in enumerate(coins):
        try:
            _, trader_plan = build_trade_report(coin)
            results.append((coin, trader_plan, None))
        except Exception as e:
            results.append((coin, None, str(e)))
        if i < len(coins) - 1:
            time.sleep(SCAN_PAUSE_SEC)
    return results


def build_scan_report(results: list) -> str:
    ranked = [(coin, plan) for coin, plan, err in results if plan and plan.get("direction") != "NO_TRADE"]
    ranked.sort(key=lambda x: abs(x[1].get("direction_score", 0)), reverse=True)

    lines = [f"*/scan* — {len(results)} coins"]

    if ranked:
        # Several signals highly correlated with BTC are probably ONE
        # move, not independent findings (spec, item 5: "do not show
        # several highly correlated signals as independent").
        same_direction_alts = [
            coin for coin, plan in ranked
            if coin != "BTC" and plan.get("direction", "").endswith(("LONG", "SHORT"))
        ]
        high_corr_alts = []
        if len(same_direction_alts) > 1:
            for coin in same_direction_alts:
                corr = compute_btc_correlation(coin)
                if corr.get("high_correlation"):
                    high_corr_alts.append((coin, corr["correlation"]))

        lines.append("\n🏆 *Signals (by score strength):*")
        for coin, plan in ranked:
            lines.append(
                f"   {coin}: *{plan['direction']}* (score {plan.get('direction_score', 0):+d}, "
                f"conf. {plan.get('confidence', 0):.0%}, R:R 1:{plan.get('risk_reward_1', '?')})"
            )

        if len(high_corr_alts) > 1:
            names = ", ".join(f"{c} ({r:+.2f})" for c, r in high_corr_alts)
            lines.append(f"\n⚠️ *{len(high_corr_alts)} signals are highly correlated with BTC ({names})* — "
                         f"this is probably one market move, not {len(high_corr_alts)} independent findings.")
    else:
        lines.append("\nNo signals — all coins NO_TRADE.")

    lines.append("\n📋 *All coins:*")
    for coin, plan, err in results:
        if err:
            lines.append(f"   {coin}: ⚠️ error — {err[:100]}")
            continue
        direction = plan.get("direction", "?")
        score = plan.get("direction_score", 0)
        price = safe_format_float(plan.get("entry_price") or (plan.get("features") or {}).get("price"), 6)
        if direction == "NO_TRADE":
            lines.append(f"   {coin}: {direction} ({score:+d}) @ {price} — {_no_trade_reason(plan)}")
        else:
            lines.append(f"   {coin}: {direction} ({score:+d}) @ {price}")

    return "\n".join(lines)
