from config import TRADER_MODEL, LONG_THRESHOLD
from utils.schemas import TRADER_SCHEMA
from utils.ollama import call_ollama_json
from agents.prompts import build_trader_prompt
from utils.signals import (
    compute_factor_scores, determine_direction, check_entry_filters,
    calculate_position_size, compute_confidence, format_signals,
    calculate_entry_levels, format_no_trade_report,
)
from utils.helpers import safe_format_float


def run_trader(coin, indicators, daily_levels, volume_profile, liquidity, orderflow=None,
                stumpy_text="", correlation_text="", funding_text="", volume_ratio=None,
                stumpy_result=None, funding_data=None):
    """
    Gemma -> description only (market summary, opportunities, analogy, risks,
    commentary). Python -> direction, levels, position size (utils/signals.py).

    ONE call per coin: previously a separate Junior agent worked right here,
    describing the same numbers BEFORE the decision was computed; it never entered scoring
    and went only into this prompt. Merged here — the model sees both the market and the
    decision already made, and there is one call fewer.
    """
    print(f"\n📈 DAY TRADER ({coin})")
    volume_ratio = volume_ratio or {}

    # Take the structured STUMPY result directly (do not parse text) —
    # the same approach and the same rationale as in agents/trader.py in stock_agent2:
    # the 3d horizon is the main one, a regex on the text risked grabbing the 1-day one.
    stumpy_data = {}
    if stumpy_result:
        if stumpy_result.get('win_rate_3d') is not None:
            stumpy_data['win_rate'] = stumpy_result['win_rate_3d']
        if stumpy_result.get('avg_ret_3d') is not None:
            stumpy_data['avg_return'] = stumpy_result['avg_ret_3d']
        if stumpy_result.get('n_3d') is not None:
            stumpy_data['n_samples'] = stumpy_result['n_3d']
        # The coin's base rate — without it win_rate would be compared with 50% and
        # the factor would measure market drift, not the pattern (see utils/signals.py).
        stumpy_data['base_win_rate'] = stumpy_result.get('base_win_rate')
        stumpy_data['base_avg_return'] = stumpy_result.get('base_avg_return')

    scores, reasons = compute_factor_scores(
        indicators=indicators, daily_levels=daily_levels, orderflow=orderflow,
        stumpy_data=stumpy_data, volume_ratio=volume_ratio, funding_data=funding_data,
    )
    direction, total_score = determine_direction(scores)
    print(f"\n   {format_signals(scores, reasons)}")
    print(f"   🎯 Direction: {direction} (score: {total_score:+d})")

    _of = orderflow or {}
    features = {
        "price": indicators.get("price"), "rsi": indicators.get("rsi"),
        "macd": indicators.get("macd"), "macd_signal": indicators.get("macd_signal"),
        "bb_upper": indicators.get("bb_upper"), "bb_lower": indicators.get("bb_lower"),
        "volume_ratio": volume_ratio.get("volume_ratio"),
        "vwap_position": daily_levels.get("vwap_position"),
        "distance_to_vwap": daily_levels.get("distance_to_vwap"),
        "trend_5h": daily_levels.get("trend_5h"), "trend_15h": daily_levels.get("trend_15h"),
        "atr_hourly": daily_levels.get("atr_hourly"),
        "stumpy_win_rate": stumpy_data.get("win_rate"), "stumpy_avg_return": stumpy_data.get("avg_return"),
        "stumpy_n": stumpy_data.get("n_samples"),
        "taker_imbalance": _of.get("taker_imbalance"), "large_player_active": _of.get("large_player_active"),
        "delta_divergence": _of.get("delta_divergence"), "absorption_detected": _of.get("absorption_detected"),
        "funding_rate_pct": (funding_data or {}).get("funding_rate_pct"),
    }

    trader_prompt = build_trader_prompt(
        coin, indicators, daily_levels, volume_profile, liquidity, orderflow,
        stumpy_text=stumpy_text, volume_ratio=volume_ratio,
        correlation_text=correlation_text, funding_text=funding_text,
        direction_hint=direction, total_score=total_score,
    )
    print(f"   📝 PROMPT: {len(trader_prompt)} chars")
    gemma_output = call_ollama_json(trader_prompt, TRADER_MODEL, json_schema=TRADER_SCHEMA, temperature=0)
    if gemma_output.get("_error"):
        print(f"   ⚠️ Model description unavailable: {gemma_output.get('_error_message')}")
        gemma_output = {"executive_summary": "", "opportunities": [], "historical_analogy": "",
                         "risks": [], "commentary": "",
                         "_error": True, "_error_message": gemma_output.get("_error_message")}

    levels = calculate_entry_levels(direction=direction, price=indicators['price'], daily_levels=daily_levels,
                                     volume_profile=volume_profile, liquidity=liquidity)

    entry_ok, entry_reason = check_entry_filters(direction=direction, daily_levels=daily_levels, trader_plan=levels,
                                                  indicators=indicators, volume_ratio=volume_ratio)

    if not entry_ok:
        print(f"\n   ⚠️ FILTER NOT PASSED: {entry_reason}")
        report = format_no_trade_report(scores=scores, reasons=reasons, total_score=total_score,
                                         indicators=indicators, daily_levels=daily_levels,
                                         volume_ratio=volume_ratio, threshold=LONG_THRESHOLD,
                                         filter_reason=entry_reason)
        print(f"\n{report}")
        return {
            "direction": "NO_TRADE", "direction_score": total_score, "scores": scores, "reasons": reasons,
            "features": features, "gemma": gemma_output, "entry_price": None, "stop_loss": None,
            "target_1": None, "target_2": None, "position_size_pct": 0, "confidence": 0.0,
            "entry_filter_reason": entry_reason, "reasoning": f"Entry filter not passed: {entry_reason}",
            "no_trade_report": report,
        }

    position_size = calculate_position_size(direction, total_score, entry_ok)
    confidence = compute_confidence(direction, total_score, entry_ok)

    print(f"\n   ✅ DIRECTION: {direction} (score: {total_score:+d})")
    print(f"   ✅ ENTRY: {safe_format_float(levels.get('entry_price'), 6)} | STOP: {safe_format_float(levels.get('stop_loss'), 6)}")
    print(f"   ✅ TARGETS: {safe_format_float(levels.get('target_1'), 6)} / {safe_format_float(levels.get('target_2'), 6)}")
    print(f"   ✅ R:R (after spread) 1:{levels.get('risk_reward_1', '?')} / 1:{levels.get('risk_reward_2', '?')}")
    print(f"   ✅ POSITION: {position_size}% | CONFIDENCE: {confidence:.0%}")

    return {
        "direction": direction, "direction_score": total_score, "scores": scores, "reasons": reasons,
        "features": features, "gemma": gemma_output,
        "entry_price": levels.get('entry_price'), "stop_loss": levels.get('stop_loss'),
        "target_1": levels.get('target_1'), "target_2": levels.get('target_2'),
        "risk_reward_1": levels.get('risk_reward_1', 0), "risk_reward_2": levels.get('risk_reward_2', 0),
        "spread_cost_pct": levels.get('spread_cost_pct'),
        "position_size_pct": position_size, "confidence": confidence,
        "reasoning": f"{direction}. Total score: {total_score:+d}.",
    }
