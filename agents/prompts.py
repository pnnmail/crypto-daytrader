"""
The prompt for the single LLM call (description + commentary on the decision).
Ported from the stock version with the news block removed — the crypto spec has
no news source. All numbers in the prompt are already computed by Python, the model only interprets them.
The prompt text itself is kept in Russian on purpose: prompt language is one of the measured variables (see agents/decider.py).
"""
from config import SPREAD_ROUND_TRIP_PCT, REGIME_BRIEF


def build_trader_prompt(coin: str, indicators: dict, daily_levels: dict, volume_profile: dict, liquidity: dict,
                         orderflow: dict = None, stumpy_text: str = "", volume_ratio: dict = None,
                         correlation_text: str = "", funding_text: str = "",
                         direction_hint: str = None, total_score: int = None) -> str:
    levels, vp, liq, price = daily_levels, volume_profile, liquidity, indicators['price']
    volume_ratio = volume_ratio or {}

    orderflow_block = ""
    if orderflow and orderflow.get('orderflow_bias'):
        of_bias = orderflow.get('orderflow_bias', '?')
        of_emoji = 'BULL' if of_bias == 'bullish' else 'BEAR' if of_bias == 'bearish' else 'NEUTRAL'
        orderflow_block = f"""
ORDER FLOW:
* Поток: {of_emoji}
* Крупный дисбаланс: {'ДА' if orderflow.get('large_player_active') else 'НЕТ'}
* Дивергенция: {'ДА' if orderflow.get('delta_divergence') else 'НЕТ'}
* Поглощение: {'ДА' if orderflow.get('absorption_detected') else 'НЕТ'}
* Дисбаланс тейкеров: {orderflow.get('taker_imbalance', 0):+.2f}
"""

    stumpy_block = f"\n{stumpy_text}\n" if stumpy_text else ""
    correlation_block = f"\n{correlation_text}\n" if correlation_text else ""
    funding_block = f"\n{funding_text}\n" if funding_text else ""

    hint_block = ""
    if direction_hint and direction_hint != "NO_TRADE":
        hint_block = f"""
💡 *СИСТЕМА СИГНАЛОВ (Python):*
• Направление: {direction_hint}
• Общий балл: {total_score:+d}
• Учитывай это при выборе направления, но не копируй слепо.
• Если твой анализ противоречит системе — объясни почему.

"""
    elif direction_hint == "NO_TRADE":
        hint_block = f"""
💡 *СИСТЕМА СИГНАЛОВ (Python):*
• Направление: NO_TRADE (балл: {total_score:+d})
• Сигналы противоречивы или потенциал не покрывает спред ({SPREAD_ROUND_TRIP_PCT}% круг) — не открывай сделку.

"""

    bb_lower, bb_upper = indicators.get('bb_lower', 0), indicators.get('bb_upper', 0)
    bb_position = (price - bb_lower) / (bb_upper - bb_lower) * 100 if bb_upper != bb_lower else 50.0
    trend_divergence = levels.get('trend_divergence', False)

    return f"""Ты - профессиональный дейтрейдер на криптобирже (Coinbase, 24/7). Решение по {coin} УЖЕ ПРИНЯТО
системой (направление, уровни входа/стопа/целей и размер позиции рассчитаны детерминированно на
Python, с учётом спреда Robinhood {SPREAD_ROUND_TRIP_PCT}% за круг). Твоя задача — НЕ принимать решение
и НЕ предлагать свой план, а описать ситуацию и прокомментировать уже принятое решение.

{orderflow_block}{stumpy_block}{correlation_block}{funding_block}
РЫНОЧНЫЕ ДАННЫЕ (все числа уже рассчитаны, используй их):
* Текущая цена: {price} ({indicators.get('price_change', 0):+.2f}%)
* RSI: {indicators.get('rsi', '?')} | MACD: {indicators.get('macd', '?')} / сигнал: {indicators.get('macd_signal', '?')}
* Bollinger: {bb_lower} — {bb_upper} (цена на {bb_position:.0f}% высоты)
* Объём: {volume_ratio.get('volume_ratio', '?')}x ({volume_ratio.get('basis', 'нет данных')})
* VWAP: {levels.get('vwap', '?')} (цена {'ВЫШЕ' if levels.get('price_above_vwap') else 'НИЖЕ'} VWAP)
* Pivot: {levels.get('pivot', '?')} | R1: {levels.get('r1', '?')} | R2: {levels.get('r2', '?')}
* S1: {levels.get('s1', '?')} | S2: {levels.get('s2', '?')}
* ATR (часовой): {levels.get('atr_hourly', '?')}
* POC (макс объём): {vp.get('poc', '?')}
* Тренд 5ч: {levels.get('trend_5h', '?')} | 15ч: {levels.get('trend_15h', '?')}{' ⚠️ РАСХОЖДЕНИЕ' if trend_divergence else ''}
* Граница суток: {levels.get('day_boundary', 'UTC 00:00')}

ЛИКВИДНОСТЬ:
* Стопы выше: {liq.get('stops_above', []) if liq.get('stops_above') else 'нет'}
* Стопы ниже: {liq.get('stops_below', []) if liq.get('stops_below') else 'нет'}
* Пред. бар High: {levels.get('prev_high', '?')} | Low: {levels.get('prev_low', '?')}

{hint_block}
{REGIME_BRIEF}

Это ВАЖНО для твоего комментария: без этих фактов легко написать «нисходящий
тренд, риск падения» под сигналом LONG и получить отчёт, который сам себе
противоречит. Решение системы контртрендовое ПО ЗАМЕРУ, а не по недосмотру.

ЗАПОЛНИ РОВНО ПЯТЬ ПОЛЕЙ:

1. executive_summary — 3-5 предложений: что сейчас происходит с монетой по
   цифрам выше (тренд, импульс, объём, положение относительно уровней).
   {'⚠️ Тренды 5ч и 15ч расходятся — обязательно упомяни это.' if trend_divergence else ''}

2. opportunities — 2-3 конкретные возможности для дейтрейдинга по этим данным.

3. historical_analogy — похожая ситуация из истории крипторынка (1-2 предложения).
   Если аналогия не приходит на ум — короткое "нет явной аналогии", не выдумывай.

4. risks — 2-4 КОНКРЕТНЫХ риска именно для этой ситуации, не общие слова.

5. commentary — 3-5 предложений: какие факторы поддерживают решение системы,
   а какие ему противоречат, и какого подтверждения не хватает.

Новостей у тебя нет — не придумывай новостной фон, опирайся только на цифры выше.

⚠️ ВАЖНО: НЕ предлагай своё направление/вход/стоп/цели/размер позиции — они уже
рассчитаны и твой план не будет использован. НЕ придумывай числа и новости, которых нет в данных выше.

ОТВЕЧАЙ ТОЛЬКО JSON, НА РУССКОМ ЯЗЫКЕ.
"""
