"""
The LLM decider for the experiment (utils/experiment.py).

The only place where the model MAKES a decision rather than describing a finished one.
The result never goes into live signals — only into a separate experiment
store, where it is compared with the deterministic calculation.

TWO MANDATORY CONDITIONS, otherwise the experiment is invalid:

1. The model does NOT SEE the calculation's decision. The live trader prompt has
   a direction_hint with a ready direction — here it is not and cannot be present,
   otherwise the model would simply copy the answer and the comparison would be meaningless.

2. A model error is an ERROR, not NO_TRADE. If after a retry the answer is
   invalid, _error is set and the point is excluded from the denominator. Otherwise failures
   would look like caution and inflate the model's statistics.
"""
from config import (TRADER_MODEL, SPREAD_ROUND_TRIP_PCT, DECIDER_CONFIDENCE_THRESHOLD,
                    DECIDER_VARIANTS, REGIME_BRIEF)
from utils.schemas import DECISION_SCHEMA
from utils.ollama import call_ollama_json


def build_decision_prompt(coin, indicators, daily_levels, volume_profile, liquidity,
                           orderflow, volume_ratio, stumpy_text="", correlation_text="",
                           funding_text="", book_text="", regime_text="") -> str:
    levels, vp, vr = daily_levels, volume_profile, (volume_ratio or {})
    price = indicators["price"]
    bb_lower, bb_upper = indicators.get("bb_lower", 0), indicators.get("bb_upper", 0)
    bb_pos = (price - bb_lower) / (bb_upper - bb_lower) * 100 if bb_upper != bb_lower else 50.0

    of_block = ""
    if orderflow:
        of_block = f"""
ПОТОК ЗАЯВОК (реальные сделки):
* Дисбаланс тейкеров: {orderflow.get('taker_imbalance', 0):+.2f} (от -1 продажи до +1 покупки)
* Крупный дисбаланс: {'ДА' if orderflow.get('large_player_active') else 'нет'}
* Дивергенция потока и цены: {'ДА' if orderflow.get('delta_divergence') else 'нет'}
* Поглощение: {'ДА' if orderflow.get('absorption_detected') else 'нет'}
"""

    return f"""Ты — крипто-трейдер. Прими решение по {coin} на горизонт 4 часа: входить в LONG,
в SHORT, или не входить вовсе.

ЦЕНА: {price} ({indicators.get('price_change', 0):+.2f}% за час)
RSI: {indicators.get('rsi')} | MACD: {indicators.get('macd')} / сигнал: {indicators.get('macd_signal')}
Bollinger: {bb_lower} — {bb_upper} (цена на {bb_pos:.0f}% высоты полосы)
Объём: {vr.get('volume_ratio', '?')}x ({vr.get('basis', 'нет данных')})

УРОВНИ (граница суток {levels.get('day_boundary', 'UTC 00:00')}):
* VWAP: {levels.get('vwap', '?')} — цена {'ВЫШЕ' if levels.get('price_above_vwap') else 'НИЖЕ'} ({levels.get('distance_to_vwap', 0):+.2f}%)
* Pivot: {levels.get('pivot', '?')} | R1: {levels.get('r1', '?')} | S1: {levels.get('s1', '?')}
* ATR (часовой): {levels.get('atr_hourly', '?')}
* POC: {vp.get('poc', '?')}
* Тренд 5ч: {levels.get('trend_5h', '?')} | 15ч: {levels.get('trend_15h', '?')}
{of_block}{stumpy_text}
{correlation_text}
{funding_text}
{book_text}
{regime_text}

ГОРИЗОНТ: ровно 4 часа, позиция закрывается по времени.

ОТВЕТЬ: КУДА, ПО-ТВОЕМУ, ПОЙДЁТ ЦЕНА за эти 4 часа.
- direction: LONG если ожидаешь рост, SHORT если падение.
- confidence: 0..1 — насколько ты в этом уверен. Слабый перевес это 0.5-0.6,
  уверенное чтение ситуации — 0.8+. Не занижай из осторожности: решение
  "торговать или нет" принимается ОТДЕЛЬНО по твоей уверенности, твоя задача —
  честно оценить направление и свою убеждённость в нём.
- NO_TRADE выбирай только если данные буквально симметричны и ни одна сторона
  не выглядит вероятнее другой.
- key_reason: главная причина, 1-2 предложения на русском.

Никаких чисел, которых нет в данных выше, не придумывай."""


def build_decision_prompt_en(coin, indicators, daily_levels, volume_profile, liquidity,
                              orderflow, volume_ratio, stumpy_text="", correlation_text="",
                              funding_text="", book_text="", regime_text="") -> str:
    """The English version of the prompt — a LITERAL counterpart of the Russian one.

    Composition, block order, wording and answer requirements match
    one to one. This is a condition of the experiment: the book3 variant must differ from book2
    ONLY in language, not in the amount of hints. Any "improvement" of the wording
    here would make the comparison meaningless — so the text is translated, not
    rewritten.
    """
    levels, vp, vr = daily_levels, volume_profile, (volume_ratio or {})
    price = indicators["price"]
    bb_lower, bb_upper = indicators.get("bb_lower", 0), indicators.get("bb_upper", 0)
    bb_pos = (price - bb_lower) / (bb_upper - bb_lower) * 100 if bb_upper != bb_lower else 50.0

    of_block = ""
    if orderflow:
        of_block = f"""
ORDER FLOW (actual trades):
* Taker imbalance: {orderflow.get('taker_imbalance', 0):+.2f} (from -1 selling to +1 buying)
* Large imbalance: {'YES' if orderflow.get('large_player_active') else 'no'}
* Flow/price divergence: {'YES' if orderflow.get('delta_divergence') else 'no'}
* Absorption: {'YES' if orderflow.get('absorption_detected') else 'no'}
"""

    return f"""You are a crypto trader. Decide on {coin} for a 4-hour horizon: enter LONG,
enter SHORT, or stay out.

PRICE: {price} ({indicators.get('price_change', 0):+.2f}% over the last hour)
RSI: {indicators.get('rsi')} | MACD: {indicators.get('macd')} / signal: {indicators.get('macd_signal')}
Bollinger: {bb_lower} — {bb_upper} (price at {bb_pos:.0f}% of band height)
Volume: {vr.get('volume_ratio', '?')}x ({vr.get('basis', 'no data')})

LEVELS (day boundary {levels.get('day_boundary', 'UTC 00:00')}):
* VWAP: {levels.get('vwap', '?')} — price is {'ABOVE' if levels.get('price_above_vwap') else 'BELOW'} ({levels.get('distance_to_vwap', 0):+.2f}%)
* Pivot: {levels.get('pivot', '?')} | R1: {levels.get('r1', '?')} | S1: {levels.get('s1', '?')}
* ATR (hourly): {levels.get('atr_hourly', '?')}
* POC: {vp.get('poc', '?')}
* Trend 5h: {levels.get('trend_5h', '?')} | 15h: {levels.get('trend_15h', '?')}
{of_block}{stumpy_text}
{correlation_text}
{funding_text}
{book_text}
{regime_text}

HORIZON: exactly 4 hours, the position is closed by time.

ANSWER: WHERE DO YOU THINK THE PRICE WILL GO over these 4 hours.
- direction: LONG if you expect a rise, SHORT if a fall.
- confidence: 0..1 — how sure you are. A slight edge is 0.5-0.6, a confident
  reading of the situation is 0.8+. Do not understate out of caution: the
  decision "to trade or not" is made SEPARATELY based on your confidence,
  your task is to honestly assess the direction and your conviction in it.
- Choose NO_TRADE only if the data is literally symmetric and neither side
  looks more likely than the other.
- key_reason: the main reason, 1-2 sentences in English.

Do not invent any numbers that are not in the data above."""


def _run_one(coin, variant, ctx, book_text, regime_text, lang="ru"):
    """One decider variant. Returns a decision or {"_error": True}.

    lang="en" builds the whole prompt in English (the book3 variant). Mixing
    a Russian prompt with English quotes from the book is not allowed: the book2/book3 pair
    is designed as "the same text, a different working language", and a half translation
    would make the difference uninterpretable.
    """
    builder = build_decision_prompt_en if lang == "en" else build_decision_prompt
    prompt = builder(
        coin, ctx["indicators"], ctx["daily_levels"], ctx["volume_profile"], ctx["liquidity"],
        ctx["orderflow"], ctx["volume_ratio"], ctx["stumpy_text"], ctx["correlation_text"],
        ctx["funding_text"], book_text=book_text, regime_text=regime_text)
    result = call_ollama_json(prompt, TRADER_MODEL, json_schema=DECISION_SCHEMA, temperature=0)
    if result.get("_error"):
        print(f"   ❌ [{variant}] no decision received: {result.get('_error_message')}")
        return {"_error": True, "_error_message": result.get("_error_message"), "variant": variant}

    raw_dir = (result.get("direction") or "NO_TRADE").upper()
    conf = result.get("confidence")
    result["variant"] = variant
    result["raw_direction"] = raw_dir
    # The threshold is applied IN CODE, not by asking in the prompt: in p2 the prompt itself
    # talked the model out of trading, and it gave NO_TRADE 100 times out of 100.
    if raw_dir != "NO_TRADE" and (conf is None or conf < DECIDER_CONFIDENCE_THRESHOLD):
        result["direction"] = "NO_TRADE"
        result["downgraded_from"] = raw_dir
    print(f"   [{variant:6}] opinion {raw_dir:9} decision {result.get('direction'):9} "
          f"conf. {conf} — {result.get('key_reason', '')[:90]}")
    return result


def run_deciders(coin, indicators, daily_levels, volume_profile, liquidity, orderflow,
                  volume_ratio, stumpy_text="", correlation_text="", funding_text=""):
    """
    Runs ALL decider variants on THE SAME point and returns
    {variant: decision}.

    Simultaneity is essential: variants are compared on the same market, without
    a time shift. Running them one after another over days would mean comparing
    not hints but different market periods.

      plain    — numbers only;
      plain_en — numbers only, but the prompt ENTIRELY in English;
      book     — plus Schwager's principles (classic technical analysis, RU);
      regime   — plus Schwager and the measured-regime briefing;
      book2    — plus Chan's principles in Russian (translated);
      book3    — the same Chan principles, but the WHOLE call in English;
      book4    — plus Aziz's principles (intraday trading around VWAP).

    Splitting book and regime is intentional: the book is general and about stocks, while the briefing is
    our crypto-specific measurement. Merged together, they would not tell us what
    exactly had an effect.

    plain_en is the CONTROL for book3, without which book3 is uninterpretable: book3
    differs from book2 in two things at once — the language of the book AND the language of the prompt.
    The plain/plain_en pair isolates the pure language effect, the plain_en/book3 pair —
    the book effect within English. The control was added after the first 8 points
    of p9, where three DIFFERENT Russian books did not change a single decision (8 of 8
    matches with plain), while the English variant changed two of eight.
    """
    from utils.knowledge import (compute_tags, query_by_tags,
                                  SCHWAGER, CHAN_RU, CHAN_EN, AZIZ)

    print(f"\n🧪 EXPERIMENT: model decisions ({coin})")
    ctx = {"indicators": indicators, "daily_levels": daily_levels,
           "volume_profile": volume_profile, "liquidity": liquidity, "orderflow": orderflow,
           "volume_ratio": volume_ratio, "stumpy_text": stumpy_text,
           "correlation_text": correlation_text, "funding_text": funding_text}

    # Market-state tags are computed ONCE and fed into all books: otherwise
    # the variants would search by different states and there would be nothing to compare.
    tags = compute_tags(indicators, daily_levels, orderflow, volume_ratio)
    books = {}
    if any(v in ("book", "regime") for v in DECIDER_VARIANTS):
        books["book"] = query_by_tags(tags, collection=SCHWAGER, lang="ru")
    if "book2" in DECIDER_VARIANTS:
        books["book2"] = query_by_tags(
            tags, collection=CHAN_RU, lang="ru",
            header="ПРИНЦИПЫ ИЗ КНИГИ (Чан, «Алгоритмическая торговля»), "
                   "подобранные под текущее состояние:")
    if "book4" in DECIDER_VARIANTS:
        books["book4"] = query_by_tags(
            tags, collection=AZIZ, lang="ru",
            header="ПРИНЦИПЫ ИЗ КНИГИ (Азиз, «Руководство по дейтрейдингу»), "
                   "подобранные под текущее состояние:")
    if "book3" in DECIDER_VARIANTS:
        books["book3"] = query_by_tags(
            tags, collection=CHAN_EN, lang="en",
            header="PRINCIPLES FROM THE BOOK (Chan, Algorithmic Trading), "
                   "selected for the current market state:")
    if books:
        print(f"   📚 principles selected for tags: {tags[:6]}")

    out = {}
    for variant in DECIDER_VARIANTS:
        bt = books.get("book", "") if variant in ("book", "regime") else books.get(variant, "")
        rt = REGIME_BRIEF if variant == "regime" else ""
        out[variant] = _run_one(coin, variant, ctx, bt, rt,
                                 lang="en" if variant in ("book3", "plain_en") else "ru")
    return out
