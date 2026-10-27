"""
Knowledge from books for the LLM decider (experiment, see utils/experiment.py).

The base is the same as in the stock bot: 389 chunks from Schwager's "Technical Analysis:
The Complete Course" (Russian edition), labelled by ENTRY / RISK / CONFIRMATION / EXIT categories
with metadata (concept, market phase, action, quality score).

A copy rather than a shared volume with the stock bot, on purpose: the projects are independent by
design, and the base is static (built once from a PDF) and weighs 4 MB —
duplication is cheaper than coupling the projects.

WHAT TO EXPECT. The book gives reasoning PRINCIPLES (when to enter, what to fear,
what counts as confirmation), but it is about stocks and was written before crypto. The key
measured fact — that in crypto at the 4-hour horizon a deviation from VWAP
reverts rather than continues — is not there. So the book is tested separately
from the regime briefing, not together: otherwise one cannot tell what exactly helped.

IMPORTANT ABOUT CONTENT (found on 2026-08-16 after retrieval was fixed). Classic
stock technical analysis is TREND-FOLLOWING by nature, and the selection honestly
reflects that. In the state "price below VWAP, lower Bollinger band, downward
trend" the book returns "sell signals are generated when the market closes
below the low" and "buyers may want to close long positions" — i.e. it
advises selling exactly where we measured a reversion up.

This is NOT a reason to throw it out: the book variant exists precisely to test what
a high-quality but market-inappropriate knowledge source does to decisions.
But the results must be read with this in mind — there is no reason to expect the book
to improve things in a mean-reverting market.
"""
import os

import requests

from config import OLLAMA_URL, EMBED_MODEL, CHROMA_DB_PATH, RAG_N_RESULTS

# A collection for EACH book separately. They must not be mixed: the book variant must
# see only Schwager, book2/book3 — only Chan, otherwise comparing books
# loses its meaning.
SCHWAGER = "trading_knowledge"
CHAN_RU = "trading_knowledge_chan_ru"
CHAN_EN = "trading_knowledge_chan_en"
AZIZ = "trading_knowledge_aziz"

_collections = {}
_init_failed = set()


def _get_collection(name: str = SCHWAGER):
    """Lazy initialisation by collection name.

    chromadb pulls heavy dependencies, and the decider can work without a book
    (the "bare" variant). If a particular base is unavailable — return None and
    silently work without knowledge, without breaking the pipeline. Failure is remembered BY NAME:
    a missing Chan collection must not disable Schwager.
    """
    if name in _collections:
        return _collections[name]
    if name in _init_failed:
        return None
    try:
        import chromadb
        from chromadb import Documents, EmbeddingFunction, Embeddings

        class OllamaEmbedding(EmbeddingFunction):
            def __call__(self, input: Documents) -> Embeddings:
                out = []
                for text in input:
                    r = requests.post(OLLAMA_URL.replace("generate", "embeddings"),
                                      json={"model": EMBED_MODEL, "prompt": text[:1000]},
                                      timeout=60).json()
                    vec = r.get("embedding") or (r.get("embeddings") or [None])[0]
                    out.append(vec if vec else [0.0] * 768)
                return out

        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        col = client.get_or_create_collection(name, embedding_function=OllamaEmbedding())
        if col.count() == 0:
            print(f"   ⚠️ Collection {name} is empty — the variant works without a book")
            _init_failed.add(name)
            return None
        _collections[name] = col
        print(f"   📚 Knowledge base {name}: {col.count()} chunks")
        return col
    except Exception as e:
        print(f"   ⚠️ Base {name} unavailable ({e}) — the decider works without a book")
        _init_failed.add(name)
        return None


# Tag weight when selecting chunks: the more specific the market state, the more important
# it is to find a principle about it. Ported from the stock bot's knowledge router, with
# tags inapplicable to crypto removed (earnings_*, near_52w_*).
TAG_WEIGHTS = {
    "delta_divergence": 5, "absorption": 4, "large_player_active": 4,
    "orderflow_bullish": 4, "orderflow_bearish": 4,
    "below_vwap": 3, "above_vwap": 3, "at_vwap": 2,
    "downtrend": 3, "uptrend": 3, "capitulation_possible": 3, "exhaustion_possible": 3,
    "macd_bearish": 2, "macd_bullish": 2,
    "overbought": 2, "oversold": 2, "depressed_rsi": 2,
    "near_bb_upper": 2, "near_bb_lower": 2,
    "low_volume": 2, "ultra_low_volume": 3, "high_volume": 2,
    "high_volatility": 2, "stops_above_cluster": 2, "stops_below_cluster": 2,
    "neutral_rsi": 1, "elevated_rsi": 1,
}


def compute_tags(indicators, daily_levels, orderflow, volume_ratio):
    """The market state as tags — principles in the book are searched by them."""
    tags = []
    rsi = indicators.get("rsi", 50)
    price = indicators.get("price", 0)

    if rsi > 70:
        tags.append("overbought")
    elif rsi > 60:
        tags.append("elevated_rsi")
    elif rsi < 30:
        tags.append("oversold")
    elif rsi < 40:
        tags.append("depressed_rsi")
    else:
        tags.append("neutral_rsi")

    pos = daily_levels.get("vwap_position")
    tags.append({"above": "above_vwap", "below": "below_vwap"}.get(pos, "at_vwap"))

    tags.append("macd_bullish" if indicators.get("macd", 0) > indicators.get("macd_signal", 0)
                else "macd_bearish")

    bb_l, bb_u = indicators.get("bb_lower", 0), indicators.get("bb_upper", 0)
    if bb_u > bb_l:
        bb_pos = (price - bb_l) / (bb_u - bb_l) * 100
        if bb_pos > 80:
            tags.append("near_bb_upper")
        elif bb_pos < 20:
            tags.append("near_bb_lower")

    for name, tag in (("trend_5h", None), ("trend_15h", None)):
        t = daily_levels.get(name)
        if t == "bullish":
            tags.append("uptrend")
        elif t == "bearish":
            tags.append("downtrend")

    vr = (volume_ratio or {}).get("volume_ratio", 1) or 1
    if vr < 0.5:
        tags.append("ultra_low_volume")
    elif vr < 0.8:
        tags.append("low_volume")
    elif vr > 1.5:
        tags.append("high_volume")

    of = orderflow or {}
    if of.get("orderflow_bias") == "bullish":
        tags.append("orderflow_bullish")
    elif of.get("orderflow_bias") == "bearish":
        tags.append("orderflow_bearish")
    if of.get("delta_divergence"):
        tags.append("delta_divergence")
    if of.get("absorption_detected"):
        tags.append("absorption")
    if of.get("large_player_active"):
        tags.append("large_player_active")

    atr = daily_levels.get("atr_hourly")
    if atr and price and atr / price * 100 > 3:
        tags.append("high_volatility")

    return list(dict.fromkeys(tags))    # no duplicates, order preserved


# Russian phrasings for SEARCH. The book is in Russian while the tags are English;
# the query "oversold" against a Russian corpus returned the same chunk about
# stop-losses for completely different market states (measured 2026-08-16).
# The phrasings below are search queries in the corpus language and stay in Russian.
TAG_QUERIES = {
    "overbought": "перекупленность рынок вырос слишком сильно откат вниз",
    "oversold": "перепроданность рынок упал слишком сильно отскок вверх",
    "elevated_rsi": "осциллятор высокие значения ослабление роста",
    "depressed_rsi": "осциллятор низкие значения ослабление падения",
    "neutral_rsi": "отсутствие выраженного сигнала осциллятора",
    "above_vwap": "цена выше средней возврат к средней продажа на росте",
    "below_vwap": "цена ниже средней возврат к средней покупка на падении",
    "at_vwap": "цена у средней отсутствие отклонения",
    "macd_bullish": "MACD пересечение вверх сигнал к покупке",
    "macd_bearish": "MACD пересечение вниз сигнал к продаже",
    "near_bb_upper": "верхняя граница полосы предел диапазона цен",
    "near_bb_lower": "нижняя граница полосы предел диапазона цен",
    "uptrend": "восходящий тренд растущий рынок продолжение роста",
    "downtrend": "нисходящий тренд падающий рынок продолжение падения",
    "capitulation_possible": "паническая распродажа капитуляция кульминация падения",
    "exhaustion_possible": "истощение тенденции угасание роста разворот",
    "high_volume": "высокий объём подтверждение движения",
    "low_volume": "низкий объём слабое участие ложное движение",
    "ultra_low_volume": "очень низкий объём отсутствие интереса",
    "high_volatility": "высокая волатильность широкий диапазон риск",
    "orderflow_bullish": "давление покупателей агрессивные покупки",
    "orderflow_bearish": "давление продавцов агрессивные продажи",
    "delta_divergence": "расхождение дивергенция цены и индикатора",
    "absorption": "поглощение крупный участник встречает поток",
    "large_player_active": "крупный участник большие объёмы",
    "stops_above_cluster": "скопление защитных остановок выше цены",
    "stops_below_cluster": "скопление защитных остановок ниже цены",
}

# English phrasings for the ENGLISH collection (the book3 variant).
# Needed for the same reason Russian ones are needed for the Russian one: a query in
# a foreign language returned the same chunk for different market states.
# This is the very bug that was fixed in p7 — it must not be repeated in mirror form.
TAG_QUERIES_EN = {
    "overbought": "overbought market rose too far pullback down",
    "oversold": "oversold market fell too far bounce up",
    "elevated_rsi": "oscillator high readings momentum weakening",
    "depressed_rsi": "oscillator low readings selling exhausted",
    "neutral_rsi": "no clear oscillator signal",
    "above_vwap": "price above the mean reversion selling into strength",
    "below_vwap": "price below the mean reversion buying weakness",
    "at_vwap": "price at the mean no deviation",
    "macd_bullish": "MACD crossover upward buy signal",
    "macd_bearish": "MACD crossover downward sell signal",
    "near_bb_upper": "upper band limit of price range volatility",
    "near_bb_lower": "lower band limit of price range volatility",
    "uptrend": "uptrend rising market momentum continuation",
    "downtrend": "downtrend falling market momentum continuation",
    "capitulation_possible": "panic selling capitulation climax",
    "exhaustion_possible": "trend exhaustion fading rally reversal",
    "high_volume": "high volume confirms the move",
    "low_volume": "low volume weak participation false move",
    "ultra_low_volume": "very low volume no interest",
    "high_volatility": "high volatility wide range risk",
    "orderflow_bullish": "buying pressure aggressive buyers",
    "orderflow_bearish": "selling pressure aggressive sellers",
    "delta_divergence": "divergence between price and indicator",
    "absorption": "large participant absorbing the order flow",
    "large_player_active": "large participant size institutional",
    "stops_above_cluster": "cluster of stop orders above price",
    "stops_below_cluster": "cluster of stop orders below price",
}


# What in a chunk's METADATA says it fits a tag. The metadata is in
# English and structured, so it matches reliably — unlike
# the earlier attempt to look for an English tag inside Russian prose.
TAG_META = {
    "overbought": {"kw": {"oscillator", "extremum", "turn"}, "action": "sell"},
    "oversold": {"kw": {"oscillator", "extremum", "turn"}, "action": "buy"},
    "elevated_rsi": {"kw": {"oscillator", "rsi"}},
    "depressed_rsi": {"kw": {"oscillator", "rsi"}},
    "above_vwap": {"kw": {"moving_average", "extremum"}, "action": "sell"},
    "below_vwap": {"kw": {"moving_average", "extremum"}, "action": "buy"},
    "macd_bullish": {"kw": {"macd", "oscillator"}},
    "macd_bearish": {"kw": {"macd", "oscillator"}},
    "near_bb_upper": {"kw": {"volatility", "support_resistance", "extremum"}},
    "near_bb_lower": {"kw": {"volatility", "support_resistance", "extremum"}},
    "uptrend": {"kw": {"trend", "trending_market", "direction"}, "phase": "uptrend"},
    "downtrend": {"kw": {"trend", "trending_market", "direction"}, "phase": "downtrend"},
    "capitulation_possible": {"kw": {"spike", "extremum", "turn"}},
    "exhaustion_possible": {"kw": {"turn", "extremum"}},
    "high_volume": {"kw": {"volume"}},
    "low_volume": {"kw": {"volume", "false_signal"}},
    "ultra_low_volume": {"kw": {"volume", "false_signal"}},
    "high_volatility": {"kw": {"volatility", "risk"}},
    "orderflow_bullish": {"kw": {"volume", "tick", "direction"}, "action": "buy"},
    "orderflow_bearish": {"kw": {"volume", "tick", "direction"}, "action": "sell"},
    "delta_divergence": {"kw": {"divergence", "false_signal"}},
    "absorption": {"kw": {"absorption", "volume"}},
    "large_player_active": {"kw": {"volume", "tick"}},
    "stops_above_cluster": {"kw": {"stops", "risk"}},
    "stops_below_cluster": {"kw": {"stops", "risk"}},
}


def _score_chunk(meta, tags):
    """How well a chunk fits the market state — BY METADATA.

    The previous version looked for an English tag ("below_vwap", "oversold") inside
    the Russian text of the book. That could never match, apart from Latin
    abbreviations like MACD and RSI, so the score was effectively made up of
    bonuses for length — i.e. retrieval ranked chunks BY SIZE. The
    2026-08-16 measurement showed this directly: the chunk about RSI scored 1.0, all of it
    for text length.
    """
    meta = meta or {}
    kws = set(meta.get("keywords") or [])
    phases = set(meta.get("market_phase") or [])
    action = (meta.get("action") or "").strip()
    score = 0.0
    for tag in tags:
        w = TAG_WEIGHTS.get(tag, 1)
        spec = TAG_META.get(tag)
        if not spec:
            continue
        hits = kws & spec.get("kw", set())
        score += w * len(hits)
        if spec.get("phase") and spec["phase"] in phases:
            score += w
        if spec.get("action") and spec["action"] == action:
            score += w * 0.5
    # The chunk quality score from the base's labelling (3..9): other things equal,
    # prefer phrasings labelled as more substantive.
    score += (meta.get("quality_score") or 3) * 0.3
    return score


def query_by_tags(tags, n_results=None, collection=SCHWAGER, lang="ru",
                   header=None):
    """Returns a ready text block with principles from the book for the tags.

    collection — which book to ask (each has its own, they must not be mixed).
    lang — the language of the query; MUST match the corpus language,
    otherwise search degenerates (see TAG_QUERIES_EN).
    """
    col = _get_collection(collection)
    if col is None or not tags:
        return ""
    n_results = n_results or RAG_N_RESULTS
    found, seen = [], set()
    try:
        for tag in tags[:10]:      # cap: each tag is a separate query with an embedding
            # Search with the RUSSIAN phrasing: the corpus is Russian, and the English tag
            # returned the same chunk about stop-losses for different states.
            qmap = TAG_QUERIES_EN if lang == "en" else TAG_QUERIES
            query = qmap.get(tag, tag.replace("_", " "))
            res = col.query(query_texts=[query], n_results=3)
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            for doc, m in zip(docs, metas):
                if doc in seen or (m or {}).get("category") == "garbage":
                    continue
                seen.add(doc)
                found.append((doc, m or {}, _score_chunk(m, tags)))
    except Exception as e:
        print(f"   ⚠️ Book search failed ({e}) — working without knowledge")
        return ""

    if not found:
        return ""
    found.sort(key=lambda x: x[2], reverse=True)
    # A per-category limit. In the base RISK takes 151 chunks of 389 (39%),
    # and with weak ranking the selection degenerated into five pieces of advice about
    # stop-losses — which pushed the decider towards caution regardless of the market.
    # Diversity matters more here than one more chunk of the same category.
    top, per_cat = [], {}
    for doc, m, sc in found:
        cat = m.get("category", "?")
        if per_cat.get(cat, 0) >= 2:
            continue
        per_cat[cat] = per_cat.get(cat, 0) + 1
        top.append((doc, m, sc))
        if len(top) >= n_results:
            break
    lines = [header or "ПРИНЦИПЫ ИЗ КНИГИ (Швагер, «Технический анализ»), подобранные под текущее состояние:"]
    for doc, m, _ in top:
        concept = (m.get("concept") or "").strip()
        head = f" ({concept})" if concept else ""
        lines.append(f"• [{m.get('category', '?')}]{head} {doc.strip()}")
    return "\n".join(lines)
