# ============================================================
# JSON SCHEMAS — structured output (format=<schema>, NOT the string "json"). Field descriptions stay in Russian on purpose: they are part of the model's input.
# ============================================================

# The model's decision for the EXPERIMENT (utils/experiment.py). The only place
# in the whole system where the LLM DECIDES something rather than describes — and even then in a separate
# store, without affecting live signals.
#
# key_reason is mandatory on purpose: without an explanation one cannot tell a meaningful
# decision from guessing, and that is exactly what the experiment tests.
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "direction": {"enum": ["LONG", "SHORT", "NO_TRADE"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "key_reason": {"type": "string", "description": "Главная причина решения, 1-2 предложения"},
    },
    "required": ["direction", "confidence", "key_reason"],
}

# Gemma fills ONLY the narrative fields — direction, entry/stop/
# target levels and position size are computed by Python (utils/signals.py). See agents/trader.py.
#
# The schema is merged: previously there was a separate Junior call (market description) BEFORE
# the decision was computed and a separate Trader call (commentary on the decision) AFTER.
# The Junior output did not take part in scoring at all — it merely retold the same
# numbers and went into the trader prompt, i.e. the model was talking to itself
# at the cost of an extra ~20 sec per coin. Merged into one call AFTER the computation: this way
# the model has strictly more context (it sees both the market and the decision made),
# and there are two calls per coin instead of three.
TRADER_SCHEMA = {
    "type": "object",
    "properties": {
        "executive_summary": {"type": "string", "description": "Что происходит с монетой (3-5 предложений)"},
        "opportunities": {"type": "array", "items": {"type": "string"}, "description": "Возможности (2-3)"},
        "historical_analogy": {"type": "string", "description": "Похожая ситуация из истории крипторынка"},
        "risks": {"type": "array", "items": {"type": "string"}, "description": "Ключевые риски (2-4)"},
        "commentary": {"type": "string", "description": "Комментарий к решению системы"},
    },
    "required": ["executive_summary", "opportunities", "historical_analogy", "risks", "commentary"],
}
