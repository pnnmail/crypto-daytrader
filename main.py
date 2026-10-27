"""
Crypto agent — an independent day-trading bot. Entry point.

build_snapshot() — the numeric context for a coin (indicators/levels/volume).
build_trade_report() builds the full pipeline on top of it
Order Flow -> Trader and returns the finished text + the trade plan for the history.
"""
import logging

from config import (
    TELEGRAM_BOT_TOKEN, MY_CHAT_ID, RESOLVE_INTERVAL_MIN, MAX_LINE_LOG,
    SPREAD_ROUND_TRIP_PCT, REALISTIC_SPREAD_MARKET_PCT, REALISTIC_SPREAD_LIMIT_PCT,
    COINBASE_MAKER_ROUND_TRIP_PCT, BEST_ACHIEVABLE_ROUND_TRIP_PCT,
    DECIDER_PRIMARY,
)
from data.fetcher import fetch_candles, fetch_trades, fetch_funding_rate, check_data_quality
from data.indicators import (
    compute_indicators, compute_daily_levels, compute_volume_profile,
    detect_liquidity_levels, compute_volume_ratio, compute_orderflow_metrics,
    compute_moving_averages,
)
from data.stumpy_finder import find_similar_patterns, format_stumpy_result
from data.correlation import compute_btc_correlation, format_correlation_text
from agents.trader import run_trader
from agents.decider import run_deciders
from utils.signal_history import append_signal
from utils.helpers import safe_format_float, escape_markdown

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("crypto_agent")


class _TelegramHttpNoise(logging.Filter):
    """
    Cleans up httpx logs (used by python-telegram-bot):

    1. getUpdates fires every ~10 seconds forever and carries no information —
       in the log it crowds out the pipeline's real events. Dropped entirely.
       Other calls (sendMessage, getMe, setMyCommands) are rare and useful
       for diagnostics — they are kept.
    2. httpx prints the bot token right in the URL, i.e. EVERY log line
       contains a working secret. Logs regularly get copied out (into a chat,
       an issue, a conversation), and the token lets anyone hijack the bot entirely. It is replaced
       with <TOKEN> before output.
    """

    def filter(self, record):
        msg = record.getMessage()
        if "getUpdates" in msg:
            return False
        if TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN in msg:
            record.msg = msg.replace(TELEGRAM_BOT_TOKEN, "<TOKEN>")
            record.args = ()
        return True


logging.getLogger("httpx").addFilter(_TelegramHttpNoise())


def fetch_market_data(coin: str):
    """Hourly candles (for indicators/levels), daily ones (for volume_ratio/MA),
    recent trades (for order flow)."""
    df_hourly = fetch_candles(coin, "1h", days_back=10)
    df_daily = fetch_candles(coin, "1d", days_back=70)
    trades = fetch_trades(coin, limit=1000)
    return df_hourly, df_daily, trades


def build_snapshot(coin: str) -> dict:
    """Collects the whole numeric context for a coin. Python computes everything —
    none of it is handed to the LLM to recompute."""
    df_hourly, df_daily, trades = fetch_market_data(coin)

    ok, quality_warning = check_data_quality(df_hourly)

    indicators = compute_indicators(df_hourly)
    daily_levels = compute_daily_levels(df_hourly)
    volume_profile = compute_volume_profile(df_hourly)
    liquidity = detect_liquidity_levels(df_hourly)
    volume_ratio = compute_volume_ratio(df_daily)
    moving_averages = compute_moving_averages(df_daily)
    orderflow = compute_orderflow_metrics(trades, df_hourly, indicators, daily_levels)

    return {
        "coin": coin,
        "data_ok": ok,
        "data_warning": quality_warning,
        "indicators": indicators,
        "daily_levels": daily_levels,
        "volume_profile": volume_profile,
        "liquidity": liquidity,
        "volume_ratio": volume_ratio,
        "moving_averages": moving_averages,
        "orderflow": orderflow,
        "df_hourly": df_hourly,
        "df_daily": df_daily,
        "trades": trades,
    }


def format_snapshot_report(snapshot: dict) -> str:
    coin = snapshot["coin"]
    ind = snapshot["indicators"]
    lv = snapshot["daily_levels"]
    vp = snapshot["volume_profile"]
    liq = snapshot["liquidity"]
    vr = snapshot["volume_ratio"]
    of = snapshot["orderflow"]

    lines = [f"*{coin}-USD* — market snapshot (day boundary: {lv.get('day_boundary', 'UTC 00:00')})"]
    if not snapshot["data_ok"]:
        lines.append(f"⚠️ {snapshot['data_warning']}")

    lines.append(
        f"\n💰 Price: `{safe_format_float(ind.get('price'), 6)}` "
        f"({ind.get('price_change', 0):+.2f}%)"
    )
    lines.append(
        f"📊 RSI: {ind.get('rsi')} | MACD: {safe_format_float(ind.get('macd'), 4)} / "
        f"{safe_format_float(ind.get('macd_signal'), 4)}"
    )
    lines.append(
        f"📉 Bollinger: {safe_format_float(ind.get('bb_lower'), 6)} — "
        f"{safe_format_float(ind.get('bb_upper'), 6)} (position {ind.get('bb_position_pct')}%)"
    )
    lines.append(
        f"\n🎯 Pivot: {safe_format_float(lv.get('pivot'), 6)} | R1: {safe_format_float(lv.get('r1'), 6)} | "
        f"S1: {safe_format_float(lv.get('s1'), 6)}"
    )
    if not lv.get('pivot_reliable', True):
        lines.append(f"⚠️ Gap from Pivot {lv.get('pivot_gap_pct')}% — R/S levels unreliable")
    lines.append(
        f"📍 VWAP: {safe_format_float(lv.get('vwap'), 6)} (price {lv.get('vwap_position')}, "
        f"{lv.get('distance_to_vwap', 0):+.2f}%)"
    )
    lines.append(f"📈 Trend 5h: {lv.get('trend_5h')} | 15h: {lv.get('trend_15h')}")
    if lv.get('trend_divergence'):
        lines.append("⚠️ 5h/15h trends diverge")

    lines.append(f"\n📦 Volume (day, {vr.get('basis', '?')}): {vr.get('volume_ratio', '?')}x")
    lines.append(f"🎯 POC: {safe_format_float(vp.get('poc'), 6)}")

    lines.append(
        f"\n🔀 Order Flow: taker imbalance {of.get('taker_imbalance', 0):+.2f} "
        f"({'BUY' if of.get('taker_imbalance', 0) > 0 else 'SELL' if of.get('taker_imbalance', 0) < 0 else 'NEUTRAL'})"
    )
    if of.get('large_player_active'):
        lines.append("⚡ Large flow imbalance")
    if of.get('delta_divergence'):
        lines.append("⚠️ Flow/price divergence")
    if of.get('absorption_detected'):
        lines.append("🧲 Signs of absorption (high volume, small range)")

    return "\n".join(lines)


def build_trade_report(coin: str):
    """Full pipeline: market snapshot -> STUMPY -> correlation with BTC -> funding
    -> Trader (Python decision + LLM description) -> history -> text.
    Returns (report_text, trader_plan).

    EXACTLY ONE LLM call per coin, and a purely descriptive one: the decision is fully
    deterministic. There used to be three (Junior + Order Flow + Trader) — Junior just
    retold the same numbers and was not in the score, while Order Flow turned
    a number already computed by Python into a category, and with a systematic
    skew at that (see ORDERFLOW_BIAS_THRESHOLD in data/indicators.py)."""
    snapshot = build_snapshot(coin)

    stumpy_result = find_similar_patterns(coin, df_hourly=snapshot["df_hourly"])
    # v6: NOT passed into the PROMPTS. The factor was measured as noise (r=0.00 at our
    # horizon), and text given to the models like "in similar situations historically
    # there was such-and-such percentage of rises" looks like statistics and reads as
    # an argument. We have twice already got a divergence between what the formula sees
    # and what the model sees — this would have been the third. The figures still go
    # into the report: they cost 0.02 s to compute and are useful as reference.
    stumpy_text = ""

    correlation_result = compute_btc_correlation(coin, df_coin_daily=snapshot["df_daily"])
    correlation_text = format_correlation_text(correlation_result)

    funding_data = fetch_funding_rate(coin)
    funding_text = format_funding_text(funding_data)

    # Order Flow is computed in build_snapshot() deterministically (Python),
    # there is no LLM call here any more — see ORDERFLOW_BIAS_THRESHOLD in
    # data/indicators.py on why the model was removed from here.
    orderflow = snapshot["orderflow"]

    trader_plan = run_trader(
        coin, snapshot["indicators"], snapshot["daily_levels"], snapshot["volume_profile"],
        snapshot["liquidity"], orderflow=orderflow, stumpy_text=stumpy_text,
        correlation_text=correlation_text, funding_text=funding_text,
        volume_ratio=snapshot["volume_ratio"], stumpy_result=stumpy_result, funding_data=funding_data,
    )

    append_signal(coin, trader_plan)

    llm_decisions = decide_and_record(coin, snapshot, trader_plan, stumpy_text,
                                       correlation_text, funding_text)

    report_text = format_trade_report(snapshot, orderflow, trader_plan, stumpy_result,
                                       correlation_result, funding_data, llm_decisions)
    return report_text, trader_plan


def decide_and_record(coin, snapshot, trader_plan, stumpy_text, correlation_text, funding_text):
    """
    Asks the model about direction and returns its decision FOR DISPLAY in the report.
    The experiment point itself is recorded only on schedule.

    The split is intentional. The model's decision must be shown in EVERY report —
    otherwise it is simply not visible next to the calculation's decision. But RECORDING into
    the experiment must stay once per EXPERIMENT_INTERVAL_HOURS, otherwise
    observations start overlapping and significance gets inflated (in the stop-loss
    backtest the overlap inflated t several times).

    Cost of the decision: ~5 s per coin on top of the descriptive call. A failure here must not
    take down the live report — the experiment is secondary, so everything is in a try.
    """
    from utils.experiment import should_record, append_point, is_enabled, build_context
    from utils.signals import calculate_entry_levels

    if not is_enabled():
        return None
    try:
        price = snapshot["indicators"]["price"]
        # Levels for BOTH sides are computed by Python in any case: we compare
        # the choice of direction, not the ability to place stops. Otherwise the model
        # would lose/win because of different trade geometry.
        lv_long = calculate_entry_levels("LONG", price, snapshot["daily_levels"],
                                          snapshot["volume_profile"], snapshot["liquidity"])
        lv_short = calculate_entry_levels("SHORT", price, snapshot["daily_levels"],
                                           snapshot["volume_profile"], snapshot["liquidity"])

        llm_decisions = run_deciders(
            coin, snapshot["indicators"], snapshot["daily_levels"], snapshot["volume_profile"],
            snapshot["liquidity"], snapshot["orderflow"], snapshot["volume_ratio"],
            stumpy_text=stumpy_text, correlation_text=correlation_text, funding_text=funding_text,
        )
        # Recording into the experiment — only on schedule, display — always.
        if should_record(coin):
            python_decision = {
                "direction": trader_plan.get("direction"),
                "score": trader_plan.get("direction_score"),
                "scores": trader_plan.get("scores"),
            }
            keep = ("stop_loss", "target_1", "target_2")
            # Donchian is computed on DAILY candles — a separate request, but it is
            # cheap and model-free. A failure here must not break recording the point,
            # so the result goes as is: {"_error": True} excludes
            # the reference line from the denominator rather than replacing it with NO_TRADE.
            from utils.donchian import donchian_state
            don = donchian_state(coin)
            append_point(coin, price,
                         {k: lv_long.get(k) for k in keep},
                         {k: lv_short.get(k) for k in keep},
                         python_decision, llm_decisions, donchian=don,
                         context=build_context(snapshot["indicators"], snapshot["daily_levels"],
                                               snapshot["orderflow"], snapshot["volume_ratio"],
                                               coin=coin, df_hourly=snapshot.get("df_hourly")))
            summary = ", ".join(f"{v}={(d or {}).get('direction', 'error')}"
                                for v, d in llm_decisions.items())
            log.info(f"experiment: point recorded {coin} — calculation "
                     f"{python_decision['direction']}, model: {summary}")
        return llm_decisions
    except Exception as e:
        log.warning(f"experiment: model decision not received for {coin}: {e}")
        return None


def format_cost_warning(trader_plan: dict) -> str:
    """
    A warning about REAL costs — across three venues, not one.

    The decision is computed with SPREAD_ROUND_TRIP_PCT=0 (research mode — otherwise
    the entry filter rejected almost everything and there was nothing to measure), so the
    R:R figures above are POTENTIAL BEFORE COSTS.

    Before the 2026-08-16 measurement only Robinhood was shown here, and the line overstated
    the cost about 20x for someone trading on an exchange: at
    Binance.US the fee is 0%/0.01%, and the whole cost is the spread, which on the liquid
    top is 0.005-0.03%. Showing only the Robinhood price meant
    declaring hopeless what pays off on another venue.
    """
    if SPREAD_ROUND_TRIP_PCT > 0:
        return f"_R:R shown after deducting a {SPREAD_ROUND_TRIP_PCT}% spread (round trip)_"

    entry = trader_plan.get("entry_price")
    t1 = trader_plan.get("target_1")
    if not entry or not t1:
        return ""

    gross_pct = abs(t1 - entry) / entry * 100
    venues = [
        ("Binance.US, liquid top", BEST_ACHIEVABLE_ROUND_TRIP_PCT),
        ("Coinbase Advanced, maker", COINBASE_MAKER_ROUND_TRIP_PCT),
        ("Robinhood, market order", REALISTIC_SPREAD_MARKET_PCT),
    ]
    lines = [f"\n⚠️ *Costs are not included in the decision* (research mode, spread=0).",
             f"   Move to target 1: {gross_pct:.2f}%. After a round trip:"]
    for name, cost in venues:
        net = gross_pct - cost
        mark = "✅" if net > 0 else "❌"
        lines.append(f"   {mark} {name} ({cost}%): *{net:+.2f}%*")
    best = gross_pct - BEST_ACHIEVABLE_ROUND_TRIP_PCT
    lines.append("   -> " + ("does not pay off even on the cheapest venue"
                             if best <= 0 else
                             "pays off only on the cheap venue"
                             if gross_pct - COINBASE_MAKER_ROUND_TRIP_PCT <= 0 else
                             "pays off everywhere except Robinhood"
                             if gross_pct - REALISTIC_SPREAD_MARKET_PCT <= 0 else
                             "pays off on any venue"))
    return "\n".join(lines)


def format_funding_text(funding_data: dict) -> str:
    if not funding_data or funding_data.get("error"):
        return f"💸 *Funding rate:* no data ({(funding_data or {}).get('error', '?')})"
    return f"💸 *Funding rate:* {funding_data['funding_rate_pct']:+.4f}% (Kraken {funding_data['symbol']})"


# Label of each variant: WHAT it sees and IN WHICH language it works.
# The order sets the output order and is built in pairs that are compared:
# plain/plain_en — pure language; book2/book3 — language with the same content;
# book/book2/book4 — different books in the same language.
VARIANT_LABEL = {
    "plain":    ("numbers only", "RU", "the reference point"),
    "plain_en": ("numbers only", "EN", "language control for plain"),
    "book":     ("+ Schwager", "RU", "classic technical analysis"),
    "regime":   ("+ Schwager and briefing", "RU", "our measured fact about the regime"),
    "book2":    ("+ Chan", "RU", "a quantitative approach"),
    "book3":    ("+ Chan", "EN", "the same rules, the whole call in English"),
    "book4":    ("+ Aziz", "RU", "intraday trading around VWAP"),
}
VARIANT_ORDER = ("plain", "plain_en", "book", "regime", "book2", "book3", "book4")


def format_collected_data(coin: str, df_hourly=None) -> str:
    """Showing the collected features is COLLECTION, not decision.

    Features are computed AFTER both deciders have already answered (the order in
    decide_and_record: first run_deciders, then build_context as an argument of
    the record), so they physically cannot influence the decision. The label saying so
    is mandatory: without it the block reads as another set of arguments, and
    the next reader of the report will decide the system takes them into account.

    Errors are swallowed: this is a display, the coin card must not fail because of it.
    """
    try:
        from utils.flow_features import build_flow_features
        from utils.market_features import build_market_features
        mf = build_market_features(coin, df_hourly=df_hourly) or {}
        ff = build_flow_features(coin) or {}
    except Exception:
        return ""
    if not mf and not ff:
        return ""

    out = ["\n━━━ 📊 *DATA (collected, NOT part of the decision)* ━━━"]
    w = ff.get("flow_15m") or {}
    if w.get("n_trades"):
        dp = w.get("delta_pct")
        line = f"   flow 15m: trades {w['n_trades']}"
        if dp is not None:
            line = f"   flow 15m: delta {dp:+.1%} · trades {w['n_trades']}"
        # An incomplete window is marked right in the report: a silently shown
        # 15-minute feature computed from three minutes misleads
        # exactly as much as one silently counted in the statistics.
        line += f" · window {'complete' if w.get('complete') else 'INCOMPLETE'}"
        out.append(line)
    b = ff.get("book") or {}
    if b.get("spread_pct") is not None:
        imb = b.get("book_imbalance_1_0")
        out.append(f"   book: spread {b['spread_pct']:.3f}%"
                   + (f" · imbalance ±1% {imb:+.2f}" if imb is not None else ""))
    if mf.get("btc_ret_4h") is not None:
        out.append(f"   regime: BTC 4h {mf['btc_ret_4h']:+.2f}% {mf.get('btc_regime') or ''}"
                   + (f" · {coin} vs BTC {mf['vs_btc_4h']:+.2f}%"
                      if mf.get("vs_btc_4h") is not None else ""))
    if mf.get("rvol20") is not None:
        out.append(f"   volume: RVOL20 {mf['rvol20']:.2f}"
                   + (f" · percentile {mf['volume_percentile']:.0f}"
                      if mf.get("volume_percentile") is not None else ""))
    return "\n".join(out) if len(out) > 1 else ""


def format_llm_variants(llm_decisions: dict, python_direction: str) -> str:
    """All decider variants as a table: shows what each hint gives.

    The point of showing all of them is the experiment itself: the book and the briefing are tested
    on THE SAME point, and the difference between rows is the measured effect.
    """
    if not llm_decisions:
        return "_model decisions were not requested (experiment is off)_"

    order = [v for v in VARIANT_ORDER if v in llm_decisions]
    order += [v for v in llm_decisions if v not in order]

    lines = []
    for variant in order:
        d = llm_decisions.get(variant)
        name, lang, note = VARIANT_LABEL.get(variant, (variant, "?", ""))
        head = f"*{escape_markdown(name)}* `[{lang}]`"
        if not d or d.get("_error"):
            lines.append(f"⚠️ {head} — model error, not counted in the statistics")
            continue
        final = str(d.get("direction", "?"))
        conf = d.get("confidence")
        mark = "⚖️" if final != python_direction else "✓"
        conf_txt = f" · conf. {conf:.0%}" if conf is not None else ""
        lines.append(f"{mark} {head} → *{escape_markdown(final)}*{conf_txt}")
        lines.append(f"   _{escape_markdown(note)}_")
        if d.get("downgraded_from"):
            lines.append(f"   _wanted {escape_markdown(str(d['downgraded_from']))}, "
                         f"not confident enough_")
        reason = (d.get("key_reason") or "").strip()
        if reason:
            lines.append(f"   {escape_markdown(reason[:170])}")

    # Summary: without it, with seven rows, the main thing is not visible — whether
    # the variants diverged at all. If they all said the same, the experiment has nothing to measure
    # at this point, and that matters more than any single row.
    dirs = [str((llm_decisions.get(v) or {}).get("direction"))
            for v in order if (llm_decisions.get(v) or {}).get("direction")]
    if dirs:
        uniq = sorted(set(dirs))
        if len(uniq) == 1:
            lines.append(f"\n_all {len(dirs)} variants agreed on "
                         f"{escape_markdown(uniq[0])} — the hints changed nothing_")
        else:
            parts = ", ".join(f"{escape_markdown(u)} {dirs.count(u)}" for u in uniq)
            lines.append(f"\n_divergence: {parts}_")
    return "\n".join(lines)


def format_trade_report(snapshot: dict, orderflow: dict,
                         trader_plan: dict, stumpy_result: dict,
                         correlation_result: dict = None, funding_data: dict = None,
                         llm_decisions: dict = None) -> str:
    coin = snapshot["coin"]
    gemma = trader_plan.get("gemma", {}) or {}

    # ══════════════════════════════════════════════════════════════════
    # The report is built in THREE BLOCKS BY SOURCE, not by topic: first the shared
    # data, then everything the calculation produced, then everything the model produced.
    # Previously the model's narrative was split (summary at the top, risks and commentary
    # at the bottom), with the breakdown of the calculation's scores in between — it was impossible
    # to tell whose judgement it was. For the "calculation vs model" experiment this is
    # critical: their conclusions must not be mixed in one stream of text.
    # ══════════════════════════════════════════════════════════════════
    lines = [f"*{coin}-USD* — trade plan (day boundary: "
             f"{snapshot['daily_levels'].get('day_boundary', 'UTC 00:00')})"]
    if not snapshot["data_ok"]:
        lines.append(f"⚠️ {snapshot['data_warning']}")

    # ───────────────── BLOCK 1: DATA (shared by both) ─────────────────
    lines.append("\n━━━ 📊 *DATA* ━━━")
    lines.append(f"💰 Price: `{safe_format_float(snapshot['indicators'].get('price'), 6)}` "
                 f"({snapshot['indicators'].get('price_change', 0):+.2f}%)")

    # Order Flow is computed deterministically, a model error can no longer
    # happen here — so there is no _error branch either.
    of_flags = []
    if orderflow.get("large_player_active"):
        of_flags.append("large imbalance")
    if orderflow.get("delta_divergence"):
        of_flags.append("divergence")
    if orderflow.get("absorption_detected"):
        of_flags.append("absorption")
    of_suffix = f" | {', '.join(of_flags)}" if of_flags else ""
    lines.append(f"🔀 Order Flow: {orderflow.get('orderflow_bias', '?')} "
                 f"(imbalance {orderflow.get('taker_imbalance', 0):+.2f}){of_suffix}")

    if stumpy_result:
        n3 = stumpy_result.get("n_3d", 0)
        wr3 = stumpy_result.get("win_rate_3d")
        base = stumpy_result.get("base_win_rate")
        ci = f"[{stumpy_result.get('ci_lower_3d')}-{stumpy_result.get('ci_upper_3d')}]" if wr3 is not None else "?"
        base_txt = f" vs base {base}%" if base is not None else ""
        lines.append(f"📈 STUMPY (3d): n={n3}, win rate {wr3}%{base_txt}, CI {ci}")
        lines.append("   _reference only, not part of the decision: measured as noise (r=0.00)_")

    corr_text = format_correlation_text(correlation_result) if correlation_result else ""
    if corr_text:
        lines.append(corr_text)

    if funding_data is not None:
        lines.append(format_funding_text(funding_data))

    # ───────────── BLOCK 2: THE CALCULATION'S DECISION (deterministic) ─────────────
    direction = trader_plan.get("direction")
    lines.append(f"\n━━━ 🐍 *CALCULATION* ━━━")
    # _md — because "NO_TRADE" contains an underscore, and in legacy Markdown
    # Telegram treats it as italics: the line "*CALCULATION: NO_TRADE*" broke the markup
    # and "NOTRADE" with misplaced asterisks went to the chat.
    lines.append(f"Decision: *{escape_markdown(str(direction))}* "
                 f"(score {trader_plan.get('direction_score', 0):+d})")

    if direction == "NO_TRADE":
        lines.append(trader_plan.get("no_trade_report", ""))
    else:
        lines.append(f"Entry: `{safe_format_float(trader_plan.get('entry_price'), 6)}` | "
                     f"Stop: `{safe_format_float(trader_plan.get('stop_loss'), 6)}`")
        lines.append(f"Target 1: `{safe_format_float(trader_plan.get('target_1'), 6)}` "
                     f"(R:R 1:{trader_plan.get('risk_reward_1', '?')}) | "
                     f"Target 2: `{safe_format_float(trader_plan.get('target_2'), 6)}` "
                     f"(R:R 1:{trader_plan.get('risk_reward_2', '?')})")
        lines.append(f"Position size: {trader_plan.get('position_size_pct', 0)}% | "
                     f"Confidence: {trader_plan.get('confidence', 0):.0%}")
        lines.append(format_cost_warning(trader_plan))

    # ───── BLOCK 3: THE MODEL'S OWN DECISION (experiment) ─────
    # Separated from block 4 on purpose. These are TWO DIFFERENT calls of one model in
    # different roles, and they regularly contradict each other:
    #   here  — it decided BY ITSELF, without seeing the calculation's answer (otherwise it would copy it);
    #   below — it saw the calculation's answer and must comment on it.
    # Without explicit labels the report reads as a contradiction inside the model:
    # "Decision: SHORT" and right after "the system's NO_TRADE decision is justified".
    lines.append(f"\n━━━ 🤖 *MODEL — ITS OWN DECISIONS* ━━━")
    lines.append("_decided independently, the calculation's answer was not shown to it_")
    lines.append(format_llm_variants(llm_decisions, direction))

    # A display of the collected features. It comes AFTER all decisions on purpose:
    # the order in the report repeats the order in the code, and the reader can see that these
    # numbers appear only after the decision has been made.
    collected = format_collected_data(coin, snapshot.get("df_hourly"))
    if collected:
        lines.append(collected)

    primary = (llm_decisions or {}).get(DECIDER_PRIMARY) or {}
    if primary.get("key_reason") and not primary.get("_error"):
        lines.append(f"\n_Rationale ({VARIANT_LABEL.get(DECIDER_PRIMARY, DECIDER_PRIMARY)}):_ "
                     f"{escape_markdown(primary['key_reason'])}")

    # ───── BLOCK 4: THE MODEL'S COMMENTARY ON THE CALCULATION'S DECISION ─────
    lines.append(f"\n━━━ 💬 *MODEL — COMMENTARY ON THE CALCULATION* ━━━")
    lines.append("_saw the calculation's decision and explains it; this is NOT an independent opinion_")

    if gemma.get("_error"):
        lines.append(f"⚠️ error — {escape_markdown(gemma.get('_error_message', ''))}")
    else:
        if gemma.get("executive_summary"):
            lines.append(f"\n🧠 *Analysis:* {escape_markdown(gemma['executive_summary'])}")
        if gemma.get("opportunities"):
            lines.append("\n🔍 *Opportunities:*")
            for opp in gemma["opportunities"]:
                lines.append(f"   • {escape_markdown(opp)}")
        if gemma.get("historical_analogy"):
            lines.append(f"\n📖 *Analogy:* {escape_markdown(gemma['historical_analogy'])}")
        if gemma.get("risks"):
            lines.append("\n⚠️ *Risks:*")
            for risk in gemma["risks"]:
                lines.append(f"   • {escape_markdown(risk)}")
        if gemma.get("commentary"):
            lines.append(f"\n💬 {escape_markdown(gemma['commentary'])}")

    return "\n".join(lines)


async def scheduled_resolve(context):
    """job_queue callback: every RESOLVE_INTERVAL_MIN minutes it goes through the signal
    history and closes everything whose horizon has expired. A periodic pass rather than
    a timer per signal — timers are lost when the container restarts
    (and it restarts on every rebuild), while a periodic pass
    self-heals: after any downtime it simply picks up what has accumulated.
    One summary chat message per pass, and only if something actually
    closed (not "checked, empty" on every tick)."""
    import asyncio
    from utils.signal_history import resolve_all_outcomes
    from utils.experiment import resolve_all_points

    summary = await asyncio.to_thread(resolve_all_outcomes)
    # The experiment's counterfactuals are resolved in the same pass — the data is the same
    # (minute candles), a separate job would only duplicate downloads.
    try:
        exp_updated = await asyncio.to_thread(resolve_all_points)
        if exp_updated:
            log.info(f"experiment: counterfactuals resolved — {exp_updated}")
    except Exception as e:
        log.warning(f"experiment: resolving counterfactuals failed: {e}")

    if summary["closed"] == 0 and summary["unresolved"] == 0:
        return
    if not MY_CHAT_ID:
        return

    lines = []
    if summary["closed"]:
        lines.append(f"🔍 Outcome resolution: closed {summary['closed']} "
                     f"({summary['wins']}W/{summary['losses']}L)")
    if summary["unresolved"]:
        lines.append(f"⚠️ could not resolve: {summary['unresolved']}")
    try:
        await context.bot.send_message(chat_id=MY_CHAT_ID, text="\n".join(lines))
    except Exception as e:
        log.warning(f"Could not send the outcome resolution summary: {e}")


def main():
    if not TELEGRAM_BOT_TOKEN:
        log.warning("CRYPTO_TELEGRAM_BOT_TOKEN is not set — the bot cannot start. "
                    "The code is ready, it needs a token from @BotFather and a chat_id.")
        return

    from telegram import Update
    from telegram.ext import Application, CommandHandler, TypeHandler
    from telegram_bot.handlers import (cmd_now, cmd_scan, cmd_coins, cmd_history,
                                        cmd_autoscan, cmd_experiment, cmd_help)
    from agents.autoscan import scheduled_autoscan, get_status, JOB_NAME

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    async def log_update(update, context):
        """Logs the chat_id of every incoming message (group=-1, i.e. before
        the command handlers and without interfering with them).

        Needed not for debugging but for real: background pushes (autoscan, outcome
        resolution summaries) are sent to MY_CHAT_ID, and if it does not match the chat
        where the user actually talks to the bot, Telegram answers "chat not
        found" — while replies to commands arrive fine (they go as a
        reply in the original chat). Without this line there is no way to find the right chat_id:
        getUpdates from outside conflicts with the bot's own polling.
        """
        msg = getattr(update, "effective_message", None)
        if msg is not None:
            log.info(f"📥 INCOMING | chat_id={msg.chat_id} | type={msg.chat.type} | "
                     f"text={(msg.text or '')[:100]}")

    app.add_handler(TypeHandler(Update, log_update), group=-1)
    app.add_handler(CommandHandler("now", cmd_now))
    app.add_handler(CommandHandler("scan", cmd_scan))
    app.add_handler(CommandHandler("coins", cmd_coins))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("autoscan", cmd_autoscan))
    app.add_handler(CommandHandler("experiment", cmd_experiment))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("start", cmd_help))

    # Descriptions for the Telegram menu (the hint when typing "/" in the client). Without
    # set_my_commands the commands keep working but do not appear in
    # autocomplete — they would have to be typed out in full every time.
    COMMAND_MENU = [
        ("now", "Full analysis of a coin right now: /now BTC"),
        ("scan", "Run the whole coin list (or /scan BTC ETH ...)"),
        ("coins", "Coin list; /coins add|remove COIN"),
        ("autoscan", "Background monitoring: on [minutes] / off / status"),
        ("history", "Signal history and statistics; /history COIN"),
        ("experiment", "Calculation vs model: who decides better"),
        ("help", "Command help"),
    ]

    async def register_menu(application):
        try:
            await application.bot.set_my_commands(COMMAND_MENU)
            log.info(f"Command menu registered ({len(COMMAND_MENU)} items)")
        except Exception as e:
            log.warning(f"Could not register the command menu: {e}")

    app.post_init = register_menu

    # JobQueue does not survive a container restart — if autoscan was on
    # before the restart (see agents/autoscan.py::set_enabled), the job must be brought
    # up again here, otherwise the "enabled=True" state in the file diverges from
    # reality (there is no job, while /autoscan status would lie that it is running).
    status = get_status()
    if status["enabled"]:
        app.job_queue.run_repeating(scheduled_autoscan, interval=status["interval_min"] * 60,
                                     first=15, name=JOB_NAME)
        log.info(f"Autoscan restored after restart: interval {status['interval_min']} min")

    # Outcome resolution always works, without a switch (unlike
    # autoscan) — signal history is written on every /now and /scan, not
    # only with autoscan on, and must be resolved independently.
    app.job_queue.run_repeating(scheduled_resolve, interval=RESOLVE_INTERVAL_MIN * 60,
                                 first=30, name="resolve_outcomes")

    log.info("Crypto agent started, polling...")
    app.run_polling()


if __name__ == "__main__":
    main()
