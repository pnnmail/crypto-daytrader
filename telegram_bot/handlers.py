import asyncio

from config import AUTOSCAN_INTERVAL_MIN, AUTOSCAN_COOLDOWN_MIN, MAX_SIGNALS_PER_COIN_PER_DAY
from utils.coins import load_coins, add_coin, remove_coin

COMMAND_MENU_TEXT = (
    "*Crypto agent* — a research day-trading bot (24/7, Coinbase)\n\n"
    "/now COIN — full analysis right now\n"
    "/scan — go through the whole coin list\n"
    "/coins — list of traded coins\n"
    "/coins add COIN — add a coin\n"
    "/coins remove COIN — remove a coin\n"
    "/autoscan on|off|status — background monitoring\n"
    "/history COIN — signal history for a coin\n"
    "/experiment — calculation vs model: who decides better\n"
    "/help — this help"
)

from telegram_bot.sender import split_message


# Critical t value (two-sided, p<0.05) by degrees of freedom.
# The |t|>2 threshold is right only with a LARGE number of blocks. With two days df=1 and
# the critical value is 12.71 — the report called t=-5.66 "significant", which was
# plainly untrue and nudged towards conclusions from two observations.
_T_CRIT = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36,
           8: 2.31, 9: 2.26, 10: 2.23, 12: 2.18, 15: 2.13, 20: 2.09, 30: 2.04}
# Below this number of blocks we do not talk about significance at all: a t from two or three
# points is so unstable that even a correct critical threshold creates
# a false sense of testing.
MIN_BLOCKS_FOR_SIGNIFICANCE = 5


def _t_critical(days: int) -> float:
    df = max(1, days - 1)
    for k in sorted(_T_CRIT):
        if df <= k:
            return _T_CRIT[k]
    return 1.96


def verdict_for_t(t: float, days: int) -> str:
    """A verbal verdict on t that accounts for the number of blocks."""
    if days < MIN_BLOCKS_FOR_SIGNIFICANCE:
        return (f"only {days} blocks, significance NOT DETERMINED "
                f"(at least {MIN_BLOCKS_FOR_SIGNIFICANCE} different days needed)")
    crit = _t_critical(days)
    if abs(t) > crit:
        return f"significant (threshold {crit:.2f} at {days} blocks)"
    return f"not significant (|t| > {crit:.2f} needed at {days} blocks)"


async def _reply_long(update, text: str):
    """A reply to a command. The splitting is shared with the background push (sender.py) —
    previously there were two different ones, and fixing one did not fix the other: that is exactly
    how autoscan crashed on a long report on 2026-08-21."""
    for chunk in split_message(text):
        try:
            await update.message.reply_text(chunk, parse_mode="Markdown")
        except Exception:
            # Badly escaped markup from LLM text must not break
            # delivery of the report — fall back to plain text for this chunk.
            await update.message.reply_text(chunk)


async def cmd_help(update, context):
    await update.message.reply_text(COMMAND_MENU_TEXT, parse_mode="Markdown")


async def cmd_now(update, context):
    if not context.args:
        await update.message.reply_text("Usage: /now BTC")
        return
    coin = context.args[0].upper()
    coins = load_coins()
    if coin not in coins:
        await update.message.reply_text(
            f"⚠️ {coin} is not in the list ({', '.join(coins)}), but I will try anyway — "
            f"if Coinbase does not trade this pair, there will be an error."
        )

    await update.message.reply_text(f"⏳ Analysing {coin} (candles, STUMPY, correlation, Order Flow, Trader)...")

    from main import build_trade_report
    try:
        report, _ = await asyncio.to_thread(build_trade_report, coin)
    except Exception as e:
        await update.message.reply_text(f"❌ Analysis error for {coin}: {e}")
        return

    await _reply_long(update, report)


async def cmd_scan(update, context):
    coins = context.args if context.args else load_coins()
    coins = [c.upper() for c in coins]
    if len(coins) < 1:
        await update.message.reply_text("The coin list is empty — add via /coins add COIN")
        return

    await update.message.reply_text(f"⏳ Scanning {len(coins)} coins: {', '.join(coins)}...")

    from agents.scanner import scan_coins, build_scan_report
    try:
        results = await asyncio.to_thread(scan_coins, coins)
    except Exception as e:
        await update.message.reply_text(f"❌ Scan error: {e}")
        return

    report = build_scan_report(results)
    await _reply_long(update, report)


async def cmd_coins(update, context):
    args = context.args
    if not args:
        coins = load_coins()
        await update.message.reply_text("Current coin list:\n" + ", ".join(coins))
        return

    action = args[0].lower()
    if action == "add" and len(args) >= 2:
        symbol = args[1].upper()
        coins, added = add_coin(symbol)
        msg = f"✅ {symbol} added." if added else f"{symbol} is already in the list."
        await update.message.reply_text(msg + "\n" + ", ".join(coins))
    elif action in ("remove", "rm") and len(args) >= 2:
        symbol = args[1].upper()
        coins, removed = remove_coin(symbol)
        msg = f"✅ {symbol} removed." if removed else f"{symbol} was not in the list anyway."
        await update.message.reply_text(msg + "\n" + ", ".join(coins))
    else:
        await update.message.reply_text("Usage: /coins, /coins add COIN, /coins remove COIN")


async def cmd_history(update, context):
    from utils.signal_history import load_signal_history, resolve_all_outcomes, compute_performance_stats

    from utils.time_utils import format_local

    coin = context.args[0].upper() if context.args else None
    await asyncio.to_thread(resolve_all_outcomes)

    history = load_signal_history()
    # The ticker is the dict KEY, the record itself does not store it. Without tagging
    # the list across all coins would print as rows without saying which
    # coin each belongs to.
    if coin:
        tagged = [(coin, r) for r in history.get(coin, [])]
    else:
        tagged = [(c, r) for c, recs in history.items() for r in recs]
    if not tagged:
        await update.message.reply_text(f"History is empty{' for ' + coin if coin else ''}.")
        return

    from config import SPREAD_ROUND_TRIP_PCT, REALISTIC_SPREAD_MARKET_PCT, REALISTIC_SPREAD_LIMIT_PCT

    stats = compute_performance_stats(coin)
    lines = [f"*Signal history* {('— ' + coin) if coin else '(all coins)'}"]
    if SPREAD_ROUND_TRIP_PCT > 0:
        lines.append(f"Rule: intraday, 4h horizon, return after a {SPREAD_ROUND_TRIP_PCT}% spread")
    else:
        lines.append("Rule: intraday, 4h horizon, *WITHOUT costs* (research mode)")
    lines.append(f"Total signals: {stats.get('total_signals', 0)}, resolved: {stats.get('resolved', 0)}")
    if stats.get("unresolved"):
        lines.append(f"⚠️ could not resolve: {stats.get('unresolved')} (no exchange data etc.)")
    if stats.get("resolved"):
        lines.append(f"Win rate: {stats.get('win_rate')}% ({stats.get('wins')}W/{stats.get('losses')}L)")
        lines.append(f"Average result (net): {stats.get('avg_return_pct')}% (σ={stats.get('std_return_pct')}%)")
        if stats.get("sharpe_like") is not None:
            lines.append(f"Sharpe-like: {stats.get('sharpe_like')}")
        if stats.get("profit_factor") is not None:
            lines.append(f"Profit factor: {stats.get('profit_factor')}")
        lines.append(f"Best trade: {stats.get('best_trade_coin')} {stats.get('best_trade_pct')}%")
        lines.append(f"Worst trade: {stats.get('worst_trade_coin')} {stats.get('worst_trade_pct')}%")

        # Without this line the statistics read as a trading result, although
        # costs are not included. Real Robinhood fees are from
        # the official fee schedule (see config.py).
        if SPREAD_ROUND_TRIP_PCT == 0:
            avg = stats.get("avg_return_pct") or 0
            lines.append(f"\n⚠️ *Costs not included.* At real Robinhood fees the average "
                         f"trade result would be:")
            lines.append(f"   • market order (1.9% round trip): *{avg - REALISTIC_SPREAD_MARKET_PCT:+.2f}%*")
            lines.append(f"   • limit order via Legend (1.0% round trip): *{avg - REALISTIC_SPREAD_LIMIT_PCT:+.2f}%*")

    recent = sorted(tagged, key=lambda x: x[1].get("timestamp", ""), reverse=True)[:10]
    lines.append("\n*Latest signals:*")
    for c, r in recent:
        outcome = r.get("outcome", "pending")
        net = r.get("return_pct_net")
        net_str = f", {net:+.2f}% net" if net is not None else ""
        when = format_local(r.get("timestamp")) if r.get("timestamp") else "?"
        lines.append(f"   {when} *{c}* {r.get('direction')} "
                     f"(score {r.get('total_score', '?')}) -> {outcome}{net_str}")

    await _reply_long(update, "\n".join(lines))


async def cmd_autoscan(update, context):
    from agents.autoscan import scheduled_autoscan, set_enabled, get_status, JOB_NAME

    args = context.args
    action = args[0].lower() if args else "status"

    if action == "on":
        interval_min = AUTOSCAN_INTERVAL_MIN
        if len(args) >= 2:
            try:
                interval_min = max(1, int(args[1]))
            except ValueError:
                pass

        existing = context.job_queue.get_jobs_by_name(JOB_NAME)
        for job in existing:
            job.schedule_removal()

        context.job_queue.run_repeating(scheduled_autoscan, interval=interval_min * 60,
                                         first=10, name=JOB_NAME)
        set_enabled(True, interval_min)
        coins = load_coins()
        await update.message.reply_text(
            f"✅ Autoscan enabled: every {interval_min} min over {len(coins)} coins "
            f"({', '.join(coins)}).\nCooldown {AUTOSCAN_COOLDOWN_MIN} min/coin, "
            f"limit {MAX_SIGNALS_PER_COIN_PER_DAY} signals/coin per UTC day."
        )
    elif action == "off":
        existing = context.job_queue.get_jobs_by_name(JOB_NAME)
        for job in existing:
            job.schedule_removal()
        set_enabled(False)
        await update.message.reply_text("⏸ Autoscan disabled.")
    else:
        status = get_status()
        jobs = context.job_queue.get_jobs_by_name(JOB_NAME)
        running = "running" if jobs else "not running (rerun /autoscan on after a bot restart)"
        state_text = "enabled" if status["enabled"] else "disabled"
        await update.message.reply_text(
            f"Autoscan: {state_text}, interval {status['interval_min']} min, job {running}."
        )


async def cmd_experiment(update, context):
    """Comparing deciders: the deterministic calculation vs an LLM."""
    from utils.experiment import (compute_experiment_stats, resolve_all_points,
                                  is_enabled, set_enabled as exp_set_enabled)
    from utils.helpers import escape_markdown

    action = context.args[0].lower() if context.args else "status"
    if action in ("on", "off"):
        exp_set_enabled(action == "on")
        await update.message.reply_text(
            f"🧪 Experiment {'enabled' if action == 'on' else 'disabled'}. "
            f"{'An extra LLM call is made at every point.' if action == 'on' else ''}"
        )
        return

    await update.message.reply_text("⏳ Computing counterfactuals...")
    await asyncio.to_thread(resolve_all_points)
    st = await asyncio.to_thread(compute_experiment_stats)

    lines = ["*🧪 Experiment: calculation vs model*"]
    lines.append(f"Collection: {'enabled' if is_enabled() else 'disabled'} | "
                 f"total points: {st.get('total_points', 0)}, resolved: {st.get('resolved', 0)}")

    if not st.get("resolved"):
        lines.append("\nNo resolved points yet — a counterfactual is computed "
                     "4 hours after recording. Check back later.")
        await _reply_long(update, "\n".join(lines))
        return

    # A warning about mixed data: different models/versions must not
    # be added into one statistic.
    models = st.get("models", {})
    versions = st.get("scoring_versions", {})
    if len(models) > 1 or len(versions) > 1:
        lines.append(f"\n⚠️ *Mixed data* — models: {models}, versions: {versions}. "
                     f"The comparison is invalid, a reset is needed.")
    else:
        lines.append(f"Model: `{list(models)[0] if models else '?'}` | "
                     f"scoring: `{list(versions)[0] if versions else '?'}`")

    # The label says WHAT the decider sees and IN WHICH language: without it the output
    # showed "llmbook3" (the markup ate the underscores) and it was impossible to tell how it
    # differs from "llmbook2".
    names = {
        "python":       "🐍 Calculation — formula, no model",
        "donchian":     "📐 Donchian 20/10 — reference line (see caveat)",
        "llm_plain":    "🤖 bare numbers · RU",
        "llm_plain_en": "🤖 bare numbers · EN — language control",
        "llm_book":     "📗 Schwager, classic technical analysis · RU",
        "llm_regime":   "🎯 Schwager + our measured briefing · RU",
        "llm_book2":    "📘 Chan, quantitative approach · RU",
        "llm_book3":    "📘 Chan, THE SAME rules · EN (prompt too)",
        "llm_book4":    "📙 Aziz, intraday around VWAP · RU",
    }
    order = ["python", "donchian", "llm_plain", "llm_plain_en", "llm_book", "llm_regime",
             "llm_book2", "llm_book3", "llm_book4"]
    decs = st.get("deciders", {})
    ordered = [d for d in order if d in decs] + [d for d in decs if d not in order]

    lines.append("\n*Results by decider.*")
    lines.append("_Variants have DIFFERENT coverage — they were added at different times._")
    lines.append("_These rows must NOT be compared with each other: see the \"common sample\" below._")
    for dec in ordered:
        s = decs[dec]
        lines.append(f"\n{names.get(dec, dec)}")
        lines.append(f"   coverage: {s.get('points', '?')} points over {s.get('n_days', '?')} d.")
        if s.get("degenerate"):
            lines.append(f"   ⚠️ *CONSTANT*: {s['dominant_direction']} in "
                         f"{s['dominant_share']}% of cases — this is not a decision but "
                         f"the market's return. It must not be read as skill.")
        lines.append(f"   entries: {s['taken']} | refusals: {s['no_trade']}"
                     + (f" | errors: {s['errors']}" if s.get("errors") else ""))
        if s.get("win_rate") is not None:
            lines.append(f"   win rate on entries: {s['win_rate']}%")
        lines.append(f"   average per point: {s['avg_per_point']:+.3f}% _(absolute = BETA)_")
        if s.get("avg_excess") is not None:
            te = s.get("t_excess_by_day")
            ed = s.get("avg_excess_by_day")
            lines.append(f"   *VS MARKET: {s['avg_excess']:+.3f}%* per point"
                         " <- the main number")
            # t is computed on DAILY means, while the effect above is per point. With
            # days of different size the signs diverge: plain_en gave +0.094%
            # by point vs -0.038% by day. Without this line "+0.094% | t
            # -0.16" reads as a calculation error.
            #
            # THE VERDICT IS MANDATORY. A bare t on few blocks is misleading:
            # on two days this line showed "t -84.46", which looks like
            # overwhelming significance, while in fact sd was estimated from two numbers
            # and the critical value at df=1 is 12.71.
            if ed is not None and te is not None:
                nd = s.get("n_days") or 0
                lines.append(f"      by day: {ed:+.3f}% | t {te} — "
                             f"{verdict_for_t(te, nd)}")
        if s.get("avg_per_trade") is not None:
            lines.append(f"   average per trade: {s['avg_per_trade']:+.3f}%")
        if s.get("portfolio_pct") is not None:
            lines.append(f"   portfolio: {s['portfolio_pct']:+.2f}% "
                         f"_(equal weights within a window, {s['n_buckets']} "
                         f"non-overlapping windows, compounding)_")
        if s.get("t_by_day") is not None:
            lines.append(f"   t by day ({s['days']}): {s['t_by_day']} — "
                         f"{verdict_for_t(s['t_by_day'], s['days'])}")
        m = s.get("avg_missed_on_no_trade")
        if m is not None:
            verdict = "refusals cost money" if m > 0 else "refusals saved money"
            lines.append(f"   cost of refusals: {m:+.3f}% — {verdict}")

    # 12-HOUR WINDOWS: the main thing that averages over the whole period hide.
    # If on a rise all variants go LONG together, and on a fall SHORT,
    # they are not deciding but following the market, and the difference between them measures
    # luck with the period. The market direction is taken from the counterfactuals — the average
    # outcome of a hypothetical LONG over all coins of the window.
    short = {"python": "calc", "donchian": "Donchian", "llm_plain": "numbers",
             "llm_plain_en": "numbers-EN",
             "llm_book": "Schwager", "llm_regime": "briefing", "llm_book2": "Chan-RU",
             "llm_book3": "Chan-EN", "llm_book4": "Aziz"}
    wins_ = st.get("windows") or []
    if wins_:
        lines.append("\n━━━ *12-HOUR WINDOWS* ━━━")
        lines.append("_shows whether the deciders follow the market or decide on their own_")
        for w in wins_:
            arrow = "▲" if w["market"] > 0 else "▼"
            lines.append(f"\n{arrow} *{w['start'].strftime('%d.%m %H:%M')} UTC* · "
                         f"market {w['market']:+.2f}% · {w['points']} points")
            lines.append("   _LONG share / result per point:_")
            for dec in order:
                v = (w.get("deciders") or {}).get(dec)
                if not v:
                    continue
                ls = f"{v['long_share']}%" if v["long_share"] is not None else "—"
                av = f"{v['avg']:+.2f}%" if v["avg"] is not None else "—"
                lines.append(f"   {short.get(dec, dec):<10} LONG {ls:>4} → {av}")

    # An HONEST comparison: only points where ALL deciders are present at once.
    # The rows above were collected over different periods and are not directly comparable:
    # a variant that exists only for a rising day looks better not
    # because it decides better but because it said LONG more often on the rise.
    com = st.get("common") or {}
    if com.get("deciders"):
        lines.append(f"\n━━━ *COMMON SAMPLE: {com['points']} points, "
                     f"{len(com.get('days', []))} d.* ━━━")
        lines.append("_only here are the deciders directly comparable_")
        for dec in ordered:
            c = com["deciders"].get(dec)
            if not c:
                continue
            wr = f"win {c['win_rate']}%" if c.get("win_rate") is not None else "no trades"
            t = c.get("t_by_day")
            tv = (f" | t {t} — {verdict_for_t(t, c['days'])}") if t is not None else ""
            lines.append(f"{names.get(dec, dec)}")
            lines.append(f"   entries {c['taken']}/{c['points']} | {wr} | "
                         f"per point {c['avg_per_point']:+.3f}%{tv}")

    raw = st.get("llm_raw") or {}
    if raw:
        lines.append("\n*Raw opinion before the confidence threshold:*")
        for dec, r in raw.items():
            dirs = r.get("directions") or {}
            lines.append(f"   {names.get(dec, dec)}: {dirs}"
                         + (f", cut by the threshold: {r['downgraded_by_threshold']}"
                            if r.get("downgraded_by_threshold") else ""))

    dis = st.get("disagreement")
    if dis:
        lines.append(f"\n*Where the deciders diverged* ({dis['n']} points, {dis['share_pct']}%):")
        lines.append(f"   calculation: {dis['python_avg']:+.3f}% | model: {dis['llm_avg']:+.3f}%")
        lines.append(f"   the model was better in {dis['llm_better_pct']}% of disputed cases")
        lines.append("   _this is the key cut: when they agree there is nothing to compare_")

    per_coin = st.get("per_coin") or {}
    if per_coin:
        best = sorted(per_coin.items(), key=lambda x: -x[1]["diff"])
        lines.append("\n*By coin* (difference model − calculation):")
        for coin, v in best[:6]:
            mark = "model" if v["diff"] > 0 else "calculation"
            lines.append(f"   {coin}: {v['diff']:+.3f}% better {mark} (n={v['n']})")
        if len(best) > 6:
            lines.append(f"   ... {len(best)-6} more coins")

    lines.append("\n_Costs not included: decision quality is compared, not profit._")
    await _reply_long(update, "\n".join(lines))
