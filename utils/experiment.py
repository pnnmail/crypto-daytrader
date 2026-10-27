"""
EXPERIMENT: who decides better — the deterministic calculation or an LLM.

The key idea without which the comparison is impossible: at each decision point
the outcome of BOTH hypothetical trades is computed — what a LONG and what a
SHORT would have given at the same levels. Only this way can the decision "stay out" be judged.

Without counterfactuals a decider that always says NO_TRADE would show
perfect statistics (zero losses) without making a single decision — the same
survivorship bias the project already cured with a control group.
no_trade_history.

The store is separate from signal_history: there the semantics are "trades the
system decided to take", its own deduplication by UTC day and its own limits. Here
EVERY decision point is needed, including refusals, and more often than once a day.

Statistics are computed by REPLAYING the file: take the decision, plug in
the corresponding counterfactual, NO_TRADE = 0. So any metric can be
added retroactively without touching data collection.
"""
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from config import (
    EXPERIMENT_HISTORY_FILE, OUTCOME_HORIZON_HOURS, EXPERIMENT_INTERVAL_HOURS,
    EXPERIMENT_ENABLED_DEFAULT, STATE_FILE, DECIDER_VARIANTS,
)
from utils.jsonstore import load_json, save_json
from utils.flow_features import build_flow_features
from utils.market_features import build_market_features
from utils.outcomes import forward_outcomes, max_horizon_min
from utils.time_utils import now_utc

# The deciders being compared: the calculation + each LLM variant.
# The variant key in a record is "llm_<variant>".
# "donchian" is a REFERENCE LINE, not an equal participant: the rule is designed for
# holding for weeks, while here its opinion is checked after 4 hours. It answers
# the question "does all our complexity give anything beyond three lines of
# pandas", and must not be read as a test of the Donchian strategy.
DECIDERS = ("python", "donchian") + tuple(f"llm_{v}" for v in DECIDER_VARIANTS)

# Version of the DATA PRESENTED to the model. Changes when WHAT the model sees
# in the prompt changes — not only the model or the scoring formula.
#   p1 — STUMPY was presented compared with a fixed 50%
#   p2 — STUMPY is compared with the coin's base rate (as in scoring);
#        under p1 the model read "win rate 47% = bad", although a base of 47% is normal,
#        and that systematically pushed it into NO_TRADE
#   p3 — the prompt NO LONGER talks the model out of entering. In p2 it said "average
#        move 0.02%, enter only in a clear situation" — and the model gave
#        NO_TRADE 100 times out of 100, i.e. there was no decider at all.
#        Now the model always names a direction and a confidence, and cutting
#        weak opinions is done in code (DECIDER_CONFIDENCE_THRESHOLD).
#   p4 — THREE variants on one point: plain / book / regime. The p3 measurement showed
#        that the model reads the data trend-following (with the calculation's bullish technical
#        score it said SHORT in 80-100% of cases), and its shorts are indistinguishable
#        from random (56% vs a 56% base). We test whether this is fixed by
#        principles from a book and/or a briefing on the measured regime.
# p5: the same set of variants as in p4, but REGIME_BRIEF rewritten with the actually
#     measured figures (p4 had figures inflated 2-5x: "+5..+19 pp" vs
#     the measured +3) and the side asymmetry added. The regime variant sees
#     a DIFFERENT text, so its decisions are not comparable with p4.
# p6: the STUMPY block removed from the decider prompt. The model no longer reads
#     "in similar situations historically there was such-and-such percentage of rises" —
#     the text looked like statistics but was measured as noise.
# p7: book retrieval fixed. Before that it was broken twice: an English tag was used
#     to search a Russian corpus (the queries "oversold" and "downtrend" returned
#     THE SAME chunk about stop-losses), and ranking compared
#     an English tag with Russian prose — which never matched, so the score
#     was effectively made up of a bonus for text LENGTH. The book and regime variants
#     before p7 got a random selection of long chunks, mostly about
#     risk management. They must not be compared with p6.
# p8: REGIME_BRIEF got a RELIABILITY MEASURE for each rule — on how many
#     coins out of those tested the sign matched in both windows (21 of 25, 11 of 12
#     etc.). Before that the model saw effect sizes but could not tell
#     a well-confirmed rule from a weak one: RSI>70 (1 of 7) looked as
#     weighty as BB low (22 of 25). It also says separately that reliability
#     grows with the depth of the deviation.
# p9: ADDED variants book2 (Chan in Russian), book3 (Chan in English, the
#     prompt in English too) and book4 (Aziz). The prompts of the plain, book and
#     regime variants did NOT change by a single character.
#
#     IMPORTANT FOR ANALYSIS: p8 and p9 points for these three variants are COMPARABLE and
#     must be merged rather than dropped because the label differs. The label
#     changed because new variants appeared, not because the old ones
#     changed. p8 points simply have no llm_book2/3/4 fields.
# p10: added the plain_en CONTROL — an English prompt WITHOUT a book. On the first 8
#     p9 points three DIFFERENT Russian books did not change a single decision (book
#     and book2 matched plain 8 of 8, book4 — 7 of 8), while the English book3
#     changed two of eight. But book3 differs from book2 in two things at once
#     — the language of the book AND the language of the prompt, so without plain_en the effect
#     cannot be attributed to either one. The prompts of the other variants again did NOT change:
#     p8-p10 points for them are comparable and must be MERGED rather than dropped
#     because the label differs.
PROMPT_VERSION = "p10_lang_control"


def is_enabled() -> bool:
    """The flag lives in state.json, not in the config: it is switched with the command
    /experiment on|off without rebuilding the container."""
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return bool(json.load(f).get("experiment_enabled", EXPERIMENT_ENABLED_DEFAULT))
    except Exception:
        return EXPERIMENT_ENABLED_DEFAULT


def set_enabled(value: bool):
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        state = {}
    state["experiment_enabled"] = bool(value)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)


def load_experiment():
    # Via jsonstore: previously there was a bare except Exception -> {} here, and
    # on 2026-09-06 it wiped three weeks of observations. Details in
    # utils/jsonstore.py.
    return load_json(EXPERIMENT_HISTORY_FILE)


def save_experiment(data):
    save_json(EXPERIMENT_HISTORY_FILE, data)


def should_record(coin: str) -> bool:
    """Points are taken no more often than once per EXPERIMENT_INTERVAL_HOURS per coin.

    The interval matches the trade horizon, so observations do NOT
    overlap — this is mandatory for honest significance (in the stop-loss
    backtest overlap inflated t several times).
    """
    hist = load_experiment().get(coin, [])
    if not hist:
        return True
    try:
        last = datetime.fromisoformat(hist[-1]["timestamp"])
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
    except (ValueError, KeyError, TypeError):
        return True
    return (now_utc() - last) >= timedelta(hours=EXPERIMENT_INTERVAL_HOURS)


def build_context(indicators, daily_levels, orderflow, volume_ratio,
                  coin=None, df_hourly=None):
    """A snapshot of the market state AT THE MOMENT of the decision.

    Needed not for the decision itself (it has already been made) but to answer retroactively
    "in what market state did the variant decide this way".
    Without it one cannot tell "the model applied the mean-reversion
    rule" from "the model always says LONG" — in both cases the file
    would contain the same LONG. Computed from the same numbers the model sees.

    The seven top-level fields STAY AS THEY WERE: analyses were written on them
    and they are present in all accumulated points. Extended features (trend,
    volatility, volume, BTC/ETH regime, relative strength) go into the nested
    `market` block — so old and new points are read by the same code,
    and a missing block means "the point was recorded before the extension", not zero.
    """
    price = indicators.get("price")
    vwap = daily_levels.get("vwap")
    macd = indicators.get("macd")
    macd_sig = indicators.get("macd_signal")
    ctx = {
        "rsi": indicators.get("rsi"),
        "macd_hist": (round(macd - macd_sig, 6)
                      if macd is not None and macd_sig is not None else None),
        "bb_position_pct": indicators.get("bb_position_pct"),
        "vwap_dev_pct": (round((price / vwap - 1) * 100, 3)
                         if price and vwap else None),
        "volume_ratio": (volume_ratio or {}).get("volume_ratio"),
        "taker_imbalance": (orderflow or {}).get("taker_imbalance"),
        "orderflow_bias": (orderflow or {}).get("orderflow_bias"),
    }
    if coin:
        # The coin's hourly bars are already loaded higher up the stack — pass them
        # so as not to go to the exchange a second time for the same data.
        mf = build_market_features(coin, df_hourly=df_hourly)
        if mf:
            ctx["market"] = mf
        # Flow and order book are the MOST unrecoverable: an instantaneous depth snapshot
        # is provided by neither the exchange nor archives. What is not captured now is lost forever,
        # so it is computed at every point despite the extra requests.
        ff = build_flow_features(coin)
        if ff:
            ctx["flow"] = ff
    return ctx


def append_point(coin, price, levels_long, levels_short, python_decision, llm_decisions,
                 context=None, donchian=None):
    """
    Records a decision point. levels_* — a dict with stop_loss/target_1/target_2
    for the corresponding side (levels are computed by Python in both cases:
    we compare the CHOICE OF DIRECTION, not the ability to place levels).
    """
    from config import TRADER_MODEL
    from utils.signal_history import SCORING_VERSION

    data = load_experiment()
    hist = data.setdefault(coin, [])
    hist.append({
        "timestamp": now_utc().isoformat(timespec="seconds"),
        "coin": coin,
        "price": price,
        "horizon_hours": OUTCOME_HORIZON_HOURS,
        # The model and scoring version are fixed IN THE RECORD: changing either
        # makes observations incomparable, and without a label they would silently mix
        # into one statistic (the same logic as scoring_version in
        # signal_history and resolution_rule for the horizon).
        "llm_model": TRADER_MODEL,
        "scoring_version": SCORING_VERSION,
        "prompt_version": PROMPT_VERSION,
        "levels": {"LONG": levels_long, "SHORT": levels_short},
        "context": context or {},
        "python": python_decision,
        # Reference line: a simple rule on daily bars, without a model or scoring.
        # Written alongside so that the comparison is computed on THE SAME points.
        **({"donchian": donchian} if donchian else {}),
        # llm_<variant> for each decider variant: they are compared on ONE
        # point, so they live in one record, not in different files.
        **{f"llm_{v}": d for v, d in (llm_decisions or {}).items()},
        "counterfactual": {},      # filled in during resolution
    })
    data[coin] = hist[-500:]       # a per-coin cap so the file does not grow forever
    save_experiment(data)


def _simulate_side(df, entry_dt, entry_px, side, levels, horizon_hours):
    """A pass over MINUTE bars: what a trade in this direction would have given.

    The stop is checked against the low/high of every minute, not the close — otherwise
    a backtest of intraday stops systematically overstates the result. When
    the stop and target hit in the same minute, the stop wins (conservative,
    biases AGAINST the system — the only safe direction of error).
    """
    stop = levels.get("stop_loss")
    t1 = levels.get("target_1")
    t2 = levels.get("target_2")
    is_long = side == "LONG"
    end = entry_dt + timedelta(hours=horizon_hours)
    win = df[(df.index > entry_dt) & (df.index <= end)]
    if win.empty:
        return None

    for ts, row in win.iterrows():
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
        if stop_hit:
            px, oc = stop, "stop"
        elif t2_hit:
            px, oc = t2, "target_2"
        elif t1_hit:
            px, oc = t1, "target_1"
        else:
            continue
        ret = (px - entry_px) / entry_px * 100 * (1 if is_long else -1)
        return {"outcome": oc, "return_pct": round(ret, 4), "time": ts.isoformat()}

    last = float(win["Close"].iloc[-1])
    ret = (last - entry_px) / entry_px * 100 * (1 if is_long else -1)
    return {"outcome": "close", "return_pct": round(ret, 4), "time": win.index[-1].isoformat()}


def _point_dt(p):
    """The point's time as an aware datetime, or None if the record is broken."""
    try:
        dt = datetime.fromisoformat(p["timestamp"])
    except (ValueError, KeyError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _elapsed_min(p):
    """How many minutes have passed since the point."""
    dt = _point_dt(p)
    if dt is None:
        return None
    return (now_utc() - dt).total_seconds() / 60.0


def resolve_all_points():
    """Computes counterfactuals and multi-horizon outcomes for matured points.

    Two different maturities in one pass: a counterfactual needs 4 hours,
    outcomes with excursions need a day. Candles are fetched once per coin.

    Returns the number of updated COUNTERFACTUALS (outcomes are counted separately and
    are not included in this number — otherwise comparison with past measurements would drift).
    """
    from data.fetcher import fetch_candles

    data = load_experiment()
    updated = 0
    # Computed outcomes, addressed independently of the position in the list:
    # by the time of writing the file is already different, indices are useless.
    resolved = {}
    # Multi-horizon outcomes and excursions — in a separate dict, because
    # they mature LATER than the counterfactual: it needs 4 hours, these need 24.
    resolved_out = {}
    max_h_min = max_horizon_min()
    for coin, points in data.items():
        pending = [p for p in points if not p.get("counterfactual")]
        # Points whose LONGEST horizon has expired but whose outcomes have not
        # been computed yet. Computed once and in full: so minute bars for
        # a day are fetched once per point, not every 15 minutes again.
        pending_out = [
            p for p in points
            if not p.get("outcomes") and _elapsed_min(p) is not None
            and _elapsed_min(p) >= max_h_min
        ]
        if not pending and not pending_out:
            continue
        # one download of minute bars per coin — enough for all its open points
        try:
            oldest = min(datetime.fromisoformat(p["timestamp"])
                         for p in (pending + pending_out))
            if oldest.tzinfo is None:
                oldest = oldest.replace(tzinfo=timezone.utc)
            # The window must cover not only the point's age but the horizon itself:
            # a daily outcome needs bars for a further day from the point.
            days_back = (now_utc() - oldest).total_seconds() / 86400 + 0.05
            if pending_out:
                days_back = max(days_back, max_h_min / 1440.0 + 0.1)
            df = fetch_candles(coin, "1m", days_back=max(days_back, 0.05))
        except Exception as e:
            print(f"   ⚠️ experiment: could not load candles for {coin}: {e}")
            continue
        if df.empty:
            continue

        for p in pending_out:
            dt = _point_dt(p)
            if dt is None:
                continue
            out = forward_outcomes(df, dt, p.get("price"))
            if out:
                p["outcomes"] = out
                resolved_out[(coin, p["timestamp"])] = out

        for p in pending:
            try:
                dt = datetime.fromisoformat(p["timestamp"])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except (ValueError, KeyError, TypeError):
                continue
            horizon = p.get("horizon_hours", OUTCOME_HORIZON_HOURS)
            if now_utc() < dt + timedelta(hours=horizon):
                continue      # the horizon has not expired yet
            cf = {}
            for side in ("LONG", "SHORT"):
                res = _simulate_side(df, dt, p["price"], side, p["levels"].get(side, {}), horizon)
                if res:
                    cf[side] = res
            if len(cf) == 2:
                p["counterfactual"] = cf
                resolved[(coin, p["timestamp"])] = cf
                updated += 1
    if updated or resolved_out:
        # RE-READ before writing. Resolution takes minutes: minute candles are
        # fetched over the network for each coin. In that time autoscan manages to
        # append new points, and writing the dict read at the start
        # would silently erase them — a classic lost update. So
        # we take the fresh file and transfer ONLY the computed outcomes into it,
        # addressing points by the pair (coin, timestamp).
        fresh = load_experiment()
        applied = 0
        for coin, points in fresh.items():
            for p in points:
                key = (coin, p.get("timestamp"))
                cf = resolved.get(key)
                if cf and not p.get("counterfactual"):
                    p["counterfactual"] = cf
                    applied += 1
                out = resolved_out.get(key)
                if out and not p.get("outcomes"):
                    p["outcomes"] = out
        save_experiment(fresh)
        if applied != updated:
            print(f"   ⚠️ experiment: outcomes computed {updated}, "
                  f"written {applied} — some points disappeared from the file")
    return updated


def _decision_return(point, decider):
    """The return this decider would have got at this point.
    NO_TRADE = 0 (stayed out). A model error -> None (excluded)."""
    d = point.get(decider) or {}
    if d.get("_error"):
        return None
    direction = (d.get("direction") or "NO_TRADE").upper()
    cf = point.get("counterfactual") or {}
    if not cf:
        return None
    if direction.endswith("LONG"):
        return (cf.get("LONG") or {}).get("return_pct")
    if direction.endswith("SHORT"):
        return (cf.get("SHORT") or {}).get("return_pct")
    return 0.0     # NO_TRADE — neither earned nor lost money


def _missed(point, decider):
    """
    What the decider missed by refusing — relative to a COIN FLIP, not the ideal.

    This used to be max(LONG, SHORT), i.e. the best outcome in hindsight.
    The metric turned out useless: if the price moved anywhere, one of the sides
    is in profit, so the value is almost always positive and ANY refusal looked like
    a mistake. The honest reference is the average of the two directions: what a decider
    flipping a coin would get. A positive value now
    means "the refusal really cost money", a negative one — "the refusal saved money".
    """
    d = point.get(decider) or {}
    if d.get("_error"):
        return None
    if (d.get("direction") or "NO_TRADE").upper() != "NO_TRADE":
        return None
    cf = point.get("counterfactual") or {}
    vals = [v.get("return_pct") for v in cf.values() if v.get("return_pct") is not None]
    return (sum(vals) / len(vals)) if len(vals) == 2 else None


def _tstat_by_day(pairs):
    """t over DAILY blocks: points within a day are correlated (coins move
    together), a naive per-trade t is inflated."""
    byday = defaultdict(list)
    for day, val in pairs:
        byday[day].append(val)
    means = [statistics.mean(v) for v in byday.values()]
    if len(means) < 2:
        return None, len(means)
    sd = statistics.stdev(means)
    if sd == 0:
        return None, len(means)
    return statistics.mean(means) / (sd / math.sqrt(len(means))), len(means)


def _mean_by_day(pairs):
    """The mean of DAILY means — the same quantity the t is computed on.

    It differs from the mean over all points when days have different sizes, and
    the signs can diverge. That is exactly what happened with plain_en: +0.094% by point vs
    -0.038% by day at t=-0.16. Printing a per-point effect next to a per-day t
    means showing a plus and a minus in one line without explanation.
    """
    byday = defaultdict(list)
    for day, val in pairs:
        byday[day].append(val)
    if not byday:
        return None
    return statistics.mean(statistics.mean(v) for v in byday.values())


def _compound(returns_pct):
    """Compounding over consecutive periods, in percent.

    Returns must not be summed: METHODOLOGY.md says so explicitly, and
    the error already cost one wrong conclusion about Donchian.
    """
    acc = 1.0
    for r in returns_pct:
        acc *= (1.0 + r / 100.0)
    return (acc - 1.0) * 100.0


def window_stats(points, hours=12, limit=4):
    """A breakdown by windows of `hours` hours: what the MARKET did and what the deciders did.

    Why separately from the overall statistics. Averages over the whole period hide the most
    important thing: if on a rise all variants go LONG together, and on a fall
    SHORT, they are not deciding but following the market — and the difference between them
    measures not the quality of thinking but who got lucky with the period.

    The market direction is taken from the counterfactuals: the average outcome of a hypothetical
    LONG over all coins of the window. This is more honest than any indicator, because it
    is computed on the same data the decisions were judged on.
    """
    if not points:
        return []
    buckets = defaultdict(list)
    for p in points:
        try:
            ts = datetime.fromisoformat(p["timestamp"])
        except (ValueError, KeyError, TypeError):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        key = ts.replace(hour=(ts.hour // hours) * hours,
                         minute=0, second=0, microsecond=0)
        buckets[key].append(p)

    rows = []
    for key in sorted(buckets, reverse=True)[:limit]:
        pts = buckets[key]
        longs = [(p["counterfactual"].get("LONG") or {}).get("return_pct")
                 for p in pts if p.get("counterfactual")]
        longs = [x for x in longs if x is not None]
        if not longs:
            continue
        market = statistics.mean(longs)
        decs = {}
        for dec in DECIDERS:
            have = [p for p in pts if dec in p and not (p.get(dec) or {}).get("_error")]
            if not have:
                continue
            dirs = [(p[dec] or {}).get("direction") for p in have]
            taken = [d for d in dirs if d in ("LONG", "SHORT")]
            rets = [_decision_return(p, dec) for p in have]
            rets = [r for r in rets if r is not None]
            decs[dec] = {
                "n": len(have),
                "long_share": (round(100 * dirs.count("LONG") / len(taken))
                               if taken else None),
                "avg": round(statistics.mean(rets), 3) if rets else None,
            }
        rows.append({"start": key, "hours": hours, "points": len(pts),
                     "market": round(market, 3), "deciders": decs})
    return rows


def compute_experiment_stats():
    """The full comparison summary. Computed by replaying the file."""
    data = load_experiment()
    points = [p for pts in data.values() for p in pts if p.get("counterfactual")]
    out = {"total_points": sum(len(v) for v in data.values()), "resolved": len(points), "deciders": {}}
    if not points:
        return out

    # Which models and scoring versions the data was collected with. If there is more
    # than one value here, the statistics must NOT be read as a single experiment.
    out["models"] = dict(sorted(
        __import__("collections").Counter(p.get("llm_model", "?") for p in points).items(),
        key=lambda x: -x[1]))
    out["scoring_versions"] = dict(sorted(
        __import__("collections").Counter(p.get("scoring_version", "?") for p in points).items(),
        key=lambda x: -x[1]))
    out["prompt_versions"] = dict(sorted(
        __import__("collections").Counter(p.get("prompt_version", "?") for p in points).items(),
        key=lambda x: -x[1]))

    # A BENCHMARK FOR EACH POINT: what a simple market hold would have given at the same moment.
    #
    # Without it absolute P&L measures BETA, not ALPHA: on a rising market
    # a decider that always goes long scores points not for decision quality.
    # That is exactly what was observed — variants with a high share of LONG looked
    # best precisely in rising windows.
    #
    # Benchmark = the average outcome of a hypothetical LONG over ALL coins of the same
    # wave. Computed from already saved counterfactuals, so the metric can be
    # added retroactively to all accumulated points.
    wave_market = {}
    _wave_acc = defaultdict(list)
    for p_ in points:
        r = (p_["counterfactual"].get("LONG") or {}).get("return_pct")
        if r is not None:
            _wave_acc[p_["timestamp"][:13]].append(r)
    for k, v in _wave_acc.items():
        wave_market[k] = statistics.mean(v)

    for dec in DECIDERS:
        rets, days, taken, no_trade, errors, wins = [], [], 0, 0, 0, 0
        excess, excess_days = [], []
        missed_vals = []
        # Buckets of OUTCOME_HORIZON_HOURS for the PORTFOLIO computation, see below.
        buckets = defaultdict(list)
        for p in points:
            # The variant may NOT HAVE EXISTED yet when the point was recorded:
            # book2/book3/book4 were added later than the others. Without this check
            # missing data counted as NO_TRADE — plain_en had
            # 193 "refusals" at 91 real points, which both inflated caution
            # and pulled the average result towards zero.
            if dec not in p:
                continue
            d = p.get(dec) or {}
            if d.get("_error"):
                errors += 1
                continue
            r = _decision_return(p, dec)
            if r is None:
                continue
            day = p["timestamp"][:10]
            rets.append(r)
            days.append((day, r))
            # Bucket = a NON-OVERLAPPING window as long as the holding horizon.
            # Within a bucket positions run in parallel, between buckets —
            # sequentially. Only then can they be combined at all.
            buckets[(day, int(p["timestamp"][11:13]) // OUTCOME_HORIZON_HOURS)].append(r)
            mk = wave_market.get(p["timestamp"][:13])
            if mk is not None:
                # A refusal also has a cost relative to the market: sitting out a rise
                # is a relative loss, sitting out a fall — a gain.
                excess.append(r - mk)
                excess_days.append((day, r - mk))
            direction = (d.get("direction") or "NO_TRADE").upper()
            if direction == "NO_TRADE":
                no_trade += 1
                m = _missed(p, dec)
                if m is not None:
                    missed_vals.append(m)
            else:
                taken += 1
                if r > 0:
                    wins += 1
        if not rets:
            continue
        t, nday = _tstat_by_day(days)
        traded = [r for r, p in zip(rets, points) if r != 0.0]
        # The days for which the variant HAS data: without this one cannot see
        # that variants were collected over different periods and are not directly comparable.
        dset = sorted(set(d for d, _ in days))
        # A DEGENERATE DECIDER: if it almost always says the same thing, it
        # is not deciding but a constant, and its "result" is just
        # the return of the corresponding side of the market. This already burned us in
        # p2, where the model gave NO_TRADE 100 times out of 100 and formally looked
        # the best. Without a flag such a row is read as skill.
        dircnt = defaultdict(int)
        for p_ in points:
            if dec not in p_:
                continue
            dd = (p_.get(dec) or {}).get("direction")
            if dd:
                dircnt[dd] += 1
        top_dir, top_n = (max(dircnt.items(), key=lambda x: x[1])
                          if dircnt else (None, 0))
        total_dir = sum(dircnt.values()) or 1
        degenerate = (top_n / total_dir) >= 0.95 and total_dir >= 20
        out["deciders"][dec] = {
            "points": len(rets), "taken": taken, "no_trade": no_trade, "errors": errors,
            "day_list": dset, "n_days": len(dset),
            "avg_excess": round(statistics.mean(excess), 4) if excess else None,
            # The same effect but computed over DAILY means — that is what the t below
            # refers to. Without this pair the line "effect +0.094% | t -0.16"
            # looks like a calculation error, although these are different averaging units.
            "avg_excess_by_day": (lambda m: round(m, 4) if m is not None else None)(
                _mean_by_day(excess_days)),
            "t_excess_by_day": (lambda r: round(r[0], 2) if r[0] is not None else None)(
                _tstat_by_day(excess_days)) if len(excess_days) > 1 else None,
            "degenerate": degenerate,
            "dominant_direction": top_dir if degenerate else None,
            "dominant_share": round(100 * top_n / total_dir) if degenerate else None,
            "win_rate": round(wins / taken * 100, 1) if taken else None,
            "avg_per_point": round(statistics.mean(rets), 4),
            "avg_per_trade": round(statistics.mean([r for r in rets if r != 0.0]), 4) if any(r != 0.0 for r in rets) else None,
            # A PORTFOLIO instead of a sum of returns.
            #
            # This used to be sum(rets), which was identical to
            # "number of points x average per point": plain gave -77.61%,
            # plain_en +63.74%, and it read as an account result. In fact
            # positions run IN PARALLEL — a median of 7 coins per wave, a horizon
            # of 4 hours, i.e. about 28 open at once — while the sum
            # added them as if they ran one after another. On top of that it
            # grew with coverage, directly contradicting the warning "the variants
            # have different coverage, the rows are not comparable".
            #
            # Here: equal weights within a NON-OVERLAPPING bucket as long as the
            # horizon, compounding between buckets. This is a real
            # capital allocation rule, not a recomputation of the average.
            "portfolio_pct": round(_compound([statistics.mean(v)
                                              for _, v in sorted(buckets.items())]), 2)
                             if buckets else None,
            "n_buckets": len(buckets),
            "t_by_day": round(t, 2) if t is not None else None,
            "days": nday,
            # cost of abstaining: >0 means refusals cost money on average,
            # <0 means refusals saved money
            "avg_missed_on_no_trade": round(statistics.mean(missed_vals), 4) if missed_vals else None,
        }

    # ─── the raw opinion of EACH variant (BEFORE the confidence threshold) ───
    # Shows whether there is a directional view at all, separately from whether
    # it passes the threshold. If raw is almost always NO_TRADE — a question for the model
    # or the prompt; if raw is directional but decisions are few — a question for the threshold.
    out["llm_raw"] = {}
    for dec in DECIDERS:
        # A block about the model's RAW OPINION before the confidence threshold. Deterministic
        # deciders are not included: they have neither raw_direction nor a threshold, and
        # in the report Donchian was printed as "{None: 100}".
        if dec in ("python", "donchian"):
            continue
        raw = [(p, (p.get(dec) or {}).get("raw_direction"))
               for p in points if p.get(dec) and not (p.get(dec) or {}).get("_error")]
        if not raw:
            continue
        cnt = __import__("collections").Counter(r for _, r in raw)
        rvals = []
        for p, rd in raw:
            cf = p.get("counterfactual") or {}
            if rd and rd.endswith("LONG"):
                v = (cf.get("LONG") or {}).get("return_pct")
            elif rd and rd.endswith("SHORT"):
                v = (cf.get("SHORT") or {}).get("return_pct")
            else:
                v = 0.0
            if v is not None:
                rvals.append(v)
        out["llm_raw"][dec] = {
            "directions": dict(cnt),
            "downgraded_by_threshold": sum(1 for p, _ in raw if (p.get(dec) or {}).get("downgraded_from")),
            "avg_per_point": round(statistics.mean(rvals), 4) if rvals else None,
        }

    # ─── head to head: only points where the deciders DIVERGED ───
    from config import DECIDER_PRIMARY
    primary = f"llm_{DECIDER_PRIMARY}"
    dis = []
    for p in points:
        pd_ = ((p.get("python") or {}).get("direction") or "NO_TRADE").upper()
        ld_ = ((p.get(primary) or {}).get("direction") or "NO_TRADE").upper()
        if not p.get(primary) or (p.get(primary) or {}).get("_error"):
            continue
        if pd_ != ld_:
            dis.append((p, _decision_return(p, "python"), _decision_return(p, primary)))
    if dis:
        pv = [x[1] for x in dis if x[1] is not None]
        lv = [x[2] for x in dis if x[2] is not None]
        out.setdefault("disagreement_variant", primary)
        out["disagreement"] = {
            "n": len(dis),
            "share_pct": round(len(dis) / len(points) * 100, 1),
            "python_avg": round(statistics.mean(pv), 4) if pv else None,
            "llm_avg": round(statistics.mean(lv), 4) if lv else None,
            "llm_better_pct": round(sum(1 for _, a, b in dis if a is not None and b is not None and b > a)
                                     / max(len([1 for _, a, b in dis if a is not None and b is not None]), 1) * 100, 1),
        }

    # ─── breakdown by coin ───
    per_coin = {}
    for coin, pts in data.items():
        res = [p for p in pts if p.get("counterfactual")]
        if len(res) < 3:
            continue
        pv = [x for x in (_decision_return(p, "python") for p in res) if x is not None]
        lv = [x for x in (_decision_return(p, primary) for p in res) if x is not None]
        if not pv or not lv:
            continue
        per_coin[coin] = {
            "n": len(res),
            "python": round(statistics.mean(pv), 4),
            "llm": round(statistics.mean(lv), 4),
            "diff": round(statistics.mean(lv) - statistics.mean(pv), 4),
        }
    out["per_coin"] = per_coin

    out["windows"] = window_stats(points)

    # A COMMON SAMPLE: only points that have ALL the compared deciders.
    #
    # Without this the comparison lies. Variants were added at different times, and
    # book2/book3/book4 exist only for the last day — a rising one.
    # Their "+0.18% per point" vs "+0.01%" for plain is not superiority but
    # different periods: on a rise whoever says LONG more often wins.
    # The report header promised "on the same sample of points" — here that
    # promise is finally kept.
    common = [p for p in points if all(dec in p for dec in DECIDERS)]
    out["common"] = {"points": len(common),
                     "days": sorted(set(p["timestamp"][:10] for p in common))}
    if common:
        rows = {}
        for dec in DECIDERS:
            rets, days = [], []
            taken = wins = 0
            for p in common:
                d = p.get(dec) or {}
                if d.get("_error"):
                    continue
                r = _decision_return(p, dec)
                if r is None:
                    continue
                rets.append(r)
                days.append((p["timestamp"][:10], r))
                if (d.get("direction") or "NO_TRADE").upper() != "NO_TRADE":
                    taken += 1
                    wins += r > 0
            if not rets:
                continue
            t, nday = _tstat_by_day(days)
            rows[dec] = {"points": len(rets), "taken": taken,
                         "win_rate": round(wins / taken * 100, 1) if taken else None,
                         "avg_per_point": round(statistics.mean(rets), 4),
                         "t_by_day": round(t, 2) if t is not None else None,
                         "days": nday}
        out["common"]["deciders"] = rows
    return out
