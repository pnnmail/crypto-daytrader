"""
Scoring and decisions.

IMPORTANT ABOUT THE DIRECTION OF FACTORS (measured 2026-08-11, 10 coins, 400 days of hourly
history, ~95k observations, horizons 4/12/24h):

  In crypto at these horizons MEAN REVERSION works, not trend
  following. Measured relative to the coin's base rate (48.8% at 4h):

    RSI < 30            -> win rate +5.9 pp     (contrarian, was right)
    BB near_lower       -> +3.6 pp              (contrarian, was right)
    price BELOW VWAP    -> +5.0 pp              <- was -1, now +1
    price ABOVE VWAP    -> -4.1 pp              <- was +1, now -1
    MACD bearish        -> +2.2 pp              <- was -1, now +1
    MACD bullish        -> -2.4 pp              <- was +1, now -1

  So MACD and VWAP were MIRROR-wrong and systematically pushed
  the system into SHORT: price below VWAP — the typical state in a pullback — gave -1,
  although historically that is exactly where price rises more often. Under the old scoring
  LONG>=2 was reached 1091 times out of 95k points vs 1473 for SHORT, and the
  ±3 threshold was almost never reached. After the flip LONG>=3 is reached 10368 times with
  a 54.9% win rate, SHORT>=3 — 8621 times, i.e. symmetric and above base.

  RSI and Bollinger are NOT touched — they are already contrarian and confirmed by measurement.

SIDE ASYMMETRY (measured 2026-08-16, ALL 25 coins, 400 days, 4h step without
overlap; a training window of ~280 days and a test window of 120 days SEPARATED in
time):

  Criterion — on how many coins the sign of the effect MATCHED in both windows. A random
  factor would match on about half, so this is the protection against
  some coin out of 25 showing anything at all.

    factor              matched     p       v5 verdict
    price below VWAP    21/25    0.001    +1  (as before)
    price above VWAP    21/25    0.001    -1  (as before)
    RSI < 30            11/12    0.006    +1  (as before)
    RSI > 70             1/7     0.125    VOTE REMOVED
    MACD bearish        19/25    0.015    +1  (as before)
    MACD bullish        16/25    0.230    VOTE REMOVED
    BB low (<20%)       22/25    0.000    +1  (as before)
    BB high (>80%)      14/25    0.690    VOTE REMOVED

  OVERSOLD PREDICTS A RISE, OVERBOUGHT DOES NOT PREDICT A FALL. Only VWAP works
  both ways. The three removed votes gave -1 without any
  measured basis and systematically pushed the system into SHORT — that is the
  reason for the complaint "the history is all shorts".

  Mean reversion is confirmed by a DOSE effect: below VWAP in general +3.2 pp (OOS),
  deeper than 0.5% -> +4.2, deeper than 1.5% -> +7.6 (19/21 coins, p=0.000). A random
  correlation would not grow with the size of the deviation. The mirror side breaks
  again: above VWAP by more than 1.5% gives 8/17, p=1.000.

  A SEPARATE REGIME PER COIN IS NOT JUSTIFIED:
  over three windows no coin goes against reversion in all three, and
  the correlation of effect strength between windows is r=+0.02..+0.35, i.e. a coin's
  character does not carry over. Coins positive in all three windows: 11 of 25 vs
  11.0 expected by chance — the whole "per coin" structure is explained by the common
  effect plus noise.

  CONSEQUENCE FOR THE RANGE: the positive side stayed ±9, the negative one
  shrank to -6 (order flow -2, STUMPY -2, VWAP -1, funding -1). This is not
  an oversight but a direct reflection of the fact that we have less evidence for shorts.
  So confidence is normalised to EACH side's own
  maximum — otherwise a short with its full set of factors would show 0.77
  where a long shows 0.90, and the number would read as "the system is less
  confident", although all available factors were collected.

EXIT RULE (backtest 2026-08-13 on minute bars, 1436 trades in-sample
+ 967 out-of-sample on an independent window; the stop was checked against the low/high of every
minute, not the hourly close):

  stop                R:R filter    in-sample        out-of-sample
  structural (old)    >= 1.0        +0.041%          +0.100%
  structural          off           +0.082%          +0.091%
  ATR 2.5             >= 1.0        +0.081% (t 0.9)  -0.124% (sign flipped!)
  ATR 2.5 (new)       off           +0.179% win 67%  +0.126% win 64%

  A structural stop (S1/POC/prev_low) does not suit contrarian entries:
  we buy BELOW VWAP expecting a bounce BACK TO VWAP, i.e. the target is close by
  construction, while the structural level is far — noise knocked out the stop before the thesis
  had a chance to work. On live data this gave 34% of trades stopped out.

  The R:R filter was turned off NOT as a separate decision but as a pair: it requires a distant
  target, while in mean reversion the close ones earn. With a wide
  stop it kept 8-12% of trades and flipped the result negative.
  THESE TWO PARAMETERS MUST NOT BE CHANGED SEPARATELY.

  An honest caveat: the harm of the filter BY ITSELF (with a structural stop)
  was confirmed only in-sample, out of sample it was neutral. What is proven is
  the pair, not the filter as such.
"""
from typing import Dict, Tuple

from config import LONG_THRESHOLD, SHORT_THRESHOLD, SPREAD_ROUND_TRIP_PCT
from utils.helpers import escape_markdown, safe_format_float

# Stop as a multiple of ATR. 2.5 was chosen by backtest: the variants 1.0/1.5/2.5/no stop
# ranked the same on both windows, and 2.5 gives practically the same
# result as no stop at all (+0.179 vs +0.181 in-sample),
# but caps the worst trade at -3.56% instead of -5.18%.
STOP_ATR_MULT = 2.5

# Minimum distance to the target as a multiple of ATR. Needed since the
# R:R filter was turned off: without it a trade with the target at the price itself (price right at VWAP)
# passed as a valid signal with R:R 1:0.0. In the backtest such cases were ~1% of the sample,
# they did not affect the average result, but as a trading instruction this is garbage.
MIN_TARGET_ATR = 0.5

MAX_REASONABLE_RR = 10.0
# 0 = filter off. See the table above: with a 2.5 ATR stop a threshold of 1.0
# kept 8% of trades and gave -0.124% on the independent window.
MIN_ENTRY_RR = 0.0

# Order Flow ±2 + Technical (oversold side only, +3) +
# VWAP ±1 + Funding ±1. STUMPY removed in v6 as measured noise.
MAX_TOTAL_SCORE = 7
# The negative side is shorter: in v5 three unconfirmed -1 votes were removed
# (RSI>70, MACD bullish, BB upper), in v6 STUMPY went. Only these play on the minus side:
# order flow -2, VWAP -1, funding -1.
MAX_SHORT_SCORE = 4

# STUMPY with top_k=100 usually gives n~100 (see data/stumpy_finder.py) — the gate
# was raised from 5 (stocks) to 20, otherwise it reacts only to completely degenerate
# cases. confidence_weight is divided by 40, not 15: at a typical n~100
# the weight is almost always 1.0, which is justified — the sample is an order of magnitude larger than for
# stocks and deserves more trust.
STUMPY_MIN_SAMPLES = 20
STUMPY_CONFIDENCE_DIVISOR = 40

# How much a pattern's win_rate must exceed THE COIN'S BASE RATE to
# count as a signal. Previously the comparison was with a fixed 50%, and that was
# a bug: the base rate in crypto is below 50 and specific to each coin (measured: BTC
# 48.5%, ETH 49.5%, SOL 49.3%, ENA 42.5%). For ENA ANY pattern showed
# ~42.5% <= the threshold of 45 and automatically got -1 — the factor measured market drift,
# not the pattern. That is also where the largest negative contribution came from (-0.33).
STUMPY_EXCESS_STRONG_PP = 10.0
STUMPY_EXCESS_WEAK_PP = 5.0

RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30


def compute_factor_scores(indicators: Dict, daily_levels: Dict, orderflow: Dict,
                           stumpy_data: Dict, volume_ratio: Dict, funding_data: Dict = None) -> Tuple[Dict, Dict]:
    """Computes factor scores. Returns (scores, reasons)."""
    scores = {"orderflow": 0, "technical": 0, "stumpy": 0, "vwap": 0, "total": 0}
    reasons = {"orderflow": "", "technical": "", "stumpy": "", "vwap": "",
               "rsi": "", "macd": "", "bb": ""}
    if funding_data is not None:
        scores["funding"] = 0
        reasons["funding"] = ""

    # ═══ 1. ORDER FLOW (±2) ═══
    if orderflow and not orderflow.get('_error'):
        of_bias = orderflow.get('orderflow_bias', '')
        large_player = orderflow.get('large_player_active', False)
        vr = volume_ratio.get('volume_ratio', 1) or 1
        delta_divergence = orderflow.get('delta_divergence', False)
        absorption = orderflow.get('absorption_detected', False)

        if of_bias == 'bullish':
            strong = large_player and vr > 1.2
            base = 2 if strong else 1
            reasons['orderflow'] = (f"BULL + large imbalance + high volume ({vr:.1f}x)" if strong
                                     else "BULL + large imbalance" if large_player else "BULL")
        elif of_bias == 'bearish':
            strong = large_player and vr > 1.2
            base = -2 if strong else -1
            reasons['orderflow'] = (f"BEAR + large imbalance + high volume ({vr:.1f}x)" if strong
                                     else "BEAR + large imbalance" if large_player else "BEAR")
        else:
            base = 0
            reasons['orderflow'] = "neutral"

        if delta_divergence and base:
            if abs(base) == 2:
                base = 1 if base > 0 else -1
            reasons['orderflow'] += " + divergence (weakened)"

        scores['orderflow'] = base
        if delta_divergence and absorption:
            scores['orderflow'] = 0
            reasons['orderflow'] += " + absorption (reversal signal)"
    elif orderflow and orderflow.get('_error'):
        reasons['orderflow'] = "model error — factor not counted"

    # ═══ 2. TECHNICAL (±3: RSI ±1, MACD ±1, BB ±1) ═══
    rsi = indicators.get('rsi', 50)
    macd, macd_signal = indicators.get('macd', 0), indicators.get('macd_signal', 0)
    price = indicators.get('price', 0)
    bb_lower = indicators.get('bb_lower', price * 0.95)
    bb_upper = indicators.get('bb_upper', price * 1.05)

    reasons_tech, tech_score = [], 0
    if rsi > RSI_OVERBOUGHT:
        # v5: NO vote. Overbought does not predict a fall — the sign
        # held on only 1 coin of 7 (p=0.125). See the table in the header.
        reasons['rsi'] = f"RSI {rsi:.1f} (overbought — not scored, see v5)"
    elif rsi < RSI_OVERSOLD:
        tech_score += 1
        reasons['rsi'] = f"RSI {rsi:.1f} (oversold)"
    else:
        reasons['rsi'] = f"RSI {rsi:.1f} (neutral)"
    reasons_tech.append(reasons['rsi'])

    # THE SIGN IS FLIPPED relative to the classic reading — see the module header:
    # in the measurement a bullish MACD preceded a fall (46.4% win at a base of 48.8%),
    # a bearish one — a rise (51.0%). Read contrarian, like RSI and Bollinger.
    if macd > macd_signal and macd > 0:
        # v5: NO vote. The sign held on 16 coins of 25 (p=0.230) —
        # indistinguishable from random. The bearish MACD was confirmed.
        reasons['macd'] = "MACD bullish (not scored, see v5)"
    elif macd < macd_signal and macd < 0:
        tech_score += 1
        reasons['macd'] = "MACD bearish (contrarian: a bounce up is more likely)"
    else:
        reasons['macd'] = "MACD neutral"
    reasons_tech.append(reasons['macd'])

    if bb_upper > bb_lower:
        bb_position = (price - bb_lower) / (bb_upper - bb_lower) * 100
        if bb_position > 80:
            # v5: NO vote. 14 coins of 25 (p=0.690) — pure noise, while
            # the lower band was confirmed most strongly of all (22/25, p=0.000).
            reasons['bb'] = f"BB near_upper ({bb_position:.0f}%) — not scored, see v5"
        elif bb_position < 20:
            tech_score += 1
            reasons['bb'] = f"BB near_lower ({bb_position:.0f}%)"
        else:
            reasons['bb'] = f"BB middle ({bb_position:.0f}%)"
        reasons_tech.append(reasons['bb'])

    scores['technical'] = tech_score
    reasons['technical'] = ", ".join(reasons_tech)

    # ═══ 3. STUMPY — NO VOTE since v6 ═══
    # Measured (1088 points, 8 coins,
    # the rule reproduced literally, without look-ahead):
    # no predictive power either at 4h or at 3d.
    #
    #   score   points   share up (base 53.4%)
    #     -1     175        46.3%
    #      0     645        55.7%   <- BEST where the factor is silent
    #     +1     262        53.8%   <- zero beyond base
    #
    # Following its sign: 53.3% wins, mean +0.008%. Correlation of
    # the prediction with the outcome r=+0.000 (at 3d r=+0.017). That the "factor is
    # silent" group beats both of its opinionated groups is the clearest sign of noise.
    #
    # The values ARE still computed (cost 0.02 s) and shown in the report
    # for reference, but they are not part of the decision and are not fed into prompts.
    scores['stumpy'] = 0
    reasons['stumpy'] = "not scored since v6 — measured as noise (r=0.00)"

    # ═══ 4. VWAP (±1) ═══
    # THE SIGN IS FLIPPED — see the module header. This is the strongest effect
    # measured: price below VWAP -> win rate 53.8% (+5.0 pp to base),
    # above VWAP -> 44.8% (-4.1 pp). A deviation from VWAP reverts
    # rather than continues.
    vwap = daily_levels.get('vwap', price)
    if vwap and price:
        vwap_diff = (price - vwap) / vwap * 100
        if vwap_diff > 1.0:
            scores['vwap'] = -1
            reasons['vwap'] = f"price above VWAP (+{safe_format_float(vwap_diff)}%) — contrarian: a return down is more likely"
        elif vwap_diff < -1.0:
            scores['vwap'] = 1
            reasons['vwap'] = f"price below VWAP ({safe_format_float(vwap_diff)}%) — contrarian: a bounce up is more likely"
        else:
            reasons['vwap'] = f"price at VWAP ({safe_format_float(vwap_diff)}%)"
    else:
        reasons['vwap'] = "no VWAP"

    # ═══ 5. FUNDING (±1, stage 4 — used only if funding_data is passed) ═══
    if funding_data is not None and not funding_data.get('error'):
        rate = funding_data.get('funding_rate_pct')
        if rate is not None:
            # Extreme funding is crowd skew, often preceding a squeeze
            # in the opposite direction (a contrarian reading).
            # The 0.01% threshold was calibrated on a live Kraken Futures measurement on 10
            # coins (2026-08-10): the typical range was 0.0001-0.003%,
            # XBT — an anomaly of 0.00035%. 0.01% is well above the whole observed
            # range, an initial calibration, to be refined empirically later.
            if rate > 0.01:
                scores['funding'] = -1
                reasons['funding'] = f"funding {rate:+.4f}% — longs overheated, risk of a squeeze down"
            elif rate < -0.01:
                scores['funding'] = 1
                reasons['funding'] = f"funding {rate:+.4f}% — shorts overheated, risk of a squeeze up"
            else:
                reasons['funding'] = f"funding {rate:+.4f}% — neutral"
        else:
            reasons['funding'] = "no data (funding rate empty in the Kraken response)"
    elif funding_data is not None:
        # funding_data was passed but with an error (no perpetual on Kraken Futures
        # for this coin, the exchange is unavailable etc.) — previously the reason stayed
        # an empty string from initialisation, and the report printed "Funding: 0 —"
        # without an explanation. Now the reason is visible.
        reasons['funding'] = f"no data ({funding_data.get('error', '?')})"

    scores['total'] = sum(v for k, v in scores.items() if k != 'total')
    return scores, reasons


def calculate_entry_levels(direction: str, price: float, daily_levels: Dict, volume_profile: Dict,
                            liquidity: Dict, spread_pct: float = SPREAD_ROUND_TRIP_PCT) -> Dict:
    """Entry/stop/target levels + R:R INCLUDING THE SPREAD:
    a full round trip (entry+exit) costs spread_pct% of the price, deducted from the target's
    potential, not only from the displayed number — otherwise R:R would look better
    than the trade really is."""
    levels = {"entry_price": price, "stop_loss": None, "target_1": None, "target_2": None,
              "risk_reward_1": 0, "risk_reward_2": 0, "raw_risk_reward_1": 0, "raw_risk_reward_2": 0,
              "spread_cost_abs": round(price * spread_pct / 100, 8), "spread_cost_pct": spread_pct}

    vwap = daily_levels.get('vwap', price)
    pivot_reliable = daily_levels.get('pivot_reliable', True)
    s1 = daily_levels.get('s1', price * 0.98) if pivot_reliable else None
    s2 = daily_levels.get('s2', price * 0.96) if pivot_reliable else None
    r1 = daily_levels.get('r1', price * 1.02) if pivot_reliable else None
    r2 = daily_levels.get('r2', price * 1.04) if pivot_reliable else None
    # poc/prev_low/prev_high are no longer needed: the stop is volatility-based, not
    # structural. The volume_profile/liquidity arguments are kept in the signature —
    # callers pass them, and they will be useful if the rule changes again.
    atr = daily_levels.get('atr_hourly') or price * 0.015

    if direction in ["LONG", "STRONG_LONG"]:
        # The stop is volatility-based, NOT structural (see the table in the module header).
        # Structural levels (S1/POC/prev_low) sit too close to the price on
        # a contrarian entry and were knocked out by noise: 34% of trades closed
        # at the stop, although by the horizon 47% of them were in profit.
        stop_loss = price - atr * STOP_ATR_MULT
        levels["stop_loss"] = stop_loss

        # Targets closer than MIN_TARGET_ATR from the entry are discarded: when the price is
        # right at VWAP, the "target" ends up at the price itself (a SHORT
        # on FET was observed with a target 0.005% from the entry and R:R 1:0.0 — not a trade but an instant
        # close at zero, and a loss after the spread). This used to be caught by
        # the R:R filter, but it had to be turned off (see the module header), so
        # the cut-off is needed here.
        targets = [x for x in [vwap, r1, r2] if x and x - price >= atr * MIN_TARGET_ATR]
        if len(targets) >= 2:
            levels["target_1"], levels["target_2"] = targets[0], targets[1]
        elif len(targets) == 1:
            levels["target_1"], levels["target_2"] = targets[0], targets[0] * 1.02
        else:
            levels["target_1"], levels["target_2"] = price + atr * 2, price + atr * 4

    elif direction in ["SHORT", "STRONG_SHORT"]:
        # Symmetric to LONG — a volatility-based stop, see the comment above.
        stop_loss = price + atr * STOP_ATR_MULT
        levels["stop_loss"] = stop_loss

        # Symmetric to LONG — targets closer than MIN_TARGET_ATR are discarded.
        targets = [x for x in [vwap, s1, s2] if x and price - x >= atr * MIN_TARGET_ATR]
        if len(targets) >= 2:
            levels["target_1"], levels["target_2"] = targets[0], targets[1]
        elif len(targets) == 1:
            levels["target_1"], levels["target_2"] = targets[0], targets[0] * 0.98
        else:
            levels["target_1"], levels["target_2"] = price - atr * 2, price - atr * 4

    if levels["stop_loss"] is not None:
        levels["stop_loss"] = round(levels["stop_loss"], 8)
    if levels["target_1"] is not None:
        levels["target_1"] = round(levels["target_1"], 8)
    if levels["target_2"] is not None:
        levels["target_2"] = round(levels["target_2"], 8)

    if levels["stop_loss"] and levels["target_1"]:
        risk = abs(price - levels["stop_loss"])
        spread_cost = levels["spread_cost_abs"]

        def rr(target):
            raw_reward = abs(target - price) if direction in ["LONG", "STRONG_LONG"] else abs(price - target)
            eff_reward = raw_reward - spread_cost
            raw_rr = round(raw_reward / risk, 2) if risk > 0 else 0
            eff_rr = round(eff_reward / risk, 2) if risk > 0 else 0
            return raw_rr, eff_rr

        levels["raw_risk_reward_1"], levels["risk_reward_1"] = rr(levels["target_1"])
        if levels["target_2"]:
            levels["raw_risk_reward_2"], levels["risk_reward_2"] = rr(levels["target_2"])

    return levels


def determine_direction(scores: dict) -> tuple:
    """STRONG_* is tied to the maximum of ITS OWN side, not to a fixed
    offset from the threshold. With an offset of -3 after v6 (short maximum -4) STRONG_SHORT
    would require -5 and would be unreachable in principle — not by measurement but by
    arithmetic."""
    total = scores['total']
    if total >= LONG_THRESHOLD:
        return ("STRONG_LONG" if total >= MAX_TOTAL_SCORE - 1 else "LONG"), total
    if total <= SHORT_THRESHOLD:
        return ("STRONG_SHORT" if total <= -(MAX_SHORT_SCORE - 1) else "SHORT"), total
    return "NO_TRADE", total


def check_entry_filters(direction: str, daily_levels: Dict, trader_plan: Dict, indicators: Dict,
                         volume_ratio: Dict = None) -> Tuple[bool, str]:
    """Entry filters. The spread is checked FIRST and separately from the overall R:R —
    the spec requires exactly this rejection reason ("the move does not cover
    costs"), not a generic "R:R too low"."""
    if direction == "NO_TRADE":
        return False, "No signal"

    rr1 = trader_plan.get('risk_reward_1', 0)
    raw_rr1 = trader_plan.get('raw_risk_reward_1', 0)
    if raw_rr1 > 0 and rr1 <= 0:
        return False, (f"the move does not cover costs (a {trader_plan.get('spread_cost_pct')}% round-trip spread "
                        f"eats the whole potential to target 1)")

    if rr1 < MIN_ENTRY_RR:
        spread_note = f" after a {SPREAD_ROUND_TRIP_PCT}% spread" if SPREAD_ROUND_TRIP_PCT > 0 else ""
        return False, (f"R:R too low ({safe_format_float(rr1)}{spread_note}) — "
                        f"target closer than the stop")
    if rr1 > MAX_REASONABLE_RR:
        return False, f"R:R implausibly high ({safe_format_float(rr1)}) — target too far"

    vr = (volume_ratio or {}).get('volume_ratio', 1)
    if vr is not None and vr < 0.4:
        return False, f"Volume too low ({safe_format_float(vr)}x)"

    atr, price = daily_levels.get('atr_hourly', 0), indicators.get('price', 0)
    if atr and price:
        atr_pct = atr / price * 100
        if atr_pct < 0.2:
            return False, f"Volatility too low ({safe_format_float(atr_pct)}%)"
        if atr_pct > 8:
            return False, f"Volatility too high ({safe_format_float(atr_pct)}%)"

    return True, "OK"


def calculate_position_size(direction: str, total_score: int, entry_filters_ok: bool) -> int:
    if not entry_filters_ok:
        return 0
    strength = abs(total_score)
    if direction in ["STRONG_LONG", "STRONG_SHORT"]:
        # The same argument as in compute_confidence: the sides have different ceilings,
        # otherwise the maximum size would be unreachable for a short in principle.
        ceiling = MAX_SHORT_SCORE if total_score < 0 else MAX_TOTAL_SCORE
        return 4 if strength >= ceiling - 1 else 3
    if direction in ["LONG", "SHORT"]:
        return 2 if strength >= 5 else 1
    return 0


def compute_confidence(direction: str, total_score: int, entry_filters_ok: bool) -> float:
    if not entry_filters_ok:
        return 0.0
    # Normalise to the maximum of ITS OWN side: after v5 fewer factors play on the minus
    # side, and a common divisor would understate a short's confidence with the full
    # set of factors available to it (see the module header).
    ceiling = MAX_SHORT_SCORE if total_score < 0 else MAX_TOTAL_SCORE
    strength = min(abs(total_score), ceiling) / ceiling
    confidence = 0.5 + 0.4 * strength
    return round(max(0.1, min(0.9, confidence)), 2)


FACTOR_NAMES = {"orderflow": "Order Flow", "technical": "Technical", "stumpy": "STUMPY",
                "vwap": "VWAP", "funding": "Funding"}
# Bounds of EACH side separately, not one bound by absolute value.
#
# After v5 and v6 the ranges became ASYMMETRIC: technical lost all
# votes for a short (RSI>70, MACD bullish, BB upper — none was confirmed by
# measurement), and STUMPY was switched off entirely as noise. A symmetric absolute bound
# made the hints block suggest the impossible — "Technical: needs -2"
# although it has no negative votes any more, and "STUMPY: needs -2"
# for a disabled factor. Advice that cannot be followed is worse than silence.
FACTOR_BOUNDS = {
    "orderflow": (-2, 2),
    "technical": (0, 3),      # oversold side only, see v5
    "stumpy": (0, 0),         # disabled in v6
    "vwap": (-1, 1),
    "funding": (-1, 1),
}
FACTOR_MAX_ABS = {k: max(abs(lo), abs(hi)) for k, (lo, hi) in FACTOR_BOUNDS.items()}


def format_signals(scores: Dict, reasons: Dict) -> str:
    lines = ["📊 *SIGNALS (points):*"]
    for key, name in FACTOR_NAMES.items():
        if key not in scores:
            continue
        score, reason = scores.get(key, 0), reasons.get(key, "")
        emoji = "🟢" if score > 0 else "🔴" if score < 0 else "⚪"
        sign = f"{score:+d}" if score else "0"
        lines.append(f"   {emoji} *{name}*: {sign} — {reason}")
    lines.append("")
    lines.append(f"   🎯 *TOTAL*: **{scores['total']:+d}**")
    return "\n".join(lines)


def get_dominant_factor(scores: dict, reasons: dict) -> tuple:
    candidates = {k: v for k, v in scores.items() if k != "total"}
    if not candidates:
        return None, 0, ""
    total = scores.get("total", 0)
    worst_key = min(candidates, key=candidates.get) if total >= 0 else max(candidates, key=candidates.get)
    return FACTOR_NAMES.get(worst_key, worst_key), candidates[worst_key], reasons.get(worst_key, "")


def format_factor_bar(score: int, max_abs: int = None) -> str:
    max_abs = max_abs or 5
    ratio = min(1.0, abs(score) / max_abs) if max_abs else 0
    filled = max(1, int(ratio * 5))
    return "█" * filled + "░" * (5 - filled)


def generate_entry_conditions_detailed(scores: dict, reasons: dict, direction: str, threshold: int) -> str:
    if direction != "NO_TRADE":
        return ""
    total = scores.get("total", 0)
    lean_long = total >= 0
    needed = (threshold - total) if lean_long else (total - (-threshold))
    if needed <= 0:
        return "All factors need re-evaluation"

    options = []
    for key, name in FACTOR_NAMES.items():
        if key not in scores:
            continue
        lo, hi = FACTOR_BOUNDS.get(key, (-2, 2))
        score = scores.get(key, 0)
        target = score + needed if lean_long else score - needed
        feasible = (target <= hi) if lean_long else (target >= lo)
        if feasible:
            options.append(f"   • {name}: needs {target:+d} (now {score:+d})")

    label = "LONG" if lean_long else "SHORT"
    lines = [f"🔹 *Any one of the options below closes the gap of {needed} points to {label}:*"]
    lines.extend(options) if options else lines.append(
        f"   • no single factor covers {needed} points alone — a combination is needed")
    return "\n".join(lines)


def format_no_trade_report(scores: dict, reasons: dict, total_score: int, indicators: dict,
                            daily_levels: dict, volume_ratio: dict, threshold: int = 3,
                            filter_reason: str = "") -> str:
    # The "NO TRADE" header and the score are printed by the CALLER (see the CALCULATION block in
    # main.py::format_trade_report) — they must not be duplicated here, otherwise
    # the report shows "Decision: NO_TRADE (score +1)" followed by "❌ NO TRADE /
    # Score: +1 of ±9", i.e. the same thing in two different wordings.
    lines = []
    if abs(total_score) >= threshold:
        lines.append(f"The ±{threshold} threshold was met, but the trade was rejected by the entry filter")
        if filter_reason:
            lines.append(f"⛔ Reason: {escape_markdown(filter_reason)}")
        lines.append("")
    else:
        lines.append(f"{threshold - abs(total_score)} short of the ±{threshold} threshold "
                      f"(scale maximum ±{MAX_TOTAL_SCORE})\n")

    factor_name, factor_score, factor_reason = get_dominant_factor(scores, reasons)
    if factor_name:
        emoji = "🔴" if factor_score < 0 else "🟢" if factor_score > 0 else "⚪"
        lines.append("🎯 *Main reason:*")
        lines.append(f"   {emoji} {factor_name} ({factor_score:+d}) — {escape_markdown(factor_reason)}\n")

    sorted_items = sorted([(k, v) for k, v in scores.items() if k != "total"], key=lambda x: abs(x[1]), reverse=True)
    max_abs = max([abs(v) for k, v in scores.items() if k != "total"] + [1])
    for_f, against_f, neutral_f = [], [], []
    for key, value in sorted_items:
        name, reason = FACTOR_NAMES.get(key, key), escape_markdown(reasons.get(key, ""))
        bar = format_factor_bar(value, max_abs)
        line = f"   {'🟢' if value>0 else '🔴' if value<0 else '⚪'} {name:12} {bar}  ({value:+d}) — {reason}"
        (for_f if value > 0 else against_f if value < 0 else neutral_f).append(line)

    if for_f:
        lines.append("🟢 *For:*"); lines.extend(for_f); lines.append("")
    if against_f:
        lines.append("🔴 *Against:*"); lines.extend(against_f); lines.append("")
    if neutral_f:
        lines.append("⚪ *Neutral:*"); lines.extend(neutral_f); lines.append("")

    lines.append("💡 *What needs to change:*")
    lines.append(generate_entry_conditions_detailed(scores, reasons, "NO_TRADE", threshold))
    lines.append("")

    vr = (volume_ratio or {}).get('volume_ratio', 1)
    lines.append(f"📦 Volume: {safe_format_float(vr)}x ({(volume_ratio or {}).get('basis', '?')})")

    return "\n".join(lines)
