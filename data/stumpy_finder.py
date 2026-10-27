"""
STUMPY Finder — searching for similar historical patterns (stumpy.match: query vs
history, NOT a self-join). Ported from the stock version almost unchanged —
the only substantive difference is WHERE the history comes from: there
the "current window" (with the prepost flag) and the "history" (with the same
flag, so bar durations matched) were loaded separately. Here the market is continuous, so
both the query and the history come from ONE candle request — bar desynchronisation
is impossible in principle.

The main research question: on continuous 24/7 data the sample
(n) of historical analogues should be an order of magnitude larger than n=5 for stocks,
where the Wilson confidence interval came out as [37%-96%] and meant nothing.
"""
import numpy as np
import pandas as pd
import stumpy
from scipy import stats

from data.fetcher import fetch_candles

# 24 hourly bars per day — the market never closes, unlike stocks
# (7 in the regular session / 17 with pre-market and after-hours there).
BARS_PER_DAY = 24


def wilson_score_interval(wins, n, confidence=0.95):
    """
    The Wilson confidence interval for a proportion of successes. Robust at small n
    (unlike the normal approximation). Returns (lower, upper) as
    fractions (0..1). Ported literally from the stock version — a general-purpose
    function with no dependence on a specific market.
    """
    if n == 0:
        return 0.0, 1.0

    z = stats.norm.ppf(1 - (1 - confidence) / 2)  # 1.96 for 95%
    p = wins / n

    denominator = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denominator

    lower = centre - margin
    upper = centre + margin

    return max(0.0, lower), min(1.0, upper)


def fmt(value, format_str="+.2f", fallback="n/a"):
    if value is None:
        return fallback
    return f"{value:{format_str}}"


def compute_base_rate(close_prices: np.ndarray, timestamps, days: int = 3, step: int = 6) -> tuple:
    """
    THE COIN'S BASE RATE: the share of ANY history windows after which the price
    `days` days later was higher, plus the average return over the same period.

    Needed to tell a signal from drift. The 2026-08-11 measurement showed that in
    crypto the base rate is NOT 50% and is specific to each coin (BTC 48.5%, ETH 49.5%,
    SOL 49.3%, ENA 42.5%). While the pattern's win_rate was compared with a fixed
    50%, the factor for ENA always gave a minus — it measured the market's fall over the year, not
    the pattern's behaviour.

    step=6 (every 6th hour) — thinning: the base rate changes slowly,
    computing it on every bar makes no sense, and a full pass is noticeably more expensive.
    """
    n = len(close_prices)
    if n < 50:
        return None, 0.0
    idx = np.arange(0, n, step)
    target_times = timestamps[idx] + pd.Timedelta(days=days)
    # searchsorted instead of a mask on each iteration: O(n log n) instead of O(n^2)
    future_pos = np.searchsorted(timestamps.values, target_times.values, side="left")
    valid = future_pos < n
    if not valid.any():
        return None, 0.0
    start = close_prices[idx[valid]]
    end = close_prices[future_pos[valid]]
    rets = (end - start) / start * 100
    return round(float((rets > 0).mean() * 100), 1), round(float(rets.mean()), 3)


def find_similar_patterns(coin: str, window: int = 20, top_k: int = 100, max_history_days: int = 400,
                           df_hourly: pd.DataFrame = None) -> dict:
    """
    Finds segments in the coin's history similar to the last `window` hourly
    candles and computes the forward return (1d/3d/5d) for each match.

    top_k=100 is not a stock legacy (there it was 20 because no more data
    accumulated). A measurement on BTC (400 days of history) showed: top_k=20 -> n=20,
    a CI about 40 pp wide; top_k=150 -> n=149, CI [39.1%-55.0%], only 16 pp.
    This is the advantage of continuous data — 100
    was chosen as a compromise between CI width and search time.

    CORRECTION (measured 2026-08-16): this used to say "~10-15 s/coin" —
    that is wrong, that is how long the ONE-OFF numba JIT compilation takes on the first
    call in a process. Each subsequent run on 400 days of history costs
    0.02 s. The bot lives long, so only the first time is paid.

    IMPORTANT: since v6 the result is NOT PART OF THE DECISION and is not fed into prompts —
    measured as noise (r=+0.00 at 4h, +0.02 at 3d; the "factor is silent" group
    beat both opinionated groups). It is computed for reference in the report and so
    it can be re-measured later; see utils/signals.py.

    df_hourly can be passed in pre-loaded (if the calling code already
    got it for indicators) — saves a repeated request to the exchange, but
    it MUST cover max_history_days, otherwise it is reloaded.
    """
    try:
        if df_hourly is None or len(df_hourly) < window * 2 or \
                (df_hourly.index[-1] - df_hourly.index[0]).days < max_history_days - 2:
            df_hourly = fetch_candles(coin, "1h", days_back=max_history_days)

        if df_hourly.empty or len(df_hourly) < window * 2:
            return None

        close_prices = df_hourly['Close'].values.astype(float)
        timestamps = df_hourly.index

        query = close_prices[-window:]

        # Exclude the current window and a buffer around it — otherwise we would find itself
        # (distance ~0) or trivially adjacent autocorrelated segments.
        cutoff = window * 2
        history_for_search = close_prices[:-cutoff] if len(close_prices) > cutoff else close_prices[:0]
        history_timestamps = timestamps[:-cutoff] if len(timestamps) > cutoff else timestamps[:0]

        if len(history_for_search) < window:
            return None

        # stumpy's default max_distance = mean(D) - 2*std(D) — too strict,
        # almost nothing passes (the same conclusion as in the stock version).
        # Relaxed to 1.5 sigma.
        matches = stumpy.match(
            query, history_for_search,
            max_matches=top_k,
            max_distance=lambda D: max(np.nanmean(D) - 1.5 * np.nanstd(D), np.nanmin(D)),
        )

        if matches is None or len(matches) == 0:
            return None

        def price_after(match_end_idx, days):
            """The price `days` calendar days later by the real timestamp.
            Crypto has no gaps, but a timestamp is still
            more reliable than a fixed offset by number of bars."""
            target_time = history_timestamps[match_end_idx] + pd.Timedelta(days=days)
            future_mask = timestamps >= target_time
            future_slice = close_prices[future_mask]
            return float(future_slice[0]) if len(future_slice) > 0 else None

        results = []
        for distance, idx in matches:
            idx = int(idx)
            match_end_idx = idx + window - 1
            current_match_end = float(history_for_search[match_end_idx])

            future_1d = price_after(match_end_idx, 1)
            future_3d = price_after(match_end_idx, 3)
            future_5d = price_after(match_end_idx, 5)

            results.append({
                'index': idx,
                'distance': float(distance),
                'ret_1d': round((future_1d - current_match_end) / current_match_end * 100, 2) if future_1d else None,
                'ret_3d': round((future_3d - current_match_end) / current_match_end * 100, 2) if future_3d else None,
                'ret_5d': round((future_5d - current_match_end) / current_match_end * 100, 2) if future_5d else None,
            })

        valid_1d = [r['ret_1d'] for r in results if r['ret_1d'] is not None]
        valid_3d = [r['ret_3d'] for r in results if r['ret_3d'] is not None]
        valid_5d = [r['ret_5d'] for r in results if r['ret_5d'] is not None]

        base_win_rate, base_avg_return = compute_base_rate(close_prices, timestamps, days=3)

        summary = {
            'coin': coin,
            'window': window,
            'matches_found': len(results),
            # The coin's base rate — the pattern's win_rate is compared with it
            # in utils/signals.py, otherwise the factor measures market drift (see there).
            'base_win_rate': base_win_rate,
            'base_avg_return': base_avg_return,
            'n_1d': len(valid_1d), 'n_3d': len(valid_3d), 'n_5d': len(valid_5d),
            'avg_ret_1d': round(np.mean(valid_1d), 2) if valid_1d else None,
            'avg_ret_3d': round(np.mean(valid_3d), 2) if valid_3d else None,
            'avg_ret_5d': round(np.mean(valid_5d), 2) if valid_5d else None,
            'win_rate_1d': round(sum(1 for r in valid_1d if r > 0) / len(valid_1d) * 100, 1) if valid_1d else None,
            'win_rate_3d': round(sum(1 for r in valid_3d if r > 0) / len(valid_3d) * 100, 1) if valid_3d else None,
            'win_rate_5d': round(sum(1 for r in valid_5d if r > 0) / len(valid_5d) * 100, 1) if valid_5d else None,
            'details': results[:3],
        }

        for horizon, data in [('1d', valid_1d), ('3d', valid_3d), ('5d', valid_5d)]:
            if data:
                wins = sum(1 for r in data if r > 0)
                n = len(data)
                lower, upper = wilson_score_interval(wins, n)
                summary[f'ci_lower_{horizon}'] = round(lower * 100, 1)
                summary[f'ci_upper_{horizon}'] = round(upper * 100, 1)

        for horizon, data in [('1d', valid_1d), ('3d', valid_3d), ('5d', valid_5d)]:
            if data:
                summary[f'max_ret_{horizon}'] = round(max(data), 2)
                summary[f'min_ret_{horizon}'] = round(min(data), 2)

        return summary

    except Exception as e:
        print(f"   ⚠️ STUMPY error ({coin}): {str(e)[:200]}")
        return None


def format_stumpy_result(result: dict) -> str:
    if not result:
        return ""

    lines = [f"\n📊 *STUMPY — HISTORICAL ANALOGUES ({result['coin']})*",
             f"• Similar patterns found: {result['matches_found']}",
             f"• Analysis window: {result['window']} hourly candles "
             f"(~{result['window'] / BARS_PER_DAY:.1f} days)", ""]

    for horizon, label in [('1d', '1 day'), ('3d', '3 days'), ('5d', '5 days')]:
        avg = result.get(f'avg_ret_{horizon}')
        if avg is None:
            continue
        n = result.get(f'n_{horizon}', '?')
        ci_low = result.get(f'ci_lower_{horizon}', '?')
        ci_high = result.get(f'ci_upper_{horizon}', '?')
        win_rate = result.get(f'win_rate_{horizon}')
        lines.append(f"• After {label} (n={n}):")
        lines.append(f"    Average result: {fmt(avg)}%")
        lines.append(f"    Win rate: {win_rate}% (95% CI: {ci_low}-{ci_high}%)")
        lines.append(f"    Max: {fmt(result.get(f'max_ret_{horizon}'))}%, "
                      f"Min: {fmt(result.get(f'min_ret_{horizon}'))}%")
        lines.append("")

    if result.get('details'):
        lines.append("📋 *Match details:*")
        for i, match in enumerate(result['details']):
            lines.append(f"  [{i + 1}] distance={fmt(match['distance'], '.2f')} | "
                          f"1d: {fmt(match['ret_1d'])}% | 3d: {fmt(match['ret_3d'])}% | "
                          f"5d: {fmt(match['ret_5d'])}%")

    if result.get('win_rate_3d') is not None:
        ci_low = result.get('ci_lower_3d', 0)
        ci_high = result.get('ci_upper_3d', 100)
        n = result.get('n_3d', 0)
        win = result.get('win_rate_3d')
        # Must be compared with THE COIN'S BASE RATE, not 50%: in crypto it is
        # below 50 and specific to each coin (BTC 48.5%, ENA 42.5%). In scoring
        # this was fixed in v3, but here the old comparison with 50% remained —
        # and the text went into the LLM prompt, i.e. the model read "46% = bad",
        # although with a base of 47% that is normal. For the "calculation vs
        # model" experiment such desynchronisation made the comparison unfair:
        # the formula saw the corrected figure, the model — a misleading one.
        base = result.get('base_win_rate')
        base_ret = result.get('base_avg_return')
        ref = base if base is not None else 50.0

        lines.append(f"\n📖 n={n} matches | win rate {win}% vs the coin's base rate "
                      f"{ref}% | 95% CI [{ci_low}%-{ci_high}%]")
        if base is not None:
            lines.append(f"   Base rate — the share of ANY windows of this coin with a rise over 3 days "
                          f"(average return {fmt(base_ret)}%). A signal is a DEVIATION "
                          f"from it, not a difference from 50%.")
            lines.append(f"   Pattern excess over base: {win - ref:+.1f} pp")

        if ci_low > ref:
            lines.append(f"💡 *Conclusion:* A strong bullish signal — even the lower CI bound is above the base ({ref}%).")
        elif ci_high < ref:
            lines.append(f"⚠️ *Conclusion:* A strong bearish signal — even the upper CI bound is below the base ({ref}%).")
        elif n < 10:
            lines.append(f"🤷 *Conclusion:* Statistically undetermined (n={n}, the interval is wide).")
        else:
            lines.append(f"🤷 *Conclusion:* Direction undetermined, the interval covers the base ({ref}%).")

    return "\n".join(lines)
