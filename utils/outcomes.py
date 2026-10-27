# -*- coding: utf-8 -*-
"""Multi-horizon outcomes and excursions (MFE/MAE).

WHY. Until now a point had ONE outcome: what a trade would give after 4 hours with
pre-set levels. That is not enough for two things that cannot be
collected after the fact:

  * at which horizon the effect lives at all, if there is one. One horizon
    answers the question only about itself;
  * what the stop and target should be. Now they come from ATR, i.e. out
    of thin air. To choose them from data, one must know how far the price
    WENT against the position before moving in the right direction.

WHY EXCURSIONS ARE STORED WITHOUT A SIDE. MFE (maximum favourable move) and MAE
(maximum adverse move) depend on whether it is LONG or SHORT. Storing them
per side means duplicating the same number with different signs. So
two neutral quantities are written here — how far the price went UP and
how far DOWN from entry — and MFE/MAE are derived from them:

    LONG :  MFE = up,   MAE = down
    SHORT:  MFE = down, MAE = up

WHY THIS IS NOT THE SAME AS counterfactual. That one computes a trade BY LEVELS: with
a stop, targets and the rule "if the stop and target hit in the same minute,
the stop wins". Here there are no levels at all — only what the price did.
They must not be mixed: one measures the execution of a rule, the other the market itself.
The `counterfactual` field stays untouched — a frozen
hypothesis rests on it, and changing its meaning in the middle of a test is unacceptable.

INCOMPLETE WINDOWS ARE MARKED. If there were not enough minute bars for the whole horizon,
the outcome is marked `complete: false`. Without this mark a truncated window
is indistinguishable from a real one, and a 24-hour outcome computed from three hours
of data would silently enter the statistics as a full one.
"""
from datetime import timedelta

# Horizons in minutes. The set is declared IN ADVANCE and must not change along the way:
# adding a horizon after looking at results is a search over
# cells, and the significance threshold must then be recalibrated.
HORIZONS_MIN = (15, 60, 240, 1440)

# The share of the horizon the bars must cover for the outcome to count as complete.
_COVERAGE = 0.95


def forward_outcomes(df, entry_dt, entry_px, horizons_min=HORIZONS_MIN):
    """What the price did after entry, without any levels.

    df        — minute bars with High/Low/Close columns and a time index
    entry_dt  — the moment of the point
    entry_px  — entry price

    Returns {"h15": {...}, "h60": {...}, ...}, where each horizon has:
        ret_pct       return at the end of the window (close of the last bar)
        max_up_pct    maximum High above entry, %
        max_down_pct  minimum Low below entry, % (<= 0)
        bars          how many minute bars fell into the window
        complete      whether there were enough bars for the whole horizon
    """
    if df is None or df.empty or not entry_px:
        return {}
    out = {}
    for h in horizons_min:
        end = entry_dt + timedelta(minutes=h)
        win = df[(df.index > entry_dt) & (df.index <= end)]
        if win.empty:
            continue
        try:
            hi = float(win["High"].max())
            lo = float(win["Low"].min())
            close = float(win["Close"].iloc[-1])
        except (TypeError, ValueError, KeyError):
            continue
        out[f"h{h}"] = {
            "ret_pct": round((close - entry_px) / entry_px * 100, 4),
            "max_up_pct": round((hi - entry_px) / entry_px * 100, 4),
            "max_down_pct": round((lo - entry_px) / entry_px * 100, 4),
            "bars": int(len(win)),
            # Bars can have gaps, so completeness is measured by coverage
            # of TIME, not by number of bars: the last bar of the window
            # is compared with the required end of the horizon.
            "complete": bool(
                (win.index[-1] - entry_dt).total_seconds() >= _COVERAGE * h * 60
            ),
        }
    return out


def mfe_mae(outcome, side):
    """MFE and MAE for a side from the neutral excursions.

    Returns (mfe_pct, mae_pct), both as absolute moves: MFE >= 0 — how much
    the price gave in favour, MAE >= 0 — how much it took against. If there is no data,
    returns (None, None).
    """
    if not outcome:
        return None, None
    up = outcome.get("max_up_pct")
    down = outcome.get("max_down_pct")
    if up is None or down is None:
        return None, None
    if side == "LONG":
        return max(up, 0.0), abs(min(down, 0.0))
    if side == "SHORT":
        return abs(min(down, 0.0)), max(up, 0.0)
    return None, None


def max_horizon_min(horizons_min=HORIZONS_MIN):
    return max(horizons_min)
