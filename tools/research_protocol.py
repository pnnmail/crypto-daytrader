"""
RESEARCH PROTOCOL: two thresholds, structured nulls, placebo.

Came out of the 2026-08-28 measurement, which showed that our pipeline gives |t|>2
on 25-38% of KNOWN-NULL features. I.e. the habitual "t>2 — candidate" means
nothing in this task.

TWO THRESHOLDS THAT MUST NOT BE MIXED
------------------------------------
DISCOVERY  — for exploration, where many cells are examined and the best one is taken. The threshold
             is empirical: the 95th percentile of the best |t| over nulls run
             through THE SAME selection procedure. It is tied to a specific procedure
             (number of features, horizons, windows, estimator) and does NOT transfer to
             another task.
CONFIRMATION — for a single pre-registered test, where there is no choice.
             Ordinary statistics applies here: the critical t for the df.

Mixing them means either strangling a real finding with an excessive threshold at
confirmation, or taking noise for a finding during exploration.

WHY NULLS OF DIFFERENT KINDS
------------------------------
White noise is too easy a control. Measured: permuting the DAYS of a real
feature gives MORE false findings than white noise (38% vs 25%), because
it preserves autocorrelation, heavy tails and intraday structure, breaking
only the link to the future. Test against the worst control, not the convenient one.

PLACEBO HORIZON
----------------
The feature is run against the PAST with the same lag. A real predictive
link must not work on the past. If it works equally well — it is a structural
artefact, and formal significance changes nothing.
"""
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE = os.path.join(HERE, "null_calibrations.json")

# Critical t, one-sided/two-sided — for CONFIRMATION, not exploration.
_T_CRIT_2 = {5: 2.57, 10: 2.23, 15: 2.13, 20: 2.09, 30: 2.04, 60: 2.00, 120: 1.98}


def confirmation_threshold(n_blocks: int) -> float:
    """The ordinary critical t by number of blocks. Applicable ONLY when the hypothesis
    was declared in advance and there is exactly one cell."""
    df = max(1, n_blocks - 1)
    for k in sorted(_T_CRIT_2):
        if df <= k:
            return _T_CRIT_2[k]
    return 1.96


def tstat_by_block(pairs):
    """t by blocks (usually days). Observations within a block are correlated,
    so the block is the unit, not the observation."""
    by = defaultdict(list)
    for k, v in pairs:
        by[k].append(v)
    m = [statistics.mean(v) for v in by.values()]
    if len(m) < 2:
        return None, len(m)
    sd = statistics.stdev(m)
    if sd == 0:
        return None, len(m)
    return statistics.mean(m) / (sd / math.sqrt(len(m))), len(m)


def make_null(kind, feature, rng):
    """A null of the given kind built from a real feature.

    Each one breaks the link to the future but preserves a different amount of internal
    structure. The more it preserves, the stricter the control.
    """
    idx, vals = feature.index, feature.values
    n = len(idx)
    if kind == "random":
        return pd.Series(rng.standard_normal(n), index=idx)
    if kind == "ar1":
        e = rng.standard_normal(n)
        x = np.empty(n)
        x[0] = e[0]
        for i in range(1, n):
            x[i] = 0.98 * x[i - 1] + e[i]
        return pd.Series(x, index=idx)
    if kind == "block_perm":
        # The most honest control: whole days are permuted.
        days = sorted(set(idx.date))
        order = list(days)
        rng.shuffle(order)
        parts = []
        for src, dst in zip(order, days):
            block = vals[idx.date == src]
            tgt = idx[idx.date == dst]
            m = min(len(block), len(tgt))
            parts.append(pd.Series(block[:m], index=tgt[:m]))
        return pd.concat(parts).sort_index()
    if kind == "reversed":
        return pd.Series(vals[::-1], index=idx)
    raise ValueError(kind)


NULL_KINDS = ("random", "ar1", "block_perm", "reversed")


def save_calibration(name, params, null_ts, real_ts):
    """The calibration is VERSIONED together with the procedure's parameters.

    Without parameters the number is useless: the threshold 3.11 is correct for 36 cells and this
    estimator, while with a different number of features or horizons the distribution
    of nulls is different. Half a year later one must be able to say "passed calibration v2",
    not "we computed something".
    """
    vals = sorted(null_ts)
    rec = {
        "name": name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "procedure": params,
        "n_nulls": len(vals),
        "null_median": round(statistics.median(vals), 3),
        "null_p90": round(vals[int(len(vals) * 0.90)], 3),
        "null_p95": round(vals[int(len(vals) * 0.95)], 3),
        "null_p99": round(vals[min(len(vals) - 1, int(len(vals) * 0.99))], 3),
        "null_max": round(max(vals), 3),
        "share_above_2": round(100 * sum(1 for v in vals if v > 2) / len(vals), 1),
        "discovery_threshold": round(vals[int(len(vals) * 0.95)], 3),
        "real_features": {k: round(v, 3) for k, v in real_ts.items()},
        "note": "discovery_threshold applies ONLY to this selection procedure. "
                "For a pre-registered test use "
                "confirmation_threshold(n_blocks).",
    }
    all_recs = []
    if os.path.exists(CALIB_FILE):
        all_recs = json.load(open(CALIB_FILE, encoding="utf-8"))
    rec["version"] = f"v{len(all_recs) + 1}"
    all_recs.append(rec)
    json.dump(all_recs, open(CALIB_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return rec

# ──────────────────────────────────────────────────────────────────────────
# CHECKING THE TARGET FOR INVARIANCE — BEFORE searching for predictors
# ──────────────────────────────────────────────────────────────────────────
# Appeared after 2026-08-29: the target log(RV_future/RV_past) contained RV_past
# in the denominator, and the control feature rv_past gave t=9.57 purely by algebra.
# Without the control, any feature correlated with the level of
# volatility would have looked like a finding.
#
# ADVERSARIAL CONTROLS are more dangerous than random noise. Noise answers the question
# "will the pipeline find a spurious correlation". An algebraic control answers
# another: "will the researcher notice a link they built into the task themselves".
# The second is the error that cannot be caught by significance.
#
# RULE: if even one control gives |t| above the discovery threshold, the research
# STOPS right there. Not "we'll take it into account when interpreting", but stop: the target
# is redefined, and everything is recomputed.

def target_invariance(target, controls, evaluator, threshold, label=""):
    """Runs algebraic controls against the target.

    target     — the target Series
    controls   — {name: Series}, features that MUST NOT predict
                 a correctly specified target: components of the target itself, its lags,
                 level, obvious algebraic derivatives
    evaluator  — callable(feature, target) -> |t| by the same procedure that
                 will be applied to real features
    threshold  — the discovery threshold of THIS procedure (see save_calibration)

    Returns (ok, results). ok=False means STOP.
    """
    # "Could not compute" is NOT EQUAL to "clean". Found on ourselves 2026-08-29: with
    # fine stratification no data was left in the cells, the estimator returned
    # the initialising 0.0, and the check took it for the absence of leakage.
    # A false pass here is more dangerous than a false alarm: it opens the way
    # to searching for features on a procedure that has no statistical power.
    res, unusable = {}, []
    for name, series in controls.items():
        t = evaluator(series, target)
        if t is None:
            unusable.append(name)
        else:
            res[name] = t
    if unusable:
        print(f"COULD NOT EVALUATE controls: {', '.join(unusable)}")
        print("  This does NOT mean 'no leakage' — it means not enough data. STOP.")
        return False, res
    worst = max(res.items(), key=lambda kv: kv[1]) if res else (None, float("inf"))
    ok = bool(res) and worst[1] <= threshold
    print(f"TARGET INVARIANCE{(' — ' + label) if label else ''}")
    print(f"  discovery threshold: |t| > {threshold:.2f}")
    for name, t in sorted(res.items(), key=lambda kv: -kv[1]):
        mark = "LEAKAGE" if t > threshold else "ok"
        print(f"    {name:<22} |t| = {t:6.2f}   {mark}")
    if not ok:
        print(f"  STOP: control '{worst[0]}' gives |t|={worst[1]:.2f} above the threshold.")
        print("  The target is specified incorrectly. We do NOT proceed to searching for predictors.")
    else:
        print("  The target passed the check — predictors may be searched for.")
    return ok, res
