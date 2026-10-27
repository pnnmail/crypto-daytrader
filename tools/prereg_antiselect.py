# -*- coding: utf-8 -*-
"""FREEZING THE ANTI-SELECTION HYPOTHESIS.

Found 2026-09-03 on 14 days: Russian-language LLM variants within one coin
systematically pick THE WRONG side. Estimator — the covariance of side (+1 LONG /
-1 SHORT) and the coin's excess return in its wave, with means subtracted PER
COIN, significance by daily blocks.

WHY THIS MUST BE FROZEN RATHER THAN USED
----------------------------------------------
1. The finding is post hoc: first the table of 9 deciders was looked at, then an estimator
   was fitted to it. Exactly the selection that freezing exists to remove.
2. Five "confirming" variants agree with each other 73-94% of the time. That is ONE
   finding, not five independent ones. Their significances cannot be added up.
3. The size is ~0.15% per trade against costs of ~0.1% per round trip. The margin is so
   small that an error in the cost estimate eats the whole effect.
4. Reversing a known-losing rule is a classic trap: the side
   flips, and with it the beta composition, and on a falling window this
   gives an illusory improvement. So ONLY the estimator in which
   the market cancels out (within-coin covariance) counts, not the return of the reversal.

THE EMPIRICAL THRESHOLD of this procedure was measured by permuting days: |t| > 2.01
(300 nulls, median 0.72, max 2.93). The classical threshold nearly coincided here,
but one must rely on the measured one: for Donchian the same procedure gave |t|=3.70
on a quantity that is zero by construction.

Run:  python3 tools/prereg_antiselect.py freeze | eval
"""
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
POINTS = os.path.join(PROJ, "experiment_history.json")
FREEZE = os.path.join(HERE, "prereg_antiselect.json")
HASHF = os.path.join(HERE, "prereg_antiselect.sha256")

VARIANT = "llm_plain"
DISCOVERY_THRESHOLD = 2.01

HYPOTHESIS = {
    "name": "llm_anti_selection",
    "statement": "Within one coin, the side chosen by llm_plain is "
                 "NEGATIVELY related to that coin's excess return "
                 "in its wave.",
    "estimator": "cov((side - side_mean_by_coin), (excess - excess_mean_by_coin)), "
                 "t by daily blocks",
    "excess": "the coin's LONG-counterfactual return MINUS the mean LONG counterfactual "
              "of all coins in the same wave (hour)",
    "direction": "negative",
    "alpha": 0.05,
    "one_sided": True,
    "unit": "day",
    "primary_variant": VARIANT,
    "corroborating_variants": ["llm_book", "llm_book2", "llm_book3", "llm_book4"],
    "corroboration_note": "These variants agree with llm_plain 73-94% of the time and are NOT "
                          "independent tests. They are listed only "
                          "to record that llm_plain counts as the confirmation, "
                          "the rest is description, not evidence.",
    "discovery_threshold": DISCOVERY_THRESHOLD,
    "discovery_note": "Measured by permuting days, 300 nulls, the same procedure.",
    "known_confounds": [
        "No single context factor (bb_position, rsi, vwap_dev, "
        "macd_hist, taker_imbalance, volume_ratio) reached the threshold, i.e. "
        "the effect does not reduce to a simple factor. But the set of factors is not complete.",
        "The discovery window was falling (mean LONG -0.10% per point). If in "
        "the forward window the effect shows up only on falling days — that is not "
        "confirmation but market direction.",
        "The model's confidence is nearly constant (0.78 in 88% of cases), so "
        "DECIDER_CONFIDENCE_THRESHOLD filters nothing and the variant enters "
        "at almost every point. Changing the model or the prompt will change this and "
        "make the forward points incomparable.",
    ],
    "rules": [
        "ONLY points strictly after frozen_at count.",
        "Look at the result only after required_days_forward is reached.",
        "llm_plain counts as the confirmation. The other variants do not enter "
        "the decision — they are correlated with it.",
        "The direction is not flipped, the estimator is not changed.",
        "Confirmation does NOT mean trading the reversal: the size is comparable "
        "to costs, that is a separate question.",
        "Failure = failure.",
    ],
}


def load():
    d = json.load(open(POINTS, encoding="utf-8"))
    return [p for v in d.values() for p in v if p.get("counterfactual")]


def cov_by_day(points, variant, after=None):
    """Daily values of the within-coin covariance of side and excess return.

    The wave benchmark is computed over the WHOLE history, not just the forward part:
    it is a characteristic of the market at that hour, not an estimated quantity.
    """
    wave = defaultdict(list)
    for p in points:
        r = (p["counterfactual"].get("LONG") or {}).get("return_pct")
        if r is not None:
            wave[p["timestamp"][:13]].append(r)
    bench = {w: statistics.mean(v) for w, v in wave.items()}

    rec = []
    for p in points:
        if after and p.get("timestamp", "") <= after:
            continue
        o = p.get(variant)
        if not isinstance(o, dict) or o.get("_error"):
            continue
        s = o.get("direction")
        if s not in ("LONG", "SHORT"):
            continue
        r = (p["counterfactual"].get("LONG") or {}).get("return_pct")
        b = bench.get(p["timestamp"][:13])
        if r is None or b is None:
            continue
        rec.append((p["coin"], p["timestamp"][:10],
                    1 if s == "LONG" else -1, r - b))
    if len(rec) < 50:
        return {}, 0
    coins = set(c for c, _, _, _ in rec)
    ms = {c: statistics.mean(s for cc, _, s, _ in rec if cc == c) for c in coins}
    me = {c: statistics.mean(e for cc, _, _, e in rec if cc == c) for c in coins}
    by = defaultdict(list)
    for c, day, s, e in rec:
        by[day].append((s - ms[c]) * (e - me[c]))
    return {d: statistics.mean(v) for d, v in sorted(by.items())}, len(rec)


def t_of(vals):
    if len(vals) < 2:
        return None
    sd = statistics.stdev(vals)
    return statistics.mean(vals) / (sd / math.sqrt(len(vals))) if sd else None


T1 = {1: 6.31, 2: 2.92, 3: 2.35, 4: 2.13, 5: 2.02, 6: 1.94, 7: 1.89, 8: 1.86,
      9: 1.83, 10: 1.81, 12: 1.78, 15: 1.75, 20: 1.72, 30: 1.70}


def tcrit(nb):
    df = max(1, nb - 1)
    for k in sorted(T1):
        if df <= k:
            return T1[k]
    return 1.65


def need_days(eff, sd):
    if not eff or not sd:
        return None
    return int(math.ceil(((1.645 + 0.842) * sd / abs(eff)) ** 2)) + 2


def fhash(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    pts = load()

    if cmd == "freeze":
        if os.path.exists(FREEZE):
            sys.exit("A freeze already exists: %s. It must not be rewritten." % FREEZE)
        days, n = cov_by_day(pts, VARIANT)
        vals = list(days.values())
        eff, sd = statistics.mean(vals), statistics.stdev(vals)
        rec = dict(HYPOTHESIS)
        rec.update({
            "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pilot": {"days": len(vals), "points": n,
                      "effect_per_day": round(eff, 5), "sd_per_day": round(sd, 5),
                      "t": round(t_of(vals), 2)},
            "required_days_forward": need_days(eff, sd),
            "note": "The pilot figures serve ONLY to compute the required n. "
                    "They are not part of the test: eval takes points strictly after "
                    "frozen_at.",
        })
        json.dump(rec, open(FREEZE, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        open(HASHF, "w", encoding="utf-8").write(fhash(FREEZE) + chr(10))
        print("HYPOTHESIS FROZEN: %s" % rec["name"])
        print("  moment: %s, hash %s" % (rec["frozen_at"], fhash(FREEZE)))
        print("  pilot: %d d., %d points, effect %+.5f, sd %.5f, t %+.2f"
              % (len(vals), n, eff, sd, rec["pilot"]["t"]))
        print("  forward days needed: %s" % rec["required_days_forward"])
        print("  Until then the result is NOT looked at.")
        sys.exit(0)

    if not os.path.exists(FREEZE):
        sys.exit("No freeze. First: python3 tools/prereg_antiselect.py freeze")
    rec = json.load(open(FREEZE, encoding="utf-8"))
    want = open(HASHF, encoding="utf-8").read().strip()
    if fhash(FREEZE) != want:
        sys.exit("THE FREEZE FILE WAS CHANGED (%s != %s). The test is invalid."
                 % (fhash(FREEZE), want))
    print("HYPOTHESIS: %s" % rec["statement"])
    print("frozen %s, hash %s" % (rec["frozen_at"], want))
    days, n = cov_by_day(pts, VARIANT, after=rec["frozen_at"])
    need = rec["required_days_forward"]
    print("forward data: %d d., %d points (need %s d.)" % (len(days), n, need))
    if len(days) < need:
        print("  Too early. The result is not shown: peeking with stopping "
              "when it looks nice itself inflates false findings.")
        sys.exit(0)
    vals = list(days.values())
    t = t_of(vals)
    tc = tcrit(len(vals))
    ok = t is not None and t < -tc
    print("  effect %+.5f, sd %.5f, t %+.2f, threshold -%.2f over %d days"
          % (statistics.mean(vals), statistics.stdev(vals), t, tc, len(vals)))
    print("  days in the expected direction: %d of %d"
          % (sum(1 for v in vals if v < 0), len(vals)))
    print()
    print("  HYPOTHESIS CONFIRMED." if ok
          else "  HYPOTHESIS NOT CONFIRMED. Close it.")
