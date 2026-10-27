"""
HYPOTHESIS PRE-REGISTRATION: freezing before data collection.

Why. The `book > plain` effect was found by SEARCHING — I looked at several pairs and
picked the one that looked best. Such a finding is not a test: with
nine deciders there are 36 pairs, and one of them will show "significance" by chance.
The only honest way to test it is to declare the hypothesis in advance and
measure it on data that DID NOT EXIST at the time of the declaration.

WHY NO MULTIPLE-COMPARISONS CORRECTION. A correction is needed when you test
many hypotheses at once. Here EXACTLY ONE is tested, named before collection, on
forward data. That is the point of freezing: it replaces the correction rather than
adding to it. If after freezing we looked at more pairs — then
the correction would come back, and the freeze file records this.

THREE SAFEGUARDS:
  1. eval takes ONLY points after the moment of freezing — the past on which
     the hypothesis was found physically cannot enter the test;
  2. the result is not shown until the required n is reached: peeking with
     stopping "when it looks nice" by itself inflates false findings;
  3. the freeze file is hashed, and eval fails if it was edited.

THE UNIT OF OBSERVATION IS THE DAY, not the point. Points within a day are correlated
(coins move together), so n is counted in days. The difference is huge: with the
same effect ~4600 observations are needed by point, ~7 by day. This is not fitting
a convenient unit but the same principle by which significance is computed everywhere
here.

Run:
  python3 tools/preregister.py freeze
  python3 tools/preregister.py eval
"""
import argparse
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
FREEZE_FILE = os.path.join(HERE, "preregistration.json")
# The hash in a SEPARATE file, not a field inside: self-hashing requires
# careful serialisation and breaks with any change in key
# order. A separate file is checked unambiguously.
HASH_FILE = os.path.join(HERE, "preregistration.sha256")

# The hypothesis is declared HERE, in code, before freeze is run.
HYPOTHESIS = {
    "name": "book_beats_plain",
    "statement": "The llm_book variant gives a HIGHER return per point than "
                 "llm_plain, on the same decision points.",
    "variant_a": "llm_book",
    "variant_b": "llm_plain",
    "direction": "a_greater",     # one-sided: the direction was declared in advance
    "alpha": 0.05,
    "power": 0.80,
    "units": ["day", "point"],
    "decision_rule": "The hypothesis counts as confirmed only if it passes "
                     "on BOTH units. On exploratory data they diverged "
                     "threefold (t=2.57 by day vs t=0.95 by point), and this "
                     "divergence by itself means instability: with "
                     "six days the daily sd is estimated from six numbers.",
    "why_one_sided": "The direction was named before the test and follows from the mechanism "
                     "(the book flips some shorts into longs). A two-sided "
                     "test would be weaker here for no reason.",
    "known_confound": "On exploratory data the whole effect came from "
                      "SHORT->LONG flips, and the period was rising. If the hypothesis "
                      "is confirmed only on rising days, it is not a confirmation.",
}


def load_points():
    with open(POINTS, encoding="utf-8") as f:
        data = json.load(f)
    return [p for pts in data.values() for p in pts if p.get("counterfactual")]


def decision_return(p, dec):
    o = p.get(dec) or {}
    if o.get("_error"):
        return None
    d = o.get("direction")
    if d not in ("LONG", "SHORT"):
        return 0.0
    return (p["counterfactual"].get(d) or {}).get("return_pct")


def paired_by_day(points, a, b, after=None):
    """Paired difference by day. after — the ISO string of the freeze moment."""
    by = defaultdict(list)
    for p in points:
        ts = p.get("timestamp", "")
        if after and ts <= after:
            continue
        ra, rb = decision_return(p, a), decision_return(p, b)
        if ra is None or rb is None:
            continue
        by[ts[:10]].append(ra - rb)
    return {d: statistics.mean(v) for d, v in sorted(by.items())}, \
           sum(len(v) for v in by.values())


def paired_flat(points, a, b, after=None):
    """A flat list of paired differences — the unit of observation is the POINT."""
    out = []
    for p in points:
        ts = p.get("timestamp", "")
        if after and ts <= after:
            continue
        ra, rb = decision_return(p, a), decision_return(p, b)
        if ra is None or rb is None:
            continue
        out.append(ra - rb)
    return out, len(out)


def required_days(effect, sd, alpha=0.05, power=0.80, one_sided=True):
    """How many daily blocks are needed. The z-approximation understates at small n,
    so a 2-day margin is added to the result."""
    if not effect or not sd:
        return None
    z_a = 1.645 if one_sided else 1.960
    z_b = 0.842                      # 80% power
    n = ((z_a + z_b) * sd / abs(effect)) ** 2
    return int(math.ceil(n)) + 2


def bootstrap_ci(xs, iters=10000, level=0.95, seed=1):
    """A bootstrap confidence interval. Does not rely on normality,
    which matters with 6 observations, where the t-approximation is shaky."""
    import random
    rnd = random.Random(seed)
    n = len(xs)
    means = sorted(sum(rnd.choice(xs) for _ in range(n)) / n for _ in range(iters))
    return (means[int((1 - level) / 2 * iters)],
            means[int((1 + level) / 2 * iters) - 1])


def t_crit(df, one_sided=True):
    table_1 = {1: 6.31, 2: 2.92, 3: 2.35, 4: 2.13, 5: 2.02, 6: 1.94, 7: 1.89,
               8: 1.86, 9: 1.83, 10: 1.81, 12: 1.78, 15: 1.75, 20: 1.72, 30: 1.70}
    table_2 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36,
               8: 2.31, 9: 2.26, 10: 2.23, 12: 2.18, 15: 2.13, 20: 2.09, 30: 2.04}
    tbl = table_1 if one_sided else table_2
    for k in sorted(tbl):
        if df <= k:
            return tbl[k]
    return 1.65 if one_sided else 1.96


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def cmd_freeze(_args):
    if os.path.exists(FREEZE_FILE):
        sys.exit(f"A freeze already exists: {FREEZE_FILE}\n"
                 f"It must not be rewritten — that is the whole point. Delete it by hand "
                 f"if you are really starting over.")
    points = load_points()
    a, b = HYPOTHESIS["variant_a"], HYPOTHESIS["variant_b"]
    days, n_pts = paired_by_day(points, a, b)
    if len(days) < 2:
        sys.exit("Too little data to estimate the required n.")
    vals = list(days.values())
    eff, sd = statistics.mean(vals), statistics.stdev(vals)
    need = required_days(eff, sd, HYPOTHESIS["alpha"], HYPOTHESIS["power"])

    rec = dict(HYPOTHESIS)
    rec.update({
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pilot": {"days": len(days), "points": n_pts,
                  "effect_per_day": round(eff, 4), "sd_per_day": round(sd, 4)},
        "required_days_forward": need,
        "note": "The pilot figures serve ONLY to compute the required n. They are "
                "not part of the test: eval takes points strictly after frozen_at.",
    })
    rec["rules"] = [
        "ONLY points with a time strictly after frozen_at count.",
        "Look at the result only after required_days_forward is reached.",
        "The hypothesis is not reworded, the direction is not flipped.",
        "Failure = failure. A negative result does NOT turn into "
        "'more data needed'.",
        "Confirmation only if it passes on BOTH units (day and point).",
    ]
    with open(FREEZE_FILE, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    rec_hash = file_hash(FREEZE_FILE)
    with open(HASH_FILE, "w", encoding="utf-8") as f:
        f.write(rec_hash + chr(10))

    print("HYPOTHESIS FROZEN")
    print(f"  {rec['statement']}")
    print(f"  freeze moment: {rec['frozen_at']}")
    print(f"  pilot (NOT part of the test): {len(days)} d., effect "
          f"{eff:+.4f}%, sd {sd:.4f}")
    print(f"  forward days needed: {need}")
    print(f"  hash: {rec_hash}")
    print()
    print("  A known confound to watch for when analysing:")
    print(f"  {rec['known_confound']}")


def cmd_eval(args):
    if not os.path.exists(FREEZE_FILE):
        sys.exit("No freeze. First: python3 tools/preregister.py freeze")
    rec = json.load(open(FREEZE_FILE, encoding="utf-8"))
    if not os.path.exists(HASH_FILE):
        sys.exit("No hash file — the freeze cannot be verified. The test "
                 "is invalid.")
    stored = open(HASH_FILE, encoding="utf-8").read().strip()
    if file_hash(FREEZE_FILE) != stored:
        sys.exit("THE FREEZE FILE WAS EDITED after creation — the hash does not match. "
                 "The test is invalid.")

    points = load_points()
    a, b = rec["variant_a"], rec["variant_b"]
    days, n_pts = paired_by_day(points, a, b, after=rec["frozen_at"])
    need = rec["required_days_forward"]

    print(f"HYPOTHESIS: {rec['statement']}")
    print(f"frozen {rec['frozen_at']}, hash {stored}")
    print(f"forward data: {len(days)} d., {n_pts} points "
          f"(need {need} d.)")

    if len(days) < need and not args.force:
        print()
        print(f"RESULT NOT SHOWN: {len(days)} of {need} days collected.")
        print("Peeking with stopping 'when it looks nice' itself creates false")
        print("findings — that is why the number is hidden until n is reached.")
        print("If you really need to look: --force (and that is no longer")
        print("pre-registration but exploration, label it as such).")
        return

    vals = list(days.values())
    if len(vals) < 2:
        sys.exit("Not enough days to compute.")

    # Computed on BOTH units. They rest on different assumptions:
    # the day does not require independence of points within a day, the point does not require
    # estimating sd from a handful of days. On exploration they diverged threefold, and
    # agreement between them is stronger evidence than either one
    # alone.
    flat, _ = paired_flat(load_points(), a, b, after=rec["frozen_at"])

    print()
    print(f"  {'unit':>8} {'n':>6} {'effect':>10} {'sd':>9} {'t':>7} "
          f"{'threshold':>9}  verdict")
    verdicts = {}
    for label, xs in (("day", vals), ("point", flat)):
        if len(xs) < 2:
            print(f"  {label:>8} too little data")
            verdicts[label] = False
            continue
        eff, sd = statistics.mean(xs), statistics.stdev(xs)
        t = eff / (sd / math.sqrt(len(xs))) if sd else 0.0
        crit = t_crit(len(xs) - 1, one_sided=(rec["direction"] == "a_greater"))
        ok = t > crit and eff > 0
        verdicts[label] = ok
        print(f"  {label:>8} {len(xs):>6} {eff:>+9.4f}% {sd:>8.4f} {t:>+7.2f} "
              f"{crit:>7.2f}  {'passed' if ok else 'NOT passed'}")

    lo, hi = bootstrap_ci(vals)
    print(f"  bootstrap interval by day: [{lo:+.4f}%, {hi:+.4f}%]")
    print(f"  days positive: {sum(1 for v in vals if v > 0)} of {len(vals)}")
    print()
    if all(verdicts.values()):
        print("  HYPOTHESIS CONFIRMED on both units.")
        print(f"  Before accepting: {rec['known_confound']}")
    elif any(verdicts.values()):
        print("  PARTIAL: the units DIVERGE. Under the freeze rule this is NOT")
        print("  a confirmation — divergence means the result rests")
        print("  on the assumptions of one counting method rather than on the data.")
    else:
        print("  HYPOTHESIS NOT CONFIRMED. Close it.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    e = sub.add_parser("eval")
    e.add_argument("--force", action="store_true",
                   help="show the result before n is reached (exploration, does NOT count)")
    args = ap.parse_args()
    (cmd_freeze if args.cmd == "freeze" else cmd_eval)(args)
