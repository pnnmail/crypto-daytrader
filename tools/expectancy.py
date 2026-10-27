# -*- coding: utf-8 -*-
"""CONDITIONAL EXPECTANCY: computes what the dataset was collected for.

Answers the question of spec v2 §20: "after states of the form <condition>, what happened
over the next N hours, and does anything remain after costs".

HOW THIS DIFFERS FROM A NAIVE COUNT
---------------------------------------
Naively: select points by the condition, count the share of winners and the mean move.
That must not be done, and in this project this was learned the hard way three times.

1. THE UNIT IS THE DAY, NOT THE POINT. Within a day coins move together. On one and
   the same sample t by point gave 0.95, and by day 2.57 — a threefold difference.
   Spec §23 proposes thresholds like "N>250 — strong evidence"; at our rate of 120
   points per day N=412 is 3.4 calendar days, where the critical value of t
   is 3.18, not 2. That is why DAILY BLOCKS are used everywhere here.

2. COMPARISON WITH THE BASE, NOT WITH ZERO. Absolute return in a rising market
   measures beta. The base is the same days, all points without the condition. The DIFFERENCE is measured.

3. EMPIRICAL THRESHOLD. Our pipeline gives |t|>2 on 25.8% of KNOWN-NULL
   features. So `calibrate` has a mode that runs random
   conditions of THE SAME support through THE SAME procedure and returns the 95th percentile
   of the best |t|. Compare against it, not against 1.96.

4. `complete: false` IS DISCARDED. A truncated window is indistinguishable from a full one,
   and a daily outcome computed from three hours of bars would enter the statistics
   as a full one.

5. A MISSING FIELD IS NOT ZERO. Points recorded before the dataset was extended do not
   have the new features. They are excluded from the condition's sample rather than counted
   as zeros: this mistake once produced 193 "rejections" out of 91 real points.

WHAT IS NOT HERE AND WHY
-----------------------
There is no search over conditions. The tool evaluates ONE given condition. Searching
is exactly the source of false findings, and if it is ever needed, it must come with
recomputing the threshold via `calibrate` with the same number of cells examined.

Run:
    python3 tools/expectancy.py describe
    python3 tools/expectancy.py run  --h 240 --where "rsi<30"
    python3 tools/expectancy.py calibrate --h 240 --support 0.2
"""
import argparse
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
POINTS = os.path.join(PROJ, "experiment_history.json")

HORIZONS = (15, 60, 240, 1440)

# Round-trip costs on the liquid top of Binance.US — measured, not made
# up: on Robinhood it would be 1.0-1.9%, and then nothing at all would be tradable.
COST_ROUND_TRIP_PCT = 0.1

MIN_BLOCKS = 5          # fewer — significance is not claimed at all
_T_CRIT_2 = {5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 12: 2.18,
             15: 2.13, 20: 2.09, 30: 2.04, 60: 2.00, 120: 1.98}


def t_crit(n_blocks):
    df = max(1, n_blocks - 1)
    for k in sorted(_T_CRIT_2):
        if df <= k:
            return _T_CRIT_2[k]
    return 1.96


def _flat(prefix, d, out):
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            _flat(f"{prefix}{k}_", v, out)
        elif isinstance(v, (int, float, bool)) or v is None:
            out[f"{prefix}{k}"] = v
        else:
            out[f"{prefix}{k}"] = v


def load_rows():
    """A flat table: one row per decision point.

    Context features are stored with flat names (market_rsi -> market_rsi,
    flow_flow_15m_delta_pct -> flow_flow_15m_delta_pct), outcomes separately.
    """
    with open(POINTS, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for coin, pts in data.items():
        for p in pts:
            r = {"coin": coin, "timestamp": p.get("timestamp", ""),
                 "day": (p.get("timestamp") or "")[:10],
                 "wave": (p.get("timestamp") or "")[:13],
                 "price": p.get("price")}
            ctx = p.get("context") or {}
            for k, v in ctx.items():
                if isinstance(v, dict):
                    _flat(f"{k}_", v, r)
                else:
                    r[k] = v
            out = p.get("outcomes") or {}
            for h in HORIZONS:
                o = out.get(f"h{h}")
                if o and o.get("complete"):
                    r[f"ret_{h}"] = o.get("ret_pct")
                    r[f"up_{h}"] = o.get("max_up_pct")
                    r[f"down_{h}"] = o.get("max_down_pct")
            rows.append(r)
    return rows


def _tstat(day_values):
    if len(day_values) < 2:
        return None
    sd = statistics.stdev(day_values)
    if sd == 0:
        return None
    return statistics.mean(day_values) / (sd / math.sqrt(len(day_values)))


def evaluate(rows, mask, horizon, side="LONG", costs=COST_ROUND_TRIP_PCT):
    """Conditional expectancy against the base at the same horizon.

    mask — a list of bool of the same length as rows.
    side — which side would be taken when the condition fires.
    """
    col = f"ret_{horizon}"
    sign = 1.0 if side == "LONG" else -1.0

    sel_by_day, all_by_day = defaultdict(list), defaultdict(list)
    for r, m in zip(rows, mask):
        v = r.get(col)
        if v is None:
            continue
        all_by_day[r["day"]].append(v * sign)
        if m:
            sel_by_day[r["day"]].append(v * sign)

    if not sel_by_day:
        return {"n": 0, "blocks": 0, "verdict": "no data"}

    sel_all = [v for vs in sel_by_day.values() for v in vs]
    base_all = [v for vs in all_by_day.values() for v in vs]

    # EXCESS OVER THE BASE BY DAY: each day takes the difference of means.
    # This way the market term cancels within a day, not between days.
    diffs = [statistics.mean(sel_by_day[d]) - statistics.mean(all_by_day[d])
             for d in sorted(sel_by_day) if all_by_day.get(d)]
    t = _tstat(diffs)
    blocks = len(diffs)

    wins = [v for v in sel_all if v > 0]
    losses = [v for v in sel_all if v <= 0]
    gross = statistics.mean(sel_all)
    res = {
        "n": len(sel_all),
        "blocks": blocks,
        "win_rate": round(100 * len(wins) / len(sel_all), 1),
        "baseline_win_rate": round(100 * sum(1 for v in base_all if v > 0)
                                   / len(base_all), 1) if base_all else None,
        "avg_win": round(statistics.mean(wins), 4) if wins else None,
        "avg_loss": round(statistics.mean(losses), 4) if losses else None,
        "gross_ev": round(gross, 4),
        "baseline_ev": round(statistics.mean(base_all), 4) if base_all else None,
        "edge": round(statistics.mean(diffs), 4) if diffs else None,
        "costs": costs,
        # NET expectancy is computed from the EXCESS, not from the gross
        # return: gross return in a rising market is beta, it must not be counted
        # as one's own merit, let alone traded.
        "net_ev": round(statistics.mean(diffs) - costs, 4) if diffs else None,
        "t": round(t, 2) if t is not None else None,
        "t_crit": t_crit(blocks),
    }
    if blocks < MIN_BLOCKS:
        res["verdict"] = f"NOT ENOUGH DATA: {blocks} blocks, need at least {MIN_BLOCKS}"
    elif t is None:
        res["verdict"] = "NOT ENOUGH DATA: zero spread"
    elif abs(t) <= res["t_crit"]:
        res["verdict"] = f"no edge (|t| {abs(t):.2f} <= {res['t_crit']:.2f})"
    elif res["net_ev"] is not None and res["net_ev"] <= 0:
        res["verdict"] = (f"significant but NOT TRADABLE: excess "
                          f"{res['edge']:+.3f}% with costs of {costs}%")
    else:
        res["verdict"] = "CANDIDATE (needs an empirical threshold and a freeze)"
    return res


def calibrate(rows, horizon, support=0.2, n_null=300, seed=20260911):
    """The empirical threshold of THIS procedure.

    Random conditions of the same support are run through the same evaluation. If
    the real condition gives |t| in the same range as the random ones, it is
    no better than a coin flip, whatever its formal significance.
    """
    rnd = random.Random(seed)
    n = len(rows)
    k = max(1, int(n * support))
    ts = []
    for _ in range(n_null):
        idx = set(rnd.sample(range(n), k))
        mask = [i in idx for i in range(n)]
        r = evaluate(rows, mask, horizon)
        if r.get("t") is not None:
            ts.append(abs(r["t"]))
    if not ts:
        return None
    ts.sort()
    return {
        "n_null": len(ts),
        "median": round(statistics.median(ts), 2),
        "p95": round(ts[int(len(ts) * 0.95)], 2),
        "max": round(max(ts), 2),
        "share_above_2": round(100 * sum(1 for v in ts if v > 2) / len(ts), 1),
    }


def make_mask(rows, where):
    """A condition like "rsi<30 and market_rvol20>2". Points where the needed field
    is MISSING give False — the absence of a feature is not zero."""
    if not where:
        return [True] * len(rows)
    mask = []
    for r in rows:
        env = {k: v for k, v in r.items() if v is not None}
        try:
            mask.append(bool(eval(where, {"__builtins__": {}}, env)))
        except Exception:
            mask.append(False)
    return mask


def cmd_describe(args):
    rows = load_rows()
    days = sorted(set(r["day"] for r in rows))
    print(f"rows: {len(rows)}, days: {len(days)} ({days[0]} .. {days[-1]}), "
          f"coins: {len(set(r['coin'] for r in rows))}")
    print()
    print("OUTCOME COVERAGE (complete only):")
    for h in HORIZONS:
        have = sum(1 for r in rows if r.get(f"ret_{h}") is not None)
        print(f"  ret_{h:<5} {have:5d} of {len(rows)}  ({100*have/len(rows):.0f}%)")
    print()
    print("FEATURES (how many rows have a value):")
    keys = defaultdict(int)
    for r in rows:
        for k, v in r.items():
            if v is not None and k not in ("coin", "timestamp", "day", "wave"):
                keys[k] += 1
    for k in sorted(keys):
        if k.startswith(("ret_", "up_", "down_")):
            continue
        print(f"  {k:<42} {keys[k]:5d}")


def cmd_run(args):
    rows = load_rows()
    mask = make_mask(rows, args.where)
    print(f"condition: {args.where or '(all points)'}")
    print(f"matching rows: {sum(mask)} of {len(rows)}")
    print()
    print(f"  {'hor.':<6} {'n':>5} {'days':>5} {'win%':>6} {'base':>6} "
          f"{'excess':>9} {'NET EV':>9} {'t':>7} {'thresh':>6}  verdict")
    for h in ([args.h] if args.h else HORIZONS):
        r = evaluate(rows, mask, h, side=args.side)
        if not r.get("n"):
            print(f"  {h:<6} no data")
            continue
        print(f"  {h:<6} {r['n']:>5} {r['blocks']:>5} {r['win_rate']:>5.1f}% "
              f"{r['baseline_win_rate']:>5.1f}% {r['edge']:>+8.3f}% "
              f"{r['net_ev']:>+8.3f}% {str(r['t']):>7} {r['t_crit']:>6.2f}  "
              f"{r['verdict']}")


def cmd_calibrate(args):
    rows = load_rows()
    h = args.h or 240
    print(f"procedure calibration: horizon {h} min, condition support "
          f"{args.support:.0%}, {args.n} nulls")
    c = calibrate(rows, h, support=args.support, n_null=args.n)
    if not c:
        print("  failed: not enough data")
        return
    print(f"  median |t| {c['median']}  95% {c['p95']}  max {c['max']}  "
          f"share |t|>2: {c['share_above_2']}%")
    print(f"  DISCOVERY THRESHOLD for this procedure: |t| > {c['p95']}")
    print("  The classical 1.96/2.0 do not apply here.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("describe")
    r = sub.add_parser("run")
    r.add_argument("--where", default=None)
    r.add_argument("--h", type=int, default=None)
    r.add_argument("--side", default="LONG", choices=["LONG", "SHORT"])
    c = sub.add_parser("calibrate")
    c.add_argument("--h", type=int, default=240)
    c.add_argument("--support", type=float, default=0.2)
    c.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    {"describe": cmd_describe, "run": cmd_run, "calibrate": cmd_calibrate}[a.cmd](a)
