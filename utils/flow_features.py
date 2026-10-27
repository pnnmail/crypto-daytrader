# -*- coding: utf-8 -*-
"""Order flow and liquidity at the moment of decision.

WHY A SEPARATE LAYER. Until now the whole flow was reduced to a single number,
`taker_imbalance`. That is both thin in content and — worse — incomparable between
coins, see below.

THE MAIN DECISION: FIXED TIME WINDOWS, NOT A NUMBER OF TRADES
--------------------------------------------------------------
Measured 2026-09-11: the last THOUSAND trades cover

    BTC    3.5 minutes
    ETH    6.9 minutes
    SOL   12.1 minutes
    ADA   28.3 minutes
    AERO 100.8 minutes

I.e. the feature "imbalance over the last thousand trades" describes three
minutes for BTC and an hour and a half for AERO. Such numbers cannot be put in one table
and compared between coins: they will differ by liquidity, not by
flow. This is a hidden defect of the existing `taker_imbalance`, and there is no need
to repeat it here.

So the windows are set IN MINUTES, and the tape is paged back with the `after` cursor
until the window is covered. If it could not be covered (the page limit was hit),
the window is marked `complete: false` — exactly as with the multi-horizon outcomes.
Silently computing a 15-minute feature from three minutes is not allowed.

VOLUMES IN THE QUOTE CURRENCY. Coinbase's size is in the base coin, and 10 units
of BTC are not the same as 10 units of AERO. Everything is multiplied by price so that the numbers
are in dollars and comparable between coins.

THE ORDER BOOK IS THE MOST UNRECOVERABLE. An instantaneous depth snapshot is provided neither by
Coinbase nor by archives: not captured now — lost forever. That is why it is here,
even though it costs a separate request per coin.

NONE OF THIS TAKES PART IN THE DECISION. It is a state description for after-the-fact
analysis; utils/signals.py still decides.
"""
import statistics

# Windows in minutes. Declared IN ADVANCE; adding more after looking at results
# is a search over cells, and the significance threshold is then recalibrated.
FLOW_WINDOWS_MIN = (1, 5, 15)

# Order book depth bands around the mid, in percent.
DEPTH_BANDS_PCT = (0.1, 0.5, 1.0)

_MAX_PAGES = 6          # 6 pages of 1000 trades are enough for 15 min even for BTC
_COVERAGE = 0.9         # which share of the window the trades must cover


def fetch_trades_window(coin, minutes, max_pages=_MAX_PAGES):
    """The tape over the last `minutes` minutes, paged back with a cursor.

    Returns (list of trades, covered seconds). A trade is a dict with fields
    time/side/size/price, as the exchange returns it.
    """
    from datetime import timedelta
    import pandas as pd
    from data.fetcher import _get, product_id

    pid = product_id(coin)
    rows, after = [], None
    newest = oldest = None
    for _ in range(max_pages):
        params = {"limit": 1000}
        if after is not None:
            params["after"] = after
        page = _get(f"/products/{pid}/trades", params=params)
        if not page:
            break
        rows.extend(page)
        ts = pd.to_datetime([r["time"] for r in page], utc=True, format="mixed")
        if newest is None:
            newest = ts.max()
        oldest = ts.min()
        if (newest - oldest) >= timedelta(minutes=minutes):
            break
        after = page[-1].get("trade_id")
        if after is None:
            break
    if not rows:
        return [], 0.0
    cutoff = newest - timedelta(minutes=minutes)
    kept = []
    for r in rows:
        t = pd.Timestamp(r["time"])
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        if t >= cutoff:
            r = dict(r)
            r["_t"] = t
            kept.append(r)
    covered = (newest - oldest).total_seconds()
    return kept, covered


def _split(trades):
    """Buys and sells in the quote currency."""
    buy = sell = 0.0
    for r in trades:
        try:
            q = float(r["size"]) * float(r["price"])
        except (TypeError, ValueError, KeyError):
            continue
        if r.get("side") == "buy":
            buy += q
        else:
            sell += q
    return buy, sell


def _window_stats(trades, minutes, covered_sec):
    buy, sell = _split(trades)
    total = buy + sell
    out = {
        "n_trades": len(trades),
        "buy_quote": round(buy, 2),
        "sell_quote": round(sell, 2),
        "delta_quote": round(buy - sell, 2),
        "delta_pct": round((buy - sell) / total, 4) if total else None,
        # Window coverage: without it a 15-minute feature computed from three
        # minutes is indistinguishable from an honest one.
        "covered_sec": round(covered_sec, 1),
        "complete": bool(covered_sec >= _COVERAGE * minutes * 60),
    }
    # Imbalance of LARGE trades separately: small fry and a large player move price
    # differently, and in the overall imbalance they are mixed.
    sizes = []
    for r in trades:
        try:
            sizes.append(float(r["size"]) * float(r["price"]))
        except (TypeError, ValueError, KeyError):
            pass
    if len(sizes) >= 20:
        thr = sorted(sizes)[int(len(sizes) * 0.9)]
        big = [r for r in trades
               if (float(r.get("size", 0)) * float(r.get("price", 0))) >= thr]
        b2, s2 = _split(big)
        t2 = b2 + s2
        out["large_delta_pct"] = round((b2 - s2) / t2, 4) if t2 else None
        out["large_share"] = round(t2 / (buy + sell), 4) if (buy + sell) else None
    return out


def _cvd_slope(trades, minutes):
    """Slope of the cumulative delta over minute buckets within the window.

    The delta itself answers "who prevailed over the window", the slope — "did
    pressure grow towards the end of the window or fade". These are different things: the same
    final delta can come from an accelerating flow and from a fading one.
    """
    if len(trades) < 10:
        return None
    buckets = {}
    for r in trades:
        t = r.get("_t")
        if t is None:
            continue
        key = int(t.timestamp() // 60)
        try:
            q = float(r["size"]) * float(r["price"])
        except (TypeError, ValueError, KeyError):
            continue
        buckets[key] = buckets.get(key, 0.0) + (q if r.get("side") == "buy" else -q)
    if len(buckets) < 3:
        return None
    keys = sorted(buckets)
    cvd, acc = [], 0.0
    for k in keys:
        acc += buckets[k]
        cvd.append(acc)
    n = len(cvd)
    xs = list(range(n))
    mx, my = statistics.mean(xs), statistics.mean(cvd)
    den = sum((x - mx) ** 2 for x in xs)
    if not den:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, cvd)) / den
    # Normalise by the average turnover per minute: otherwise the slope is measured in dollars and
    # is incomparable between coins of different size.
    scale = statistics.mean(abs(v) for v in buckets.values()) or 1.0
    return round(slope / scale, 4)


def book_features(coin):
    """Spread, depth by bands and order book imbalance."""
    from data.fetcher import fetch_order_book
    ob = fetch_order_book(coin)
    bids, asks = ob.get("bids") or [], ob.get("asks") or []
    if not bids or not asks:
        return {}
    best_bid = max(b[0] for b in bids[:5]) if bids else None
    best_ask = min(a[0] for a in asks[:5]) if asks else None
    if not best_bid or not best_ask:
        return {}
    mid = (best_bid + best_ask) / 2
    out = {"spread_pct": round((best_ask - best_bid) / mid * 100, 5)}
    for band in DEPTH_BANDS_PCT:
        lo, hi = mid * (1 - band / 100), mid * (1 + band / 100)
        bd = sum(p * s for p, s in bids if p >= lo)
        ad = sum(p * s for p, s in asks if p <= hi)
        tag = str(band).replace(".", "_")
        out[f"bid_depth_{tag}"] = round(bd, 2)
        out[f"ask_depth_{tag}"] = round(ad, 2)
        if bd + ad:
            out[f"book_imbalance_{tag}"] = round((bd - ad) / (bd + ad), 4)
    return out


def build_flow_features(coin):
    """The full block. Errors are not propagated: the features are a description, the
    loop must not crash because of them."""
    feats = {}
    # The longest window is fetched once, the short ones are cut out of it —
    # otherwise one coin would take three page-backs instead of one.
    try:
        longest = max(FLOW_WINDOWS_MIN)
        trades, covered = fetch_trades_window(coin, longest)
    except Exception:
        trades, covered = [], 0.0
    if trades:
        import pandas as pd
        newest = max(r["_t"] for r in trades)
        for m in FLOW_WINDOWS_MIN:
            cutoff = newest - pd.Timedelta(minutes=m)
            sub = [r for r in trades if r["_t"] >= cutoff]
            if not sub:
                continue
            feats[f"flow_{m}m"] = _window_stats(sub, m, min(covered, m * 60.0))
        slope = _cvd_slope(trades, longest)
        if slope is not None:
            feats["cvd_slope"] = slope
    try:
        book = book_features(coin)
        if book:
            feats["book"] = book
    except Exception:
        pass
    return feats
