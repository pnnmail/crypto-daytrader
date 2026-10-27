"""
Downloading the Binance microstructure archive for one coin.

WHY. All our previous measurements were made on features from CANDLES (RSI, MACD,
VWAP, channels), and none gave an edge in returns. Order flow is
a fundamentally different source: who was the aggressor, was the flow absorbed,
does depth recover. It had never been tested here because there was no
history. It turned out there is — free and for years.

WHAT IS DOWNLOADED (data.binance.vision, a public archive without a geo-block):
  aggTrades    spot, every trade with the aggressor side (~23 MB/day)
  bookDepth    futures, depth at +-1..5% from mid, a snapshot ~every 25 sec
  metrics      futures, open interest and position ratios, every 5 min
  fundingRate  futures, funding rates, monthly

WHY FUTURES DATA WHEN TRADING SPOT. Perpetuals are not available to us
(Binance.US and Robinhood are spot only), but open interest, funding and
positioning are FEATURES, not instruments. Derivatives positioning
may well predict spot, and checking it costs nothing.

THE VENUE SPLIT, which must not be forgotten: data from Binance.com, trading on
Coinbase/Binance.US. It was measured that hourly returns of the venues match
(correlation 0.9997 for ETH, max price divergence 0.17%), so transferring
the signal is legitimate. But the OUTCOME must still be measured on the venue where
you trade, otherwise the wrong thing is tested.

RAW DATA IS NOT DELETED. The features one will want to compute in a month are
unknown now, and re-downloading a month of tape takes hours. Disk is cheaper than time.
"""
import argparse
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta

BASE = "https://data.binance.vision/data"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(os.path.dirname(HERE), "microstructure")

# (dataset, path template, file name template, period)
DATASETS = [
    ("aggTrades", "spot/daily/aggTrades/{sym}", "{sym}-aggTrades-{d}.zip", "daily"),
    ("bookDepth", "futures/um/daily/bookDepth/{sym}", "{sym}-bookDepth-{d}.zip", "daily"),
    ("metrics", "futures/um/daily/metrics/{sym}", "{sym}-metrics-{d}.zip", "daily"),
    ("fundingRate", "futures/um/monthly/fundingRate/{sym}",
     "{sym}-fundingRate-{m}.zip", "monthly"),
]


def download(url, dest, retries=3):
    """Downloads with checksum verification if the archive has one.
    An already existing file is not re-downloaded — the run can be interrupted."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "exists"
    for attempt in range(retries):
        try:
            tmp = dest + ".part"
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            # The checksum lies next to it as .CHECKSUM. If it is missing —
            # no problem, but if it is there, a corrupted file must not be silently accepted:
            # it would surface a month later in the middle of a computation.
            try:
                with urllib.request.urlopen(url + ".CHECKSUM", timeout=30) as r:
                    want = r.read().decode().split()[0]
                got = hashlib.sha256(open(tmp, "rb").read()).hexdigest()
                if want != got:
                    os.remove(tmp)
                    if attempt == retries - 1:
                        return "bad checksum"
                    continue
            except urllib.error.HTTPError:
                pass
            os.replace(tmp, dest)
            return "downloaded"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "not on server"
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return "error"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--end", default=None, help="last date YYYY-MM-DD")
    args = ap.parse_args()

    sym = args.symbol.upper()
    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    days = [end - timedelta(days=i) for i in range(args.days)]
    months = sorted({d.strftime("%Y-%m") for d in days})

    total_bytes = 0
    for name, path_t, file_t, period in DATASETS:
        out_dir = os.path.join(OUT_ROOT, sym, name)
        os.makedirs(out_dir, exist_ok=True)
        keys = months if period == "monthly" else [d.isoformat() for d in sorted(days)]
        ok = miss = 0
        for k in keys:
            fname = file_t.format(sym=sym, d=k, m=k)
            url = f"{BASE}/{path_t.format(sym=sym)}/{fname}"
            dest = os.path.join(out_dir, fname)
            status = download(url, dest)
            if status in ("downloaded", "exists"):
                ok += 1
                total_bytes += os.path.getsize(dest)
            else:
                miss += 1
                print(f"   {fname}: {status}", flush=True)
            if status == "downloaded":
                time.sleep(0.2)
        print(f"{name:12} done {ok}, skipped {miss}", flush=True)

    print(f"TOTAL on disk: {total_bytes/1048576:.0f} MB in {OUT_ROOT}/{sym}")


if __name__ == "__main__":
    main()
