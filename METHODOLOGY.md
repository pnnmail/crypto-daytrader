# Methodology and measurement rules

This is a research project: the goal is to measure, not to make money. So the main thing
here is not speed but making sure the measurements do not lie.

## The main rule: full sample first, conclusions second

**Before celebrating a result, extend the sample to everything that actually
traded in that period — including what died.**

Over the project this mechanism fired THREE times in a row, and each time the result
first looked convincing:

| finding | how it looked | what killed it |
|---|---|---|
| cross-sectional momentum | excess over the market +0.4%, t=3.6 | lived only in illiquid coins: on tradable ones t=0.4 |
| Donchian on daily bars | +115% vs +46% for buy-and-hold | computed on 24 survivors; on the 374-coin universe −30.6% |
| trend-strength filter | crash zone −1.71%, t=−2.39 | it was THREE calendar days, the sign flipped between windows |

The difference from sample completeness can be larger than the whole measured effect: for
Donchian the median buy-and-hold was +47% on the Robinhood list and −79% on the
full universe — a 126 pp difference from coin composition ALONE.

In practice:
* the coin list from the bot's config is TODAY's survivors; it is unusable for
  history;
* Coinbase `/products` also returns delisted pairs, and candles EXIST for them
  up to the delisting date — the full universe is available, there is no excuse;
* a coin must be present in the sample for exactly as long as it traded.

## Metrics

* **THE MAIN NUMBER is return RELATIVE TO THE MARKET**, not absolute.
  Absolute P&L in a rising market measures beta: a decider that goes long
  more often scores points not for decision quality. Checked directly — for
  Donchian, which is always LONG, the excess over the market is exactly +0.000%,
  i.e. it is pure beta and nothing else. Benchmark per point = the mean outcome
  of a hypothetical LONG across all coins of the same wave.
* Abstaining also has a price relative to the market: sitting out a rise is a relative
  loss, sitting out a fall is a gain.
* A PAIRED comparison of variants subtracts beta automatically: the market term
  cancels if both sides are measured on the same points.
* **Excess over the market REQUIRES A COIN-FLIP CONTROL.** The wave benchmark
  is built from LONG counterfactuals, i.e. it is LONG-biased. In a falling window anyone
  who sometimes shorts will beat it mechanically, and the whole table goes positive.
  Measured 2026-09-03 (10 days, 5 of them falling): a 50/50 coin flip gave +0.151%
  over the market and beat 8 of 9 deciders, "always SHORT" gave +0.233% and beat
  all of them. A "vs market" row without these two controls is unreadable.
* **Measure timing skill by permuting the decider's OWN decisions within a day.**
  This preserves the LONG/SHORT share and the number of abstentions and breaks only the link
  "this coin at this moment". That separates timing from holding a
  side — and the side alone explained the whole table of excesses.
* **Hit rate and return are DIFFERENT things.** They diverged in
  opposite directions (overbought: rises more often, but falls are larger).
  Return is what gets traded.
* **Measure return as excess over the market on the same day.** Otherwise in a
  trending period you measure drift, not signal.
* **Compound vs compound.** The sum of trade returns
  cannot be compared with buy-and-hold. This was violated right in `/experiment`:
  the "total" row printed `sum(rets)`, i.e. identically
  "number of points × mean per point". plain came out at **−77.61%**, plain_en at
  **+63.74%** — it read like a trading result. Positions run in PARALLEL
  (median 7 coins per wave, 4 h horizon, ~28 open at once), while the
  sum added them up as sequential, and on top of that grew with coverage —
  directly against the "rows are incomparable" warning. Replaced with a portfolio:
  equal weights within a non-overlapping window as long as the horizon, compounding
  between windows. The same data gave **−5.02%** and **+3.30%**.
  Metric check: for Donchian (LONG 99.7% of the time) the portfolio is −4.53% vs
  −4.43% for plain "hold everything LONG" — a 0.1 pp discrepancy.
* **An effect and its t must be computed in ONE unit.** "VS MARKET" was printed
  as a mean over POINTS, and the t next to it over DAILY means. With days of different size
  the signs diverge: plain_en had +0.094% by points vs −0.038% by days with
  t=−0.16, and the row looked like a calculation error. Now both are printed.
* **The median across coins is not a portfolio.** A portfolio needs capital
  allocation rules, and the result is usually noticeably worse.

## Significance: TWO THRESHOLDS, never mix them

The 2026-08-28 measurement (`tools/null_calibrations.json`, version v1) showed that our
pipeline gives |t|>2 on **25.8% of KNOWN-NULL** features. The habitual
"t>2 — candidate" means nothing in this task.

* **DISCOVERY** — exploration, where many cells are examined and the best is taken. The threshold
  is empirical: the 95th percentile of the best |t| over nulls run through THE SAME
  procedure. For the minute-level flow pipeline it is **|t| > 3.11**.
  The threshold is tied to the procedure (5 features × 3 horizons × 2 windows = 36 cells)
  and does NOT transfer to another task. Change the feature set, horizons,
  universe or estimator — recalibrate and version it.
* **CONFIRMATION** — a single pre-registered test, where there is no choice.
  Here the ordinary critical t by number of blocks applies (`confirmation_threshold`).
  Freezing exists precisely to remove selection.

The strictest null is **permuting the DAYS of a real feature**: it gives
MORE false findings than white noise (38% vs 25%), because it preserves
autocorrelation, heavy tails and intraday structure, breaking only the link
to the future. Test against the worst control.

**A PLACEBO HORIZON is mandatory.** The same feature against the PAST with the same
lag. If it explains the past as well as the future, it is a structural artefact.
The measurement caught exactly that: `imbalance` gave t=8.31 on the past vs 1.60 on
the future, because aggressive trades move the price within the same minute.
No other test showed this.

## Other significance rules

* **A t on a near-zero quantity is garbage.** For Donchian, which is always LONG,
  the within-wave rank relation is zero BY CONSTRUCTION, yet the procedure gave
  |t| = 3.70: mean ~1e-17, spread even smaller. So for each procedure the
  threshold comes from measuring nulls, not from a table. For the within-coin
  covariance the measurement gave |t| > 2.01 (300 day permutations).
* **Agreement between variants must be measured.** LLM variants agree on direction
  73–94% of the time (except regime — 34–44%). Five "confirmations" from such variants
  are ONE finding, not five; their significances cannot be added up.
* Significance only by **daily blocks**: within a day coins move together,
  the naive t is inflated several-fold.
* The |t|>2 threshold is valid only with many blocks. With 2 blocks the critical
  value is 12.7. Fewer than 5 blocks — no significance is claimed at all.
* Observation step = horizon, otherwise observations overlap.
* Training and test windows are **separated in time**; a random split for
  a time series is meaningless.

## The decider experiment

* Variants run on the SAME point simultaneously and are compared **pairwise** —
  this cancels the common market move and increases power by an order of magnitude.
* The model does NOT see the computed decision, otherwise it would copy it.
* A model error is an error, not NO_TRADE.
* A change of `PROMPT_VERSION` does not always mean incomparability: if the prompts
  of old variants did not change, their points should be MERGED. See the comment
  next to the version.

## Already measured — no need to ask again

* There is no directional edge in returns at any horizon from 4
  hours to 14 days — with or without a stop, for large or small coins.
* STUMPY has no predictive power (r=0.00), removed in v6.
* Oversold predicts a rise, overbought a fall — NO.
* Cost breakeven: ~0.1% per round trip on the liquid top of
  Binance.US, not 1.0–1.9% as on Robinhood.
* `book > plain` — CLOSED 2026-09-03. The pre-registered test on
  8 forward days: t=1.76 by days with a threshold of 1.89, and t=0.89 by points with
  1.65. It passed on neither unit. A book in the prompt gives nothing.
* The model's `confidence` is nearly constant: 0.78 in 88% of answers, 0.68 in another 10%.
  So `DECIDER_CONFIDENCE_THRESHOLD = 0.6` filters out NOTHING, and the variants
  enter at almost every point. The intended "filtering out weak opinions in code"
  does not work; LLM variants practically never abstain.

## Dataset: what is written per point and what it is NOT

Extended 2026-09-11 per spec v2. The decision logic was NOT touched — only a
description layer was added, because it cannot be collected after the fact.

* `counterfactual` — a trade BY LEVELS over 4 hours, with a stop and targets. Its meaning
  and format are FROZEN: a pre-registered hypothesis rests on this field
  (`tools/prereg_antiselect.py`). It must not change until unblinding.
* `outcomes` — what the PRICE did, without any levels: horizons 15m/1h/4h/24h,
  each with `ret_pct`, `max_up_pct`, `max_down_pct`, `bars`, `complete`.
  These are DIFFERENT things from the counterfactual: the first measures rule execution, the second
  the market itself. They must not be mixed in one table.
* Excursions are stored WITHOUT a side (`max_up`/`max_down`); MFE and MAE are derived
  from them via `utils.outcomes.mfe_mae`. Otherwise one number would be duplicated
  with different signs for LONG and SHORT.
* **`complete: false` must be read.** A truncated window is indistinguishable from a
  full one, and a daily outcome computed from three hours of bars would silently enter
  the statistics as a full one.
* `context.market` — trend (distances and slopes of EMA20/50/200), volatility
  (ATR%, realised, percentile), volume (RVOL20/50, z-score, percentile),
  BTC/ETH regime and RELATIVE strength (coin return minus BTC/ETH over the same
  window). The seven old context fields remain at the top level unchanged —
  analyses were written on them.
* Rolling statistics are computed on CLOSED bars: the last forming bar
  is dropped. Its close is known and gives no look-ahead, but High,
  Low and Volume are incomplete, and RVOL on a partial bar is systematically understated.
  One rule for both live computation and history — otherwise the feature would not match
  itself.
* A missing `market` or `outcomes` means "the point was recorded before the extension",
  NOT zero. The same mistake already cost an analysis: a missing variant was counted
  as an abstention, and plain_en came out with 193 "abstentions" out of 91 real points.
* `context.flow` — flow and order book. Windows of **1m / 5m / 15m by TIME**, not by
  number of trades: the 2026-09-11 measurement showed that the last thousand trades
  cover 3.5 minutes for BTC and 100.8 minutes for AERO. The feature "imbalance over
  a thousand trades" is incomparable between coins — it would differ by
  liquidity, not by flow. This is a hidden defect of the old `taker_imbalance`,
  which stays at the top level only for compatibility.
  The tape is paged back with the `after` cursor; each window has `covered_sec` and
  `complete`. Volumes are in the QUOTE currency (size × price): 10 units of BTC and
  10 units of AERO are different money.
* `context.flow.book` — spread, depth in ±0.1/0.5/1% bands and order book
  imbalance. **The most unrecoverable of all**: an instantaneous depth snapshot is provided
  neither by the exchange nor by archives. What is not captured now is lost forever — so
  a separate request per coin is tolerated for it.

**The set of horizons and features is declared IN ADVANCE.** Adding a horizon or
feature after looking at results is a search over cells, and the significance
threshold must then be recalibrated (see the section on two thresholds).

**FILE SIZE is a future problem.** A point's context has grown to ~1.8 KB, the cap
`hist[-500:]` per coin with 25 coins gives up to ~30 MB, and `save_json`
rewrites the file WHOLE on every point. While there are 500 points and the file is ~1.5 MB this
is unnoticeable, but within three months it will be, and the unsafe-write window on
a mounted file will grow again. The right fix is appending lines
(JSONL) instead of rewriting a dictionary; do it once the file exceeds 5 MB.

## Conditional expectancy: tools/expectancy.py

Computes spec v2 §20 — "what happened after states of the form X". Three commands:
`describe` (what is in the dataset at all), `run --where "rsi<30"` (evaluation),
`calibrate` (the procedure's empirical threshold).

* The unit is the **daily block**. Spec thresholds by number of points ("N>250 — strong
  evidence") do not apply here: 412 points at our rate is 3.4 days, where
  the critical value of t is 3.18, not 2.
* What is measured is **excess over the base of THE SAME day**, not absolute return.
  The market term cancels within a day.
* NET EV is computed from the excess minus 0.1% costs, not from gross
  return: gross return in a rising market is beta, it cannot be traded.
* Rows with `complete: false` or without the needed feature do NOT enter the sample.

**Measured 2026-09-11 (calibration v4).** Random conditions with 20% support through
the same procedure: median |t| = 0.8, **95th percentile 3.15**, maximum on
pure noise **15.15**, share of |t|>2 was **16%**. I.e. on six days
of observations the naive t lies catastrophically. The threshold is preliminary — only
6 blocks, recompute at 20+ days.

## Storing histories: only via utils/jsonstore

On 2026-09-06 at 05:04 UTC the experiment history was wiped ENTIRELY — three weeks,
1529 points. The log showed the cause second by second: writing a point (`open("w")`
truncates the file, then 6.8 MB) and outcome resolution diverged by ONE second,
the reader got a fragment, fell into `except Exception: return {}` and wrote
an empty history on top. There were no backups; recovery was impossible.

* **No `open(path,"w")` for histories.** Only `save_json` — a temporary
  file, `fsync`, `os.replace`. The replace is atomic: a reader sees either the old one
  whole or the new one whole.
* **A bare `except Exception: return {}` on read is forbidden.** "No file" and
  "file corrupted" are DIFFERENT things. `load_json` on a corrupted file sets aside
  a copy and raises `CorruptStoreError`. A loud crash is better than silent
  loss: an empty history must NEVER appear silently.
* **A long read-modify-write re-reads the file before saving.**
  Outcome resolution pulls candles for minutes; meanwhile the autoscan appends
  points, and writing a dictionary read at the start would erase them. Outcomes
  are carried into the fresh file by the pair (coin, timestamp), and a mismatch between
  the number of computed and written outcomes is printed as a warning.
* An empty file is considered CORRUPTED, not an empty history: `save_json` writes at
  least `{}`, so zero length means an interrupted write.

## Environment pitfalls

* Git Bash on Windows breaks docker in three ways: it rewrites paths in `docker cp`,
  rewrites arguments in `docker run`, and `MSYS_NO_PATHCONV=1` fixes the source and
  breaks the destination. A working combination: a relative path for the `docker cp`
  destination, `pwd -W` for volumes, MSYS_NO_PATHCONV for the whole call.
* `export PYTHONIOENCODING=utf-8` is mandatory — otherwise printing Russian text
  crashes the script, and only AFTER the work is done.
* Long tasks must write intermediate results to disk after every
  step: distilling a book takes hours.
