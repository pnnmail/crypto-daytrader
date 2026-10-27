# crypto-daytrader

A Telegram bot that analyses 25 crypto coins around the clock, plus a **controlled experiment**
that pits a rule-based scorer, a Donchian baseline and several local-LLM deciders against each other
on the same decision points.

It is a research project: the goal was to **measure** whether any of these approaches has an edge,
not to trade. The answer, so far, is no — and the interesting part is how many convincing
"edges" turned out to be measurement artefacts along the way. The rules that came out of that are
written up in [METHODOLOGY.md](METHODOLOGY.md).

## Results at a glance

| Question | Result |
|---|---|
| Directional edge at any horizon from 4 h to 14 d (with or without stops, large or small coins) | **None found** |
| Oversold → rise / overbought → fall | **No.** Short-horizon deviations from VWAP *mean-revert* rather than continue |
| STUMPY historical-analogue matching | **No predictive power** (r = 0.00 at 4 h, +0.02 at 3 d); removed from the decision in v6 |
| Does a trading book in the LLM prompt help? (`book > plain`, pre-registered) | **Failed.** t = 1.76 by day vs threshold 1.89; t = 0.89 by point vs 1.65. Passed on neither unit |
| LLM decider portfolio (equal weights, compounded, same points) | plain (Russian prompt) **−5.02%**, plain_en (English prompt) **+3.30%**; plain_en's excess over market by day is −0.038%, t = −0.16 — no edge |
| Donchian/Turtle on daily bars | **+115%** on today's 24 surviving coins vs **−30.6%** median on the full 374-coin Coinbase universe (incl. delisted) |
| Null calibration of the analysis pipeline | **25.8%** of known-null features reach \|t\| > 2; the empirical discovery threshold is \|t\| > 3.11 |
| Round-trip costs | ~0.1% on the liquid top of Binance.US (vs 1.0–1.9% on Robinhood) — so most "edges" would have to clear 0.1% *after* beta removal |

### Three "findings" that died when the sample was completed

| finding | how it looked | what killed it |
|---|---|---|
| cross-sectional momentum | +0.4% over market, t = 3.6 | lived only in illiquid coins; on tradable ones t = 0.4 |
| Donchian on daily bars | +115% vs +46% buy-and-hold | 24 survivors; on 374 coins −30.6% |
| trend-strength filter | crash zone −1.71%, t = −2.39 | it was three calendar days; the sign flipped between windows |

Survivorship alone moved median buy-and-hold from **+47%** (Robinhood list) to **−79%** (full
universe) — a 126 pp swing, larger than any effect being measured.

### Other measured pitfalls

- **Summing trade returns lies.** The first `/experiment` report printed `sum(returns)`: −77.61% for
  one variant and +63.74% for another. Positions run in parallel (~28 open at once), so the correct
  portfolio figure from the same data is −5.02% / +3.30%.
- **Excess over market needs a coin-flip control.** The market benchmark is built from LONG
  counterfactuals; in a falling 10-day window a 50/50 coin flip scored +0.151% over market and beat
  8 of 9 deciders, and "always SHORT" beat all of them.
- **Placebo horizon.** Order-flow `imbalance` gave t = 8.31 against the *past* and 1.60 against the
  future — aggressive trades move price within the same minute. Only the placebo test caught it.
- **LLM confidence is a constant.** The 4B model answered 0.78 in 88% of cases, so the confidence
  threshold filtered nothing.
- **LLM variants are not independent.** They agree on direction 73–94% of the time; five
  "confirmations" are one finding.

## How it works

```
Coinbase candles / trades / order book ─┐
Kraken Futures funding ─────────────────┤
                                        ▼
                 indicators, VWAP/pivots (UTC day), flow & book features,
                 BTC correlation, market regime            (data/, utils/)
                                        │
           ┌────────────────────────────┼─────────────────────────────┐
           ▼                            ▼                             ▼
  rule-based scorer            Donchian/Turtle baseline       LLM deciders (Ollama, qwen3.5:4b)
  (utils/signals.py)           (utils/donchian.py)            plain / plain_en / regime (+ book variants in earlier rounds)
           │                            │                             │
           └────────────── same decision point, paired ───────────────┘
                                        │
                     multi-horizon outcomes, MFE/MAE, counterfactual trade
                     (utils/outcomes.py)   →  /experiment report, tools/*.py
```

- **Live bot** (`main.py`, `telegram_bot/`): `/now <coin>`, `/scan`, `/autoscan on|off`, `/coins add|remove`,
  `/experiment`. Python decides direction, levels and size; the LLM only writes the narrative.
- **Experiment** (`utils/experiment.py`): every few hours per coin, all deciders decide on the same
  snapshot; outcomes are resolved later from minute bars. Comparisons are paired, significance is by
  daily blocks.
- **Research tools** (`tools/`): conditional expectancy with empirical thresholds (`expectancy.py`),
  null calibration and target-invariance checks (`research_protocol.py`), hash-frozen
  pre-registrations (`preregister.py`, `prereg_antiselect.py`), Binance microstructure archive
  downloader, book distillation into a RAG base.

## Notes on language

The code, comments and docs are in English. Two things are **deliberately left in Russian**:

- **LLM prompt texts** (`agents/prompts.py`, parts of `agents/decider.py`, `REGIME_BRIEF` in
  `config.py`, schema field descriptions, Russian search queries in `utils/knowledge.py`). Prompt
  language is itself an experimental variable — `plain` (Russian) vs `plain_en` (English) — so
  translating them would change the experiment.
- **Frozen pre-registration files** (`tools/preregistration.json`, `prereg_*.json` with their
  `.sha256`). They are hash-verified; editing them would invalidate the tests by design.

## Running

```bash
pip install -r requirements.txt
cp .env.example .env   # Telegram token and chat id
python main.py
```

The bot expects a local [Ollama](https://ollama.com) server with the model from `config.py`.
The book knowledge base (`chroma_db/`) is not included (it is built from copyrighted books via
`tools/distill_book.py` and `tools/ingest_books.py`); without it the book variants simply run
without a book. Market data comes from public Coinbase / Kraken / Binance endpoints — no API keys.
