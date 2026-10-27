import os

# ============================================================
# EXCHANGE
# ============================================================
# Binance.com is geo-blocked for the US (verified: HTTP 451 "restricted
# location" from this machine) — not used at all. Coinbase Exchange is the
# main source (candles, order book, trades), covering all 10 coins
# of the starting list without an API key. Kraken Futures — funding rate only
# (Coinbase spot has no perpetuals).
COINBASE_BASE = "https://api.exchange.coinbase.com"
KRAKEN_FUTURES_BASE = "https://futures.kraken.com/derivatives/api/v3"

# The Coinbase product for a coin is "<COIN>-USD". A mapping in case
# the exchange ticker differs from the coin's short name.
COINBASE_QUOTE = "USD"

# Kraken Futures perpetual tickers per coin (for funding rate, stage 4).
# Checked live against /derivatives/api/v3/tickers on 2026-08-10 for the current
# robinhood_coins.json list (top 25 by liquidity) — 23 of 25 covered, only
# AERO and VVV (small/new tokens) have no perpetuals. Do not hardcode
# "by eye": when the coin list changes, check with a real request to Kraken
# rather than assuming by analogy with other exchanges.
KRAKEN_FUTURES_SYMBOL = {
    "BTC": "PF_XBTUSD", "ETH": "PF_ETHUSD", "SOL": "PF_SOLUSD", "XRP": "PF_XRPUSD",
    "ZEC": "PF_ZECUSD", "ADA": "PF_ADAUSD", "XLM": "PF_XLMUSD", "ALGO": "PF_ALGOUSD",
    "SUI": "PF_SUIUSD", "HYPE": "PF_HYPEUSD", "LINK": "PF_LINKUSD", "LTC": "PF_LTCUSD",
    "DOGE": "PF_DOGEUSD", "ONDO": "PF_ONDOUSD", "ENA": "PF_ENAUSD", "UNI": "PF_UNIUSD",
    "AVAX": "PF_AVAXUSD", "NEAR": "PF_NEARUSD", "PENGU": "PF_PENGUUSD", "FET": "PF_FETUSD",
    "CRV": "PF_CRVUSD", "WLD": "PF_WLDUSD", "HBAR": "PF_HBARUSD",
    # AERO, VVV — no perpetual on Kraken Futures (verified).
}

# Pause between paged candle requests (politeness to a public API;
# Coinbase returns no official rate limit in its headers).
CANDLE_PAGE_PAUSE_SEC = 0.25
# Coinbase returns no more than ~300 candles per request.
CANDLE_MAX_PER_REQUEST = 300

# ============================================================
# DEFINITION OF A "DAY" — A 24/7 MARKET
# ============================================================
# Crypto has no session close. All "day/session" notions that in stocks
# are tied to exchange hours (Pivot Points, VWAP, session_key for anti-spam)
# are computed here from the UTC day. This convention is fixed explicitly both in the code
# and in the bot's output (the signal card shows "UTC day").
DAY_BOUNDARY_UTC = True

# The zone for DISPLAYING time to the user. Storage and all logic stay
# in UTC (otherwise session_key, pivots, VWAP and deduplication would drift), but reading
# "16:40" when the clock says 12:40 is impossible — so in Telegram time
# is converted to local and labelled with the zone to avoid ambiguity.
DISPLAY_TIMEZONE = "America/New_York"

# ============================================================
# OLLAMA
# ============================================================
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://ollama:11434") + "/api/generate"

# The same model as in the stock bots — the only build
# that fits entirely in VRAM on an RTX 3050 (6GB), see the rationale in
# the stock-daytrader config.py. This project's container shares one Ollama with them.
TRADER_MODEL = "qwen3.5:4b"

# Families with a hybrid reasoning mode (Qwen3 and descendants, including
# qwen3.5), where thinking is ON BY DEFAULT. Without an explicit
# think=False such a model goes into a <think> block, and with structured
# output (json_schema) the "response" from Ollama comes back EMPTY — verified
# live on qwen3.5:4b with TRADER_SCHEMA. gemma4:e4b does not have this problem,
# although Ollama formally lists "thinking" as a capability for it too —
# what differs is the default behaviour of the particular family, not the
# ability to think, so the check is by name prefix, not by the API.
#
# utils/ollama.py checks this list itself — switching TRADER_MODEL to
# another "thinking" model needs no code changes, only this line.
THINKING_MODEL_PREFIXES = ("qwen3",)

# ============================================================
# TELEGRAM / STATE
# ============================================================
TELEGRAM_BOT_TOKEN = os.environ.get("CRYPTO_TELEGRAM_BOT_TOKEN", "")

# The TOKEN must be its own (two bots on one token compete for getUpdates),
# but CHAT_ID is the user's own ID, the same for all of their
# bots. Hence the fallback to the shared MY_CHAT_ID from .env: there is no point in a separate
# variable with the same number. CRYPTO_MY_CHAT_ID is kept as
# the priority one in case crypto reports need to go to another chat.
# .strip() — because in .env the value is written as "MY_CHAT_ID = 123",
# and Telegram will not forgive an extra space in an ID.
MY_CHAT_ID = (os.environ.get("CRYPTO_MY_CHAT_ID") or os.environ.get("MY_CHAT_ID", "")).strip()

STATE_FILE = "/app/state.json"
SIGNAL_HISTORY_FILE = "/app/signal_history.json"
# NO_TRADE is the control group, a separate file: without it one cannot
# test whether the entry threshold distinguishes anything real.
#
NO_TRADE_HISTORY_FILE = "/app/no_trade_history.json"
COINS_FILE = "/app/robinhood_coins.json"
AUTOSCAN_STATE_FILE = "/app/autoscan_state.json"

# The "calculation vs LLM" experiment is a separate store, see utils/experiment.py.
# It is not mixed with signal_history: that one has only TAKEN trades with per-day dedup,
# while here every decision point (including refusals) is needed, and more often.
EXPERIMENT_HISTORY_FILE = "/app/experiment_history.json"

# ============================================================
# KNOWLEDGE FROM THE BOOK (RAG for the decider, see utils/knowledge.py)
# ============================================================
# A copy of the stock bot's knowledge base (Schwager, "Technical Analysis", 389 chunks),
# not a shared volume: the projects are independent by design, the base is static and weighs
# 4 MB — duplicating is cheaper than coupling the projects.
CHROMA_DB_PATH = "/app/chroma_db"
EMBED_MODEL = "nomic-embed-text"
RAG_N_RESULTS = 5

# DECIDER VARIANTS run SIMULTANEOUSLY on one point. This way
# the comparison is not muddied by different times: the market is the same for all variants.
#   plain  — numbers only (as in p3)
#   book   — plus principles from the book
#   regime — plus the book and a briefing on the measured regime (mean reversion)
# "primary" is the one shown in the Telegram report.
#
# CUT on 2026-09-03 from seven variants to three. Measured on 1482 points:
#
#   variant   diverges from plain   t on divergences   decision
#   regime          64%             +2.14         kept: the only one
#                                                 whose divergences carry
#                                                 information (threshold 2.01)
#   plain_en        16%             +0.92         kept NOT for this, see below
#   book3           23%             -0.40         removed
#   book            12%             +0.74         removed
#   book4           10%             -0.57         removed
#   book2            6%              n/a          removed: diverges from plain
#                                                 on only 80 of 1296 points
#
# The book variants were removed because the hypothesis "book > plain" FAILED
# a pre-registered check over 8 forward days (t=1.76 at a threshold of
# 1.89 by day, t=0.89 at 1.65 by point), and their divergences from plain are
# indistinguishable from noise. Four model calls per coin were going to duplicates.
#
# plain_en is kept as the ONLY live question in the data: on the same
# 1281 points plain gives a within-coin covariance of t=-2.81, while plain_en
# gives t=-0.07, although the directions match in 83% of cases and only
# the prompt language differs. This is not explained by a difference in data volume — it was computed
# on matching points.
#
# plain must NOT be removed: it is the main variant of the frozen hypothesis
# tools/prereg_antiselect.py (hash d11ae026fafd2669).
#
# Cost: a cycle over 17 coins took 18.4 min median (~65 s per coin).
# Three variants instead of seven roughly halve it, and the wave becomes
# denser — which means the wave benchmark (the average LONG in the same hour) is more precise.
DECIDER_VARIANTS = ("plain", "plain_en", "regime")
DECIDER_PRIMARY = "regime"

# The measured-regime briefing for the "regime" variant. It is OUR result, not
# from a book: tested on two independent windows with significance by daily
# blocks (see the utils/signals.py header). We give it to the model to separate
# "the model cannot read the market" from "the model did not know this specific fact".
#
# The text is deliberately kept in Russian: the decider variants test the prompt
# language itself (plain = Russian, plain_en = English), so translating it would
# change the experiment. In English: on a 4-hour horizon crypto mean-reverts after
# downside deviations — price below VWAP, RSI < 30, lower Bollinger band and a
# bearish MACD each raise the hit rate of a LONG by 3–15 pp (sign stable on most
# coins in two time-separated windows); overbought signals do NOT predict declines,
# the only confirmed SHORT argument is price above VWAP; the effect grows with the
# depth of the drop.
REGIME_BRIEF = """ИЗМЕРЕННЫЙ ФАКТ ОБ ЭТОМ РЫНКЕ (25 монет, 400 дней, проверочное окно
отделено по времени от обучающего): в крипте на горизонте 4 часа отклонение вниз
ВОЗВРАЩАЕТСЯ, а не продолжается. Цифры — прибавка к доле удачных исходов
относительно базовой ставки самой монеты, медиана по монетам:
В скобках — на скольких монетах из проверенных знак эффекта совпал в ОБОИХ
окнах. Случайный признак совпал бы примерно на половине, поэтому это и есть
мера надёжности, а не размер выигрыша:
* цена НИЖЕ VWAP -> чаще РОСТ (+3 п.п.; глубже 1.5% -> +8 п.п.) [21 из 25]
* RSI < 30 -> чаще рост (+15 п.п.) [11 из 12]
* BB у нижней границы -> чаще рост (+7 п.п.) [22 из 25]
* MACD медвежий -> чаще рост (+3 п.п.) [19 из 25]
* цена ВЫШЕ VWAP -> чаще падение (-4 п.п.) [21 из 25]

ВАЖНАЯ АСИММЕТРИЯ: перепроданность предсказывает рост, а перекупленность падение
НЕ предсказывает. На проверке оказались неотличимы от случайных:
* RSI > 70 [совпал лишь на 1 монете из 7]
* верхняя граница Bollinger [14 из 25 — ровно как у монетки]
* MACD бычий [16 из 25]
Не считай их доводом за SHORT. Единственный подтверждённый довод за SHORT —
цена выше VWAP.

Чем сильнее отклонение вниз, тем надёжнее эффект: это проверялось отдельно, и
прибавка растёт с глубиной просадки (+3 -> +4 -> +8 п.п.). Случайная связь так
себя не ведёт, поэтому глубокому отклонению можно доверять больше, чем мелкому.

То есть классическое трендследящее чтение ("падает, значит шорт") на этом
горизонте работает ПРОТИВ. Учитывай это, но не слепо: эффект статистический и
небольшой, это сдвиг вероятности, а не гарантия для конкретной точки."""

# ============================================================
# SPREAD / COSTS
# ============================================================
# RESEARCH MODE: 0 — costs are NOT deducted from the decision. This shows the pure
# quality of signals; with the real spread the entry filter rejected almost everything and
# there was nothing to measure. Real costs are still computed and
# shown as a warning (see REALISTIC_SPREAD_* below) — so that
# the statistics are not read as a promise of profit.
SPREAD_ROUND_TRIP_PCT = 0.0

# Round-trip costs (two trades) by venue. Used ONLY for
# the warning in the report, not part of the decision.
#
# Measured on 2026-08-16: before that only Robinhood was here, and the report
# overstated the real cost about 20x for someone trading on an
# exchange. Fees are taken from official fee pages, spreads measured
# on live order books (Coinbase /book, Binance.US /ticker/bookTicker) for all
# coins from robinhood_coins.json.
#
# Robinhood Crypto, market maker routing — the app's default mode.
# No explicit fee, but Robinhood gets $0.95 per $100 of volume from
# the market maker, and it is baked into the price: 0.95% per side.
# https://cdn.robinhood.com/assets/robinhood/legal/rhc-fee-schedule.pdf
REALISTIC_SPREAD_MARKET_PCT = 1.9
# Robinhood exchange routing (Legend/Ladder), maker on the lowest tier $0-10K:
# 0.50% per side. The cheapest AT ROBINHOOD, but not overall.
REALISTIC_SPREAD_LIMIT_PCT = 1.0

# Coinbase Advanced Trade, lowest tier (<$10K per 30 days): maker 0.40% /
# taker 0.60% per side. The spread is NARROW on all 25 coins —
# median 0.0175% per side, worst (LTC) 0.045%. So at Coinbase
# you pay in fees but get liquidity everywhere.
COINBASE_MAKER_ROUND_TRIP_PCT = 0.84      # 0.40*2 + spread 0.035

# Binance.US: 0% maker / 0.01% taker for ALL users, with no volume
# requirements (checked on binance.us/fees). The fee is practically zero,
# so the whole cost is the spread, and it is uneven there:
#   ETH 0.005%, XRP 0.010%, BTC 0.017%, SOL 0.027%  — excellent;
#   DOGE..LTC 0.06-0.11%                            — acceptable;
#   ENA 1.4%, NEAR 1.6%, ONDO 1.6%, WLD 3.1%,
#   ALGO 4.6%, CRV 5.8%, FET 8.0%                   — untradeable;
#   AERO and VVV are absent altogether.
# So the figure below applies to the LIQUID TOP, not the whole list.
BINANCE_US_LIQUID_ROUND_TRIP_PCT = 0.1

# The cheapest actually achievable round-trip cost. This is what any edge found
# should be compared with: the break-even is 0.1%, not 1.0%.
# IMPORTANT: it is achievable only on the liquid top of Binance.US — on small
# coins from our list the spread there reaches 8% per side, which is worse than
# Robinhood. Cross-sectional momentum broke exactly on this: the edge
# was found precisely in the coins that cannot be bought cheaply.
BEST_ACHIEVABLE_ROUND_TRIP_PCT = BINANCE_US_LIQUID_ROUND_TRIP_PCT

# ============================================================
# ANTI-SPAM: COOLDOWN + DEDUPLICATION + DAILY LIMIT
# ============================================================
# Deduplication (session_key = coin + UTC day + direction) keeps
# a repeated /autoscan within the same day from recording a signal as a new observation —
# ported from the stock bot almost as is, only session_key is computed
# differently. The cooldown and daily limit come IN ADDITION to dedup: in stocks they
# did not exist separately (only dedup), the spec explicitly requires all three mechanisms.
AUTOSCAN_COOLDOWN_MIN = 60
MAX_SIGNALS_PER_COIN_PER_DAY = 3
AUTOSCAN_INTERVAL_MIN = 15

# ============================================================
# OUTCOME RESOLUTION HORIZON
# ============================================================
# Day trading, not position trading — 5 days (the old value) does not match
# how the system actually holds a position. The round-trip spread SPREAD_ROUND_TRIP_PCT
# already eats 1.5% on entry and exit alone, so the window must be large
# enough for the move to cover costs, but not stretch a trade
# over a day+ in a market with no session close. 4 hours is a compromise.
OUTCOME_HORIZON_HOURS = 4

# Exploratory cut-offs (recorded "for free" in the same pass over
# minute candles as the main horizon) — NOT used in win rate/conclusions,
# only material for a future comparison of horizons on live data.
CHECKPOINT_HOURS_LIST = [1, 2, 4, 8, 24]

# How often job_queue goes through the history and closes signals expired by time.
# Aligned with AUTOSCAN_INTERVAL_MIN — the same order of magnitude
# for background task frequency.
RESOLVE_INTERVAL_MIN = 15

# How often an experiment point is taken per coin. Matches
# OUTCOME_HORIZON_HOURS on purpose: observations do not overlap, which means
# the t-statistic needs no correction for autocorrelation (in the stop-loss
# backtest overlapping entries inflated the naive t several times).
EXPERIMENT_INTERVAL_HOURS = OUTCOME_HORIZON_HOURS

# Experiment switch: each coin run makes an ADDITIONAL
# LLM call (a decision without the calculation's hint), i.e. it doubles the load on
# the shared Ollama. Controlled by the /experiment on|off command.
EXPERIMENT_ENABLED_DEFAULT = True

# The model's confidence threshold below which its direction counts as a refusal.
# Applied IN CODE, not by asking in the prompt: in version p2 the prompt itself
# talked the model out of trading, and it gave NO_TRADE 100 times out of 100 —
# there was no decider. Now the model always names a direction, and
# the threshold cuts weak opinions deterministically and uniformly.
# 0.6 — because in p2 the model almost always gave 0.45, i.e. "a slight
# edge"; the threshold must cut exactly those but pass confident 0.65+.
DECIDER_CONFIDENCE_THRESHOLD = 0.6

# ============================================================
# LOGS
# ============================================================
MAX_LINE_LOG = 5000

# ============================================================
# DECISION THRESHOLDS
# ============================================================
# In stocks the threshold was 4 out of a reachable maximum of 10 (orderflow ±2 + technical
# ±3 + news ±2 + stumpy ±2 + vwap ±1). Here there is no "news" factor (see the spec —
# there is no news among the crypto data sources), and instead funding ±1 is added
# at stage 4. Maximum without funding = 8, with funding = 9. The threshold
# was rescaled proportionally (4/10 ≈ 0.4 -> 3 of 8-9) rather than inherited as
# the same number — otherwise it would be easier or harder than before for no reason.
#
# v5/v6 compressed the range ASYMMETRICALLY: three -1 votes with no measured
# basis were removed (RSI>70, MACD bullish, BB upper) as well as the STUMPY ±2 vote. The maximum became
# +7 for long and -4 for short. By the same proportion rule (0.4 of
# the reachable) the long threshold stays 3 (0.4*7=2.8), and the short threshold becomes
# -2 (0.4*4=1.6). Keeping -3 would mean requiring the short to reach 75% of its
# maximum vs 43% for the long — i.e. raising the bar by arithmetic, not by
# measurement. Shorts will become rarer anyway: they simply lost
# factors that predicted nothing.
LONG_THRESHOLD = 3
SHORT_THRESHOLD = -2
