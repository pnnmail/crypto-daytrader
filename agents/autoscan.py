"""
Background monitoring (/autoscan on/off/status). Implements THREE separate
anti-spam mechanisms (the spec requires all three explicitly, not just dedup):

  1. Deduplication (utils/signal_history.py) — one UTC day + direction =
     one observation in the HISTORY (does not affect whether we send to Telegram).
  2. Cooldown (AUTOSCAN_COOLDOWN_MIN) — the minimum interval between PUSHES
     of one coin to Telegram, regardless of history.
  3. Daily limit (MAX_SIGNALS_PER_COIN_PER_DAY) — how many times per UTC day
     a coin may reach Telegram at all.

Dedup by session_key does not replace 2 and 3: it does not prevent sending the same
signal EVERY tick (session_key does not change all day), so the cooldown and
daily limit are mandatory here, not redundant.
"""
import json
from datetime import datetime, timezone

from config import (
    AUTOSCAN_STATE_FILE, AUTOSCAN_COOLDOWN_MIN, MAX_SIGNALS_PER_COIN_PER_DAY,
    AUTOSCAN_INTERVAL_MIN, MY_CHAT_ID,
)
from utils.time_utils import now_utc, utc_date_str
from telegram_bot.sender import send_long

JOB_NAME = "autoscan"


def _load_state() -> dict:
    try:
        with open(AUTOSCAN_STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"last_push": {}, "daily_count": {}, "enabled": False, "interval_min": AUTOSCAN_INTERVAL_MIN}


def _save_state(state: dict):
    with open(AUTOSCAN_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def can_push(coin: str) -> tuple:
    state = _load_state()
    last_push = state.get("last_push", {}).get(coin)
    if last_push:
        elapsed_min = (now_utc() - datetime.fromisoformat(last_push["time"])).total_seconds() / 60
        if elapsed_min < AUTOSCAN_COOLDOWN_MIN:
            return False, f"cooldown ({elapsed_min:.0f}/{AUTOSCAN_COOLDOWN_MIN} min)"

    daily = state.get("daily_count", {}).get(coin)
    today = utc_date_str()
    count_today = daily["count"] if daily and daily["date"] == today else 0
    if count_today >= MAX_SIGNALS_PER_COIN_PER_DAY:
        return False, f"daily limit reached ({count_today}/{MAX_SIGNALS_PER_COIN_PER_DAY})"

    return True, "OK"


def register_push(coin: str, direction: str):
    state = _load_state()
    now_iso = now_utc().isoformat(timespec="seconds")
    state.setdefault("last_push", {})[coin] = {"time": now_iso, "direction": direction}

    today = utc_date_str()
    daily = state.setdefault("daily_count", {})
    prev = daily.get(coin)
    count = prev["count"] + 1 if prev and prev["date"] == today else 1
    daily[coin] = {"date": today, "count": count}

    _save_state(state)


def set_enabled(enabled: bool, interval_min: int = None):
    state = _load_state()
    state["enabled"] = enabled
    if interval_min:
        state["interval_min"] = interval_min
    _save_state(state)


def get_status() -> dict:
    state = _load_state()
    return {"enabled": state.get("enabled", False), "interval_min": state.get("interval_min", AUTOSCAN_INTERVAL_MIN)}


async def scheduled_autoscan(context):
    """JobQueue callback: goes over all coins in the list, sends to Telegram
    only what passed direction != NO_TRADE AND the cooldown AND the daily limit."""
    import asyncio
    from main import build_trade_report
    from utils.coins import load_coins

    if not MY_CHAT_ID:
        print("⚠️ autoscan: CRYPTO_MY_CHAT_ID is not set, nowhere to send")
        return

    coins = load_coins()
    for coin in coins:
        try:
            report, trader_plan = await asyncio.to_thread(build_trade_report, coin)
        except Exception as e:
            print(f"⚠️ autoscan: error on {coin}: {e}")
            continue

        direction = trader_plan.get("direction")
        if direction == "NO_TRADE":
            continue

        ok, reason = can_push(coin)
        if not ok:
            print(f"   autoscan {coin}: signal {direction}, but skipped — {reason}")
            continue

        register_push(coin, direction)
        # send_long, not bot.send_message: the report exceeded the Telegram limit
        # on 2026-08-21 and brought down the whole pass over coins (see telegram_bot/sender.py).
        await send_long(context.bot, MY_CHAT_ID, report)
