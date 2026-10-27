"""
The list of traded coins — read from robinhood_coins.json at runtime,
NOT hardcoded (the user's list changed from 7 to 50 coins over a year,
see the spec). /coins, /coins add, /coins remove edit the file on the fly.
"""
import json
from config import COINS_FILE


def load_coins():
    try:
        with open(COINS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        coins = data.get("coins", [])
        return [c.upper() for c in coins]
    except Exception:
        return []


def save_coins(coins):
    # Normalise order and case, remove duplicates, but keep the order
    # of first appearance — so the list reads predictably in /coins.
    seen = set()
    normalized = []
    for c in coins:
        c = c.upper().strip()
        if c and c not in seen:
            seen.add(c)
            normalized.append(c)
    with open(COINS_FILE, "w", encoding="utf-8") as f:
        json.dump({"coins": normalized}, f, ensure_ascii=False, indent=2)
    return normalized


def add_coin(symbol):
    coins = load_coins()
    symbol = symbol.upper().strip()
    if symbol in coins:
        return coins, False
    coins.append(symbol)
    return save_coins(coins), True


def remove_coin(symbol):
    coins = load_coins()
    symbol = symbol.upper().strip()
    if symbol not in coins:
        return coins, False
    coins.remove(symbol)
    return save_coins(coins), True
