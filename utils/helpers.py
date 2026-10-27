import json
from config import STATE_FILE


def load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)


def safe_format_float(value, decimals=2, default="?"):
    """Safely formats a number. None/non-number -> default instead of an exception."""
    if value is None:
        return default
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return default


def escape_markdown(text: str) -> str:
    """Escapes Telegram parse_mode='Markdown' (legacy) special characters: _ * ` ["""
    if not text:
        return ""
    escape_chars = r'_*`['
    return ''.join(f'\\{c}' if c in escape_chars else c for c in text)
