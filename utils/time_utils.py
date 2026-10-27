"""
The "day" boundary for a round-the-clock market is 00:00 UTC. Explicit per the spec
(item 4): unlike stocks, there is no session close here, so Pivot Points,
VWAP, the anti-spam session_key and volume_ratio are computed from the UTC day.

IMPORTANT: UTC is the format of STORAGE and computation. For display to the user there is
format_local() — it converts to DISPLAY_TIMEZONE and labels the zone. The separation
is deliberate: moving storage to local time would break session_key
(deduplication), pivots and VWAP, which are all tied to the UTC day.
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from config import DISPLAY_TIMEZONE


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def utc_day_start(dt: datetime = None) -> datetime:
    dt = dt or now_utc()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def utc_date_str(dt: datetime = None) -> str:
    return utc_day_start(dt).strftime("%Y-%m-%d")


def session_key(coin: str, direction: str, dt: datetime = None) -> str:
    """Key for signal deduplication: coin + UTC day + direction."""
    return f"{coin.upper()}_{utc_date_str(dt)}_{direction}"


def is_weekend(dt: datetime = None) -> bool:
    dt = dt or now_utc()
    return dt.astimezone(timezone.utc).weekday() >= 5


def to_local(value) -> datetime:
    """UTC time (datetime or ISO string) -> datetime in DISPLAY_TIMEZONE."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        # Naive values in this project are always UTC (see append_signal).
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(ZoneInfo(DISPLAY_TIMEZONE))


def format_local(value, fmt: str = "%d.%m %H:%M", with_zone: bool = True) -> str:
    """Time for display to the user: local, with a zone label.

    The label is mandatory: without it the user cannot tell whether they are shown
    UTC (as stored) or their own time — and a 4-5 hour discrepancy
    looks like a data bug rather than a time zone difference.
    """
    try:
        local = to_local(value)
    except (ValueError, TypeError):
        return str(value)
    text = local.strftime(fmt)
    return f"{text} {local.strftime('%Z')}" if with_zone else text
