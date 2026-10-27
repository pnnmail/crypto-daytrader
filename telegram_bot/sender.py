"""
Sending long messages to Telegram.

Appeared after the 2026-08-21 crash: the report grew to 4339 characters against
Telegram's 4096 limit, and the autoscan crashed with "Message is too long". Splitting into chunks
EXISTED in the project, but only in the command handlers — the background push sent
the report whole. Worse, the fallback path "retry without Markdown" on
a too-long message failed a SECOND time with the same exception, cutting off
the autoscan pass over the remaining coins (skipped jobs followed in the
logs).

So here there is one sender for both paths. The difference between "the markup
broke" and "the message is too long" is substantial: the first is fixed by
a retry without parse_mode, the second only by splitting, and a retry does not fix it.
"""
TELEGRAM_HARD_LIMIT = 4096
# A margin from the hard limit: Markdown markup and service characters may
# be counted differently, and hitting the boundary costs more than sending an extra chunk.
CHUNK_SIZE = 3800


def split_message(text: str, limit: int = CHUNK_SIZE):
    """Cuts the text by lines, without breaking them unless necessary.

    A line that is ITSELF longer than the limit is handled separately: the previous
    version put it into a chunk as is, and such a chunk still failed.
    The case is not theoretical — one long LLM field without line breaks gives
    exactly that.
    """
    if len(text) <= limit:
        return [text]
    chunks, current = [], ""
    for line in text.split("\n"):
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) + 1 > limit:
            chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)
    return chunks


async def send_long(bot, chat_id, text: str, parse_mode: str = "Markdown"):
    """Sends text of any length. A chunk that failed because of markup
    is retried as plain text — its length is already known to fit."""
    for chunk in split_message(text):
        try:
            await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=parse_mode)
        except Exception:
            await bot.send_message(chat_id=chat_id, text=chunk)
