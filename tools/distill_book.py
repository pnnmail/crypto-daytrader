"""
DISTILLING ANY BOOK INTO RULES (a generalisation of distill_chan.py).

One book — one run, parameters via environment variables:
  BOOK_PDF      path to the PDF inside the container
  BOOK_OUT      where to write the JSON with facts
  BOOK_SOURCE   source name for the metadata
  BOOK_FIRST_PAGE  which page to start from (to cut the preface), default 0
  BOOK_TRANSLATE   "1" if the book is NOT in Russian and the rules must be translated

ABOUT TRANSLATION. It is needed only for foreign-language books and only so that
the corpora are in one language: otherwise the variants would differ in content AND
language at once, and the difference could not be explained. The English original
is kept in original_text_en — later an English-language collection is built from it
for a separate "same text, different language" comparison.

TEXT CLEANING is mandatory, and here is why. In a PDF words are split by hyphenation
("hyphen- ated"), and running headers sit in the middle of the text. Rules are extracted
as VERBATIM quotes, so the junk ends up both in the base and in the embedding —
i.e. it spoils both the output and the search.

The result is written AFTER EACH paragraph: the work takes hours, and a crash
in the middle must not cost everything before it.
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, "/app")

import pymupdf

from config import DISTILLER_MODEL
from knowledge.distiller import distill_paragraph, extract_keywords_from_chunk
from utils.helpers import compute_quality_score, is_outline_or_junk
from utils.ollama import call_ollama_json

PDF = os.environ.get("BOOK_PDF", "/tmp/chan.pdf")
OUT = os.environ.get("BOOK_OUT", "/tmp/chan_facts.json")
SOURCE_NAME = os.environ.get("BOOK_SOURCE", "unknown.pdf")
FIRST_PAGE = int(os.environ.get("BOOK_FIRST_PAGE", "0"))
NEED_TRANSLATE = os.environ.get("BOOK_TRANSLATE", "0") == "1"
MIN_PARA = 200

TRANSLATE_SCHEMA = {
    "type": "object",
    "properties": {"ru": {"type": "string"}},
    "required": ["ru"],
}


def translate(text):
    """Translation of one rule. On failure returns None, and the fact is SKIPPED
    rather than saved in the original language: a corpus mixed by language would spoil
    the comparison more than losing a few rules. The prompt is in Russian on purpose (corpus language)."""
    prompt = (
        "Переведи на русский язык это правило трейдинга. Сохрани смысл точно, "
        "термины переводи так, как принято в русской литературе по трейдингу "
        "(moving average -> скользящая средняя, mean reversion -> возврат к "
        "среднему, drawdown -> просадка). Ничего не добавляй и не объясняй.\n\n"
        f"ТЕКСТ: {text[:1500]}"
    )
    try:
        r = call_ollama_json(prompt, DISTILLER_MODEL, json_schema=TRANSLATE_SCHEMA,
                              temperature=0)
    except Exception as e:
        print(f"      ! translation failed: {e}")
        return None
    if not r or r.get("_error"):
        return None
    ru = (r.get("ru") or "").strip()
    return ru or None


def clean(text: str) -> str:
    # Word hyphenation: "hyph-\nenated" and "hyph- enated" -> "hyphenated". The second
    # case is more frequent (the PDF returns the hyphen already joined into a line) —
    # in Aziz there were 91 such paragraphs out of 366.
    # The next letter must be LOWERCASE: otherwise a dash before a capitalised
    # word would be glued ("trader - Ivanov" -> "traderIvanov").
    text = re.sub(r"(\w)-\s*\n\s*([а-яёa-z])", r"\1\2", text)
    text = re.sub(r"([а-яёa-z])-\s+([а-яёa-z])", r"\1\2", text)
    text = text.replace("ژ", " ").replace("­", "")
    text = re.sub(r"John Wiley & Sons[^\n]*", "", text)
    text = re.sub(r"https?://\S+", "", text)
    # A running header ("35 BACKTESTING AND AUTOMATED EXECUTION", "CHAPTER 2. ... 61")
    # sits on the page boundary and serves as a paragraph separator. It is replaced with
    # an EMPTY LINE rather than deleted: plain deletion glued adjacent
    # paragraphs, the average length jumped from 1557 to 4297 characters, and
    # the distiller reads only the first 4000 — text was silently lost.
    # Running headers and page numbers come as SEPARATE lines, and in a different
    # order ("CHAPTER 3. RISK MANAGEMENT" / "71" in Aziz, "35 BACKTESTING"
    # in Chan). Regexes caught this poorly, so the filter is line-by-line:
    # it does not depend on the order or on where the number is.
    keep = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            keep.append("")
            continue
        if s.isdigit() and len(s) <= 4:          # a bare page number
            keep.append("")
            continue
        letters = [c for c in s if c.isalpha()]
        # A short line where almost all letters are uppercase is a running header.
        # Threshold 0.8, not 1.0: lowercase tails like "AG" get into them.
        if (letters and len(s) < 70
                and sum(c.isupper() for c in letters) / len(letters) > 0.8):
            keep.append("")
            continue
        keep.append(line)
    return "\n".join(keep)


def paragraphs():
    doc = pymupdf.open(PDF)
    text = "".join(doc[i].get_text() + "\n"
                   for i in range(FIRST_PAGE, doc.page_count))
    doc.close()
    text = clean(text)
    out = []
    for p in re.split(r"\n\s*\n", text):
        p = " ".join(p.split())
        if len(p) >= MIN_PARA and not is_outline_or_junk(p):
            out.append(p)
    return out


def main():
    facts, done = [], 0
    if os.path.exists(OUT):
        saved = json.load(open(OUT, encoding="utf-8"))
        facts, done = saved.get("facts", []), saved.get("paragraphs_done", 0)
        print(f"resuming from paragraph {done}, facts so far {len(facts)}")

    paras = paragraphs()
    print(f"book: {SOURCE_NAME} | model: {DISTILLER_MODEL} | "
          f"translation: {'yes' if NEED_TRANSLATE else 'no'}")
    print(f"paragraphs: {len(paras)} (from {done})", flush=True)
    t0 = time.time()

    for i, para in enumerate(paras):
        if i < done:
            continue
        try:
            res = distill_paragraph(para, chapter_title=SOURCE_NAME)
        except Exception as e:
            print(f"   ! paragraph {i}: {e}", flush=True)
            res = None

        if res and res.get("facts"):
            for f in res["facts"]:
                src = (f.get("original_text") or "").strip()
                if not src:
                    continue
                if NEED_TRANSLATE:
                    ru = translate(src)
                    if not ru:
                        print("      ! fact skipped: no translation received", flush=True)
                        continue
                    f["original_text_en"] = src
                else:
                    ru = src
                f["text_ru"] = ru
                # The quality score is computed on the RUSSIAN text: it is tuned
                # for Russian and on English gave 1 versus 3-9 for Schwager,
                # i.e. chunks of a foreign-language book would systematically drop out
                # of selection for a reason unrelated to their content.
                f["quality_score"] = compute_quality_score(ru)
                f["keywords"] = extract_keywords_from_chunk(ru) or ["none"]
                facts.append(f)

        json.dump({"paragraphs_done": i + 1, "facts": facts, "source": SOURCE_NAME},
                  open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

        if (i + 1) % 10 == 0 or i + 1 == len(paras):
            el = time.time() - t0
            spd = el / max(1, i + 1 - done)
            print(f"[{i+1}/{len(paras)}] facts {len(facts)}, "
                  f"elapsed {el/60:.0f} min, left ~{(len(paras)-i-1)*spd/60:.0f} min",
                  flush=True)

    print(f"DONE: {SOURCE_NAME} -> {len(facts)} facts in {OUT}")


if __name__ == "__main__":
    main()
