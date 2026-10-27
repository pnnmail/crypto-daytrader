"""
STAGE 2: inserting the rules of both books into chroma — into THREE collections.

  trading_knowledge_chan_ru — Chan, Russian translation
  trading_knowledge_chan_en — Chan, English original
  trading_knowledge_aziz    — Aziz, Russian original

Each book in ITS OWN collection: they must not be mixed, otherwise a variant stops
being "so-and-so's book" and comparing books loses its meaning. Schwager
(trading_knowledge) is not touched at all.

Chan gives two collections from ONE distillation: the content is identical,
only the language differs. That is a separate question — does the model reason better
in English with the content held constant.

Cheap and repeatable: embeddings only. If it crashes, a restart does not cost
a multi-hour distillation, which is already in JSON.
"""
import json
import os

import chromadb
import requests
from chromadb import Documents, EmbeddingFunction, Embeddings

DB = os.environ.get("BOOKS_DB", "/chroma")
FACTS_DIR = os.environ.get("BOOKS_FACTS", "/facts")
OLLAMA = os.environ.get("OLLAMA_EMBED_URL", "http://ollama:11434/api/embeddings")
EMBED_MODEL = "nomic-embed-text"

# (file, collection, text field, source name)
PLAN = [
    ("chan_facts.json", "trading_knowledge_chan_ru", "text_ru",
     "Chan_Algorithmic_Trading_2013.pdf"),
    ("chan_facts.json", "trading_knowledge_chan_en", "original_text_en",
     "Chan_Algorithmic_Trading_2013.pdf"),
    ("aziz_facts.json", "trading_knowledge_aziz", "text_ru",
     "Aziz_Day_Trading_Guide.pdf"),
]


class OllamaEmbedding(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        out = []
        for text in input:
            r = requests.post(OLLAMA, json={"model": EMBED_MODEL,
                                            "prompt": text[:1000]}, timeout=120).json()
            vec = r.get("embedding") or (r.get("embeddings") or [None])[0]
            out.append(vec if vec else [0.0] * 768)
        return out


def norm_list(v):
    """Returns a LIST, not a string.

    Checked on the live base: for Schwager keywords and market_phase are stored
    as lists, and selection in utils/knowledge.py reads them via set(...). With
    a string, set() would split it into individual LETTERS, and chunks of the new books
    would get a zero keyword score — they would drop out of the output for a
    reason unrelated to their content.
    """
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v] or ["none"]
    if v is None or v == "":
        return ["none"]
    return [str(v)]


MIN_FACT_LEN = 40


def is_junk(text: str) -> bool:
    """Filtering out fragments.

    The distiller occasionally returns not a rule from the book but a TEMPLATE PHRASE FROM
    ITS OWN PROMPT ("If X — then Y") or a stub like "Enter the
    trade.". The built-in hallucination guard lets them through: it checks
    the quote against the source text only if it is longer than 20 characters, and these
    are shorter. In Aziz there turned out to be 26 such of 494 (5%).

    A length threshold closes both cases at once: a real rule with a condition
    and an action shorter than 40 characters practically never occurs.
    """
    t = (text or "").strip().strip('"').strip()
    return len(t) < MIN_FACT_LEN


def main():
    client = chromadb.PersistentClient(path=DB)
    ef = OllamaEmbedding()
    for fname, colname, field, source in PLAN:
        path = os.path.join(FACTS_DIR, fname)
        if not os.path.exists(path):
            print(f"{colname}: no file {path} — skipping")
            continue
        facts = json.load(open(path, encoding="utf-8"))["facts"]

        # For a PAIR of collections (one book, two languages) the selection must be shared:
        # a fact gets into both or neither. Otherwise there were 232 chunks in
        # the Russian one versus 228 in the English one — the content stopped being
        # identical, and the language comparison lost its cleanliness, because
        # different rule sets would also be compared.
        pair_fields = [fld for fn, _, fld, _ in PLAN if fn == fname]
        def keep(fact):
            return all(not is_junk((fact.get(fl) or "").strip())
                       for fl in pair_fields)

        docs, ids, metas = [], [], []
        skipped = 0
        for i, f in enumerate(facts):
            text = (f.get(field) or "").strip()
            if not text:
                continue
            if not keep(f):
                skipped += 1
                continue
            cat = (f.get("category") or "entry").lower()
            docs.append(f"[{cat.upper()}] {text}")
            ids.append(f"{colname}_{i}")
            metas.append({
                "source": source,
                "concept": f.get("concept") or "",
                "action": f.get("action") or "",
                "market_phase": norm_list(f.get("market_phase")),
                "confidence": f.get("confidence") or "",
                "category": cat,
                "keywords": norm_list(f.get("keywords")),
                "quality_score": f.get("quality_score") or 3,
                "original_text": text,
            })
        if not docs:
            print(f"{colname}: nothing to insert (field {field} is empty)")
            continue
        try:
            client.delete_collection(colname)   # idempotent on restart
        except Exception:
            pass
        col = client.get_or_create_collection(colname, embedding_function=ef)
        for s in range(0, len(docs), 50):
            col.add(documents=docs[s:s + 50], ids=ids[s:s + 50],
                    metadatas=metas[s:s + 50])
            print(f"   {colname}: {min(s + 50, len(docs))}/{len(docs)}", flush=True)
        print(f"{colname}: done, {col.count()} chunks "
              f"(fragments filtered out: {skipped})")


if __name__ == "__main__":
    main()
