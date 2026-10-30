
import re
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi


DATA_ROOT = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "pan-plagiarism-corpus-2011"
)

SOURCE_DIR = DATA_ROOT / "external-detection-corpus" / "source-document"


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


@lru_cache(maxsize=1)
def load_source_index():
    paths = sorted(SOURCE_DIR.rglob("source-document*.txt"))

    documents = []
    for path in paths:
        try:
            text = path.read_text(
                encoding="utf-8", errors="replace"
            )
            if text.strip():
                documents.append({
                    "name": path.name,
                    "path": str(path),
                    "text": text,
                })
        except OSError:
            continue

    if not documents:
        raise RuntimeError(
            f"No source documents found in {SOURCE_DIR}"
        )

    tokenized = [tokenize(doc["text"]) for doc in documents]
    index = BM25Okapi(tokenized)

    return documents, index


def retrieve_sources(query, top_k=5):
    if not query.strip():
        return []

    documents, index = load_source_index()
    tokens = tokenize(query)

    if not tokens:
        return []

    scores = index.get_scores(tokens)
    ranked = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    results = []

    for i in ranked[:top_k]:
        if scores[i] <= 0:
            continue

        results.append({
            "source_document": documents[i]["name"],
            "source_text": documents[i]["text"],
            "bm25_score": round(float(scores[i]), 4),
        })

    return results
