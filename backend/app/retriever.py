
import re
import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "reconstructed-source-documents"
)

DB_PATH = BASE_DIR / "data" / "source_search.db"


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def retrieve_sources(query, top_k=5):
    if not query or not query.strip():
        return []

    if top_k <= 0:
        return []

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Search index not found at {DB_PATH}. "
            "Run app/build_search_index.py first."
        )

    tokens = list(dict.fromkeys(tokenize(query)))

    if not tokens:
        return []

    # Quote each token and combine using OR.
    fts_query = " OR ".join(
        '"' + token.replace('"', '""') + '"'
        for token in tokens
    )

    connection = sqlite3.connect(
        f"{DB_PATH.as_uri()}?mode=ro",
        uri=True
    )

    try:
        rows = connection.execute(
            """
            SELECT
                source_document,
                source_text,
                bm25(source_index) AS score
            FROM source_index
            WHERE source_index MATCH ?
            ORDER BY score ASC
            LIMIT ?
            """,
            (fts_query, top_k)
        ).fetchall()

        results = []

        for source_document, source_text, score in rows:
            results.append({
                "source_document": source_document,
                "source_text": source_text,
                "bm25_score": round(-float(score), 4),
            })

        return results

    finally:
        connection.close()
