
import sqlite3
from pathlib import Path

from app.retriever import SOURCE_DIR, tokenize

DB_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "source_search.db"
)

# Select one document as our test query.
test_file = SOURCE_DIR / "source-document00002.txt"

text = test_file.read_text(
    encoding="utf-8",
    errors="replace"
)

# Use a short excerpt from the document.
query_text = text[:1000]
tokens = tokenize(query_text)

# FTS5 uses OR between tokens to retrieve candidates.
query = " OR ".join(
    '"' + token.replace('"', '""') + '"'
    for token in tokens[:30]
)

connection = sqlite3.connect(DB_PATH)

try:
    results = connection.execute(
        """
        SELECT
            source_document,
            bm25(source_index) AS score
        FROM source_index
        WHERE source_index MATCH ?
        ORDER BY score
        LIMIT 5
        """,
        (query,)
    ).fetchall()

    print("Query document:", test_file.name)
    print("Query tokens:", len(tokens))
    print("\nTop retrieved documents:")

    for name, score in results:
        print(name, "->", round(score, 6))

finally:
    connection.close()
