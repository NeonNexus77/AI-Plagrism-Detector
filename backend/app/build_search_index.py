
import sqlite3
from pathlib import Path

SOURCE_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "reconstructed-source-documents"
)

DB_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "source_search.db"
)


def build_index():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)

    try:
        connection.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS source_index
            USING fts5(
                source_document UNINDEXED,
                source_text,
                tokenize='unicode61'
            )
        """)

        # Clear old index entries before rebuilding.
        connection.execute("DELETE FROM source_index")

        paths = sorted(SOURCE_DIR.glob("source-document*.txt"))

        if not paths:
            raise RuntimeError("No source documents found.")

        batch = []

        for i, path in enumerate(paths, start=1):
            text = path.read_text(
                encoding="utf-8",
                errors="replace"
            )

            if text.strip():
                batch.append((path.name, text))

            if len(batch) >= 100:
                connection.executemany(
                    """
                    INSERT INTO source_index
                    (source_document, source_text)
                    VALUES (?, ?)
                    """,
                    batch
                )
                connection.commit()
                batch.clear()

            if i % 1000 == 0:
                print(f"Processed {i}/{len(paths)} documents")

        if batch:
            connection.executemany(
                """
                INSERT INTO source_index
                (source_document, source_text)
                VALUES (?, ?)
                """,
                batch
            )
            connection.commit()

        count = connection.execute(
            "SELECT COUNT(*) FROM source_index"
        ).fetchone()[0]

        print("\nIndex building completed.")
        print("Documents indexed:", count)
        print("Database:", DB_PATH)
        print(
            "Database size (MB):",
            round(DB_PATH.stat().st_size / (1024 * 1024), 2)
        )

    finally:
        connection.close()


if __name__ == "__main__":
    build_index()
