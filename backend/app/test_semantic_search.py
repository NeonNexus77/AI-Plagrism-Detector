
import sqlite3
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from app.chunker import split_into_chunks


BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "reconstructed-source-documents"
)

INDEX_DIR = BASE_DIR / "data" / "semantic_index"

INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "chunk_metadata.db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def test_semantic_search():

    print("Loading FAISS index...")

    index = faiss.read_index(str(INDEX_PATH))

    print("Loading MiniLM...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    # Known source document for the test.
    source_name = "source-document09874.txt"

    source_path = SOURCE_DIR / source_name

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    source_text = source_path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    chunks = split_into_chunks(source_text)

    if not chunks:
        raise RuntimeError("No chunks found in test document.")

    query = chunks[0]

    print("\nQuery source:", source_name)
    print("Query preview:", query[:250])

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    scores, indices = index.search(
        query_embedding,
        5
    )

    connection = sqlite3.connect(
        f"{METADATA_PATH.as_uri()}?mode=ro",
        uri=True
    )

    try:
        print("\nTop 5 semantic search results:\n")

        for rank, (score, vector_id) in enumerate(
            zip(scores[0], indices[0]),
            start=1
        ):

            if vector_id == -1:
                continue

            row = connection.execute(
                """
                SELECT source_document, chunk_index, chunk_text
                FROM chunk_metadata
                WHERE vector_id = ?
                """,
                (int(vector_id),)
            ).fetchone()

            if row is None:
                print("Missing metadata for vector:", vector_id)
                continue

            document, chunk_index, chunk_text = row

            print(f"Rank: {rank}")
            print(f"Similarity: {float(score):.4f}")
            print(f"Document: {document}")
            print(f"Chunk index: {chunk_index}")
            print(f"Correct document: {document == source_name}")
            print("Passage preview:", chunk_text[:200])
            print("-" * 60)

    finally:
        connection.close()


if __name__ == "__main__":
    test_semantic_search()
