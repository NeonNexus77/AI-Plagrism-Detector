
import sqlite3
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_DIR = BASE_DIR / "data" / "semantic_index"

INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "chunk_metadata.db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# A real source document from our indexed corpus.
SOURCE_DOCUMENT = "source-document09874.txt"


def test_real_paraphrase():

    connection = sqlite3.connect(
        f"{METADATA_PATH.as_uri()}?mode=ro",
        uri=True
    )

    try:
        row = connection.execute(
            """
            SELECT chunk_text
            FROM chunk_metadata
            WHERE source_document = ?
            AND chunk_index = 0
            """,
            (SOURCE_DOCUMENT,)
        ).fetchone()

        if row is None:
            raise RuntimeError("Original passage not found.")

        original_text = row[0]

    finally:
        connection.close()

    print("Original passage:")
    print(original_text[:1000])

    print("\nNow write a paraphrase of this passage.")
    paraphrased_text = input("\nYour paraphrase: ").strip()

    if not paraphrased_text:
        raise ValueError("Paraphrase cannot be empty.")

    print("\nLoading MiniLM...")
    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    embeddings = model.encode(
        [original_text, paraphrased_text],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    similarity = float(
        embeddings[0] @ embeddings[1]
    )

    print("\nOriginal-paraphrase similarity:", round(similarity, 4))

    print("\nLoading FAISS...")
    index = faiss.read_index(str(INDEX_PATH))

    scores, indices = index.search(
        embeddings[1:2],
        10
    )

    connection = sqlite3.connect(
        f"{METADATA_PATH.as_uri()}?mode=ro",
        uri=True
    )

    try:
        print("\nTop 10 retrieved passages:\n")

        for rank, (score, vector_id) in enumerate(
            zip(scores[0], indices[0]),
            start=1
        ):

            if vector_id == -1:
                continue

            result = connection.execute(
                """
                SELECT source_document, chunk_index, chunk_text
                FROM chunk_metadata
                WHERE vector_id = ?
                """,
                (int(vector_id),)
            ).fetchone()

            if result is None:
                continue

            document, chunk_index, passage = result

            print("Rank:", rank)
            print("Similarity:", round(float(score), 4))
            print("Document:", document)
            print("Chunk:", chunk_index)
            print("Correct source:", document == SOURCE_DOCUMENT)
            print("Passage:", passage[:250])
            print("-" * 60)

    finally:
        connection.close()


if __name__ == "__main__":
    test_real_paraphrase()
