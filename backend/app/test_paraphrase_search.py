
import sqlite3
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_DIR = BASE_DIR / "data" / "semantic_index"

INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "chunk_metadata.db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Original passage from the known source.
original_text = """
Machine learning enables computers to learn patterns from data
and make predictions without being explicitly programmed.
"""

# Manually rewritten version of the same meaning.
paraphrased_text = """
Computers can identify patterns in datasets and predict outcomes
through machine learning, without requiring explicit instructions
for every task.
"""


def test_paraphrase_search():

    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_PATH))

    print("Loading MiniLM...")
    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    embeddings = model.encode(
        [original_text, paraphrased_text],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    similarity = float(embeddings[0] @ embeddings[1])

    print("\nOriginal vs paraphrased similarity:", round(similarity, 4))

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

            row = connection.execute(
                """
                SELECT source_document, chunk_index, chunk_text
                FROM chunk_metadata
                WHERE vector_id = ?
                """,
                (int(vector_id),)
            ).fetchone()

            if row:
                document, chunk_index, chunk_text = row

                print("Rank:", rank)
                print("Similarity:", round(float(score), 4))
                print("Document:", document)
                print("Chunk:", chunk_index)
                print("Passage:", chunk_text[:250])
                print("-" * 60)

    finally:
        connection.close()


if __name__ == "__main__":
    test_paraphrase_search()
