
import sqlite3
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.chunker import split_into_chunks


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

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

TEMP_INDEX_PATH = INDEX_DIR / "faiss.tmp.index"
TEMP_METADATA_PATH = INDEX_DIR / "chunk_metadata.tmp.db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDING_DIMENSION = 384
BATCH_SIZE = 32
PROGRESS_INTERVAL = 100


# --------------------------------------------------
# INITIALIZATION
# --------------------------------------------------

def initialize_metadata_database(db_path):
    connection = sqlite3.connect(db_path)

    connection.execute("""
        CREATE TABLE chunk_metadata (
            vector_id INTEGER PRIMARY KEY,
            source_document TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL
        )
    """)

    connection.execute("""
        CREATE INDEX idx_source_document
        ON chunk_metadata(source_document)
    """)

    connection.commit()

    return connection


# --------------------------------------------------
# BUILD SEMANTIC INDEX
# --------------------------------------------------

def build_semantic_index():

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Source directory not found: {SOURCE_DIR}"
        )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    source_files = sorted(
        SOURCE_DIR.glob("source-document*.txt")
    )

    if not source_files:
        raise RuntimeError("No source documents found.")

    print("Documents found:", len(source_files))
    print("Loading MiniLM on CPU...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    index = faiss.IndexFlatIP(
        EMBEDDING_DIMENSION
    )

    # Start fresh temporary output files.
    TEMP_INDEX_PATH.unlink(missing_ok=True)
    TEMP_METADATA_PATH.unlink(missing_ok=True)

    connection = initialize_metadata_database(
        TEMP_METADATA_PATH
    )

    pending_texts = []
    pending_metadata = []

    total_chunks = 0
    processed_documents = 0

    def process_batch():

        nonlocal total_chunks

        if not pending_texts:
            return

        embeddings = model.encode(
            pending_texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        # FAISS IDs follow insertion order.
        index.add(embeddings)

        rows = []

        for offset, metadata in enumerate(pending_metadata):
            vector_id = total_chunks + offset

            rows.append((
                vector_id,
                metadata["source_document"],
                metadata["chunk_index"],
                metadata["chunk_text"]
            ))

        connection.executemany(
            """
            INSERT INTO chunk_metadata (
                vector_id,
                source_document,
                chunk_index,
                chunk_text
            )
            VALUES (?, ?, ?, ?)
            """,
            rows
        )

        connection.commit()

        total_chunks += len(pending_texts)

        pending_texts.clear()
        pending_metadata.clear()

    try:

        for file_path in source_files:

            text = file_path.read_text(
                encoding="utf-8",
                errors="replace"
            )

            chunks = split_into_chunks(text)

            for chunk_index, chunk in enumerate(chunks):

                pending_texts.append(chunk)

                pending_metadata.append({
                    "source_document": file_path.name,
                    "chunk_index": chunk_index,
                    "chunk_text": chunk
                })

                if len(pending_texts) >= BATCH_SIZE:
                    process_batch()

            processed_documents += 1

            if (
                processed_documents % PROGRESS_INTERVAL == 0
                or processed_documents == len(source_files)
            ):
                print(
                    f"Processed {processed_documents}/"
                    f"{len(source_files)} documents | "
                    f"Indexed chunks: {total_chunks + len(pending_texts)}"
                )

        # Process the final incomplete batch.
        process_batch()

        if index.ntotal != total_chunks:
            raise RuntimeError(
                "FAISS and SQLite metadata counts do not match."
            )

        faiss.write_index(
            index,
            str(TEMP_INDEX_PATH)
        )

        connection.close()
        connection = None

        # Publish completed files only after a successful build.
        TEMP_INDEX_PATH.replace(INDEX_PATH)
        TEMP_METADATA_PATH.replace(METADATA_PATH)

        print("\nSemantic indexing completed successfully!")
        print("Documents processed:", processed_documents)
        print("Total chunks indexed:", total_chunks)
        print("FAISS vectors:", index.ntotal)
        print("Embedding dimensions:", EMBEDDING_DIMENSION)
        print("FAISS index:", INDEX_PATH)
        print("Metadata database:", METADATA_PATH)

    except Exception:
        if connection is not None:
            connection.close()

        raise


if __name__ == "__main__":
    build_semantic_index()
