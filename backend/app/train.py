
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR / "data" / "raw" / "processed" / "pan_pairs.csv"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_POSITIVE_PAIRS = 3000
RANDOM_SEED = 42
TEST_SIZE = 0.2
BATCH_SIZE = 32


def load_positive_pairs():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required = [
        "suspicious_document",
        "source_document",
        "suspicious_passage",
        "source_passage",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    df = df.dropna(subset=required).copy()

    for col in ["suspicious_passage", "source_passage"]:
        df[col] = df[col].astype(str).str.strip()

    df = df[
        (df["suspicious_passage"] != "")
        & (df["source_passage"] != "")
    ]

    df = df.drop_duplicates(
        subset=[
            "suspicious_document",
            "source_document",
            "suspicious_passage",
            "source_passage",
        ]
    )

    if len(df) > MAX_POSITIVE_PAIRS:
        df = df.sample(
            n=MAX_POSITIVE_PAIRS,
            random_state=RANDOM_SEED,
        )

    return df.reset_index(drop=True)


def split_positive_pairs(df):
    """
    Split by suspicious document, then remove test positives
    whose source document also appears in training positives.
    """

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    groups = df["suspicious_document"].to_numpy()

    train_indices, test_indices = next(
        splitter.split(df, groups=groups)
    )

    train_df = df.iloc[train_indices].copy().reset_index(drop=True)
    test_df = df.iloc[test_indices].copy().reset_index(drop=True)

    train_sources = set(train_df["source_document"])

    original_test_size = len(test_df)

    test_df = test_df[
        ~test_df["source_document"].isin(train_sources)
    ].reset_index(drop=True)

    removed = original_test_size - len(test_df)

    if train_df.empty or test_df.empty:
        raise ValueError(
            "The split produced an empty dataset. "
            "Check the corpus and adjust the split strategy."
        )

    train_suspicious = set(train_df["suspicious_document"])
    test_suspicious = set(test_df["suspicious_document"])

    train_sources = set(train_df["source_document"])
    test_sources = set(test_df["source_document"])

    assert train_suspicious.isdisjoint(test_suspicious)
    assert train_sources.isdisjoint(test_sources)

    print("Training positive pairs:", len(train_df), flush=True)
    print("Test positive pairs:", len(test_df), flush=True)
    print("Test pairs removed for source overlap:", removed, flush=True)

    return train_df, test_df


def create_examples(positive_df, seed):
    """
    Create synthetic negative pairs by pairing each suspicious
    passage with a passage from a different source document.

    These are assumed negatives, not manually verified negatives.
    """

    rng = np.random.default_rng(seed)

    positive = positive_df[
        [
            "suspicious_document",
            "source_document",
            "suspicious_passage",
            "source_passage",
        ]
    ].copy()

    positive["label"] = 1

    negative_rows = []

    source_documents = positive_df["source_document"].to_numpy()
    source_passages = positive_df["source_passage"].to_numpy()

    for i, row in positive_df.iterrows():
        valid = np.flatnonzero(
            source_documents != row["source_document"]
        )

        if len(valid) == 0:
            continue

        j = int(rng.choice(valid))

        negative_rows.append({
            "suspicious_document": row["suspicious_document"],
            "source_document": source_documents[j],
            "suspicious_passage": row["suspicious_passage"],
            "source_passage": source_passages[j],
            "label": 0,
        })

    negative = pd.DataFrame(negative_rows)

    if negative.empty:
        raise ValueError("Could not create negative examples.")

    examples = pd.concat(
        [positive, negative],
        ignore_index=True,
    )

    examples = examples.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    return examples


def calculate_character_similarity(df, vectorizer):
    left = df["suspicious_passage"].astype(str).tolist()
    right = df["source_passage"].astype(str).tolist()

    left_vectors = vectorizer.transform(left)
    right_vectors = vectorizer.transform(right)

    # TF-IDF rows are L2-normalized, so their dot product is cosine.
    scores = np.asarray(
        left_vectors.multiply(right_vectors).sum(axis=1)
    ).ravel()

    return scores


def calculate_semantic_similarity(df, model):
    left = df["suspicious_passage"].astype(str).tolist()
    right = df["source_passage"].astype(str).tolist()

    print("Encoding suspicious passages...", flush=True)

    left_embeddings = model.encode(
        left,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print("Encoding source passages...", flush=True)

    right_embeddings = model.encode(
        right,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    scores = np.sum(
        left_embeddings * right_embeddings,
        axis=1,
    )

    return np.clip(scores, -1.0, 1.0)


def calculate_length_ratio(df):
    left_lengths = df["suspicious_passage"].astype(str).str.len().to_numpy()
    right_lengths = df["source_passage"].astype(str).str.len().to_numpy()

    denominator = np.maximum(left_lengths, right_lengths)

    return np.divide(
        np.minimum(left_lengths, right_lengths),
        denominator,
        out=np.zeros(len(df), dtype=float),
        where=denominator != 0,
    )


def fit_vectorizer(train_df):
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        sublinear_tf=True,
        max_features=100000,
        dtype=np.float32,
    )

    train_texts = (
        train_df["suspicious_passage"].astype(str).tolist()
        + train_df["source_passage"].astype(str).tolist()
    )

    print("Fitting character n-gram vectorizer...", flush=True)
    vectorizer.fit(train_texts)

    return vectorizer


def build_features(df, vectorizer, model):
    character_scores = calculate_character_similarity(
        df, vectorizer
    )

    semantic_scores = calculate_semantic_similarity(
        df, model
    )

    length_ratios = calculate_length_ratio(df)

    return np.column_stack([
        character_scores,
        semantic_scores,
        length_ratios,
    ])


def main():
    print("Loading positive passage pairs...", flush=True)
    positives = load_positive_pairs()

    print("Loaded positive pairs:", len(positives), flush=True)

    print("Splitting data by document...", flush=True)
    train_positive, test_positive = split_positive_pairs(positives)

    print("Creating training examples...", flush=True)
    train_df = create_examples(train_positive, RANDOM_SEED)

    print("Creating test examples...", flush=True)
    test_df = create_examples(test_positive, RANDOM_SEED + 1)

    X_train_labels = train_df["label"].to_numpy()
    X_test_labels = test_df["label"].to_numpy()

    print("Training examples:", len(train_df), flush=True)
    print("Test examples:", len(test_df), flush=True)

    vectorizer = fit_vectorizer(train_df)

    print("Loading MiniLM model...", flush=True)
    model = SentenceTransformer(MODEL_NAME)

    print("Building training features...", flush=True)
    X_train = build_features(train_df, vectorizer, model)

    print("Building test features...", flush=True)
    X_test = build_features(test_df, vectorizer, model)

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )

    print("Training fusion classifier...", flush=True)
    classifier.fit(X_train, X_train_labels)

    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(
            accuracy_score(X_test_labels, predictions)
        ),
        "precision": float(
            precision_score(
                X_test_labels, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                X_test_labels, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                X_test_labels, predictions, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(X_test_labels, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            X_test_labels, predictions
        ).tolist(),
        "positive_pairs_used": int(len(positives)),
        "training_examples": int(len(train_df)),
        "test_examples": int(len(test_df)),
        "features": [
            "character_similarity",
            "semantic_similarity",
            "length_ratio",
        ],
        "model_name": MODEL_NAME,
        "negative_examples": "synthetically sampled",
    }

    artifact = {
        "classifier": classifier,
        "vectorizer": vectorizer,
        "model_name": MODEL_NAME,
        "feature_names": [
            "character_similarity",
            "semantic_similarity",
            "length_ratio",
        ],
    }

    model_path = MODEL_DIR / "fusion_model.joblib"
    metrics_path = MODEL_DIR / "metrics.json"

    joblib.dump(artifact, model_path)

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nEvaluation metrics:", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)

    print("\nModel saved:", model_path, flush=True)
    print("Metrics saved:", metrics_path, flush=True)


if __name__ == "__main__":
    main()
