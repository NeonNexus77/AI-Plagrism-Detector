
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "fusion_model.joblib"


@lru_cache(maxsize=1)
def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run app/train.py first."
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_semantic_model():
    return SentenceTransformer(load_artifact()["model_name"])


def score_pairs(pairs):
    if not pairs:
        return []

    pairs = [
        (str(a).strip(), str(b).strip())
        for a, b in pairs
    ]

    if any(not a or not b for a, b in pairs):
        raise ValueError("Text pairs must not be empty.")

    artifact = load_artifact()
    vectorizer = artifact["vectorizer"]
    classifier = artifact["classifier"]

    texts = []
    for a, b in pairs:
        texts.extend([a, b])

    char_vectors = vectorizer.transform(texts)
    semantic_model = load_semantic_model()

    embeddings = semantic_model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    features = []
    raw_scores = []

    for i, (text1, text2) in enumerate(pairs):
        first = i * 2
        second = first + 1

        char_score = float(
            char_vectors[first].multiply(char_vectors[second]).sum()
        )
        semantic_score = float(
            np.clip(np.dot(embeddings[first], embeddings[second]), -1.0, 1.0)
        )
        length_ratio = min(len(text1), len(text2)) / max(
            len(text1), len(text2)
        )

        features.append([char_score, semantic_score, length_ratio])
        raw_scores.append((char_score, semantic_score))

    probabilities = classifier.predict_proba(
        np.asarray(features)
    )[:, 1]

    results = []

    for i, (char_score, semantic_score) in enumerate(raw_scores):
        results.append({
            "fingerprint_similarity": round(char_score, 4),
            "semantic_similarity": round(semantic_score, 4),
            "fusion_score": round(float(probabilities[i]), 4),
        })

    return results


def score_pair(text1, text2):
    return score_pairs([(text1, text2)])[0]
