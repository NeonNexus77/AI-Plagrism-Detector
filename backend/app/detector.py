
import re
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_model():
    return SentenceTransformer(MODEL_NAME)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fingerprint_similarity(text1, text2):
    text1 = clean_text(text1)
    text2 = clean_text(text2)

    if not text1 or not text2:
        return 0.0

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        sublinear_tf=True,
    )

    try:
        vectors = vectorizer.fit_transform([text1, text2])
    except ValueError:
        return 0.0

    return float(cosine_similarity(vectors[0], vectors[1])[0, 0])


def semantic_similarity(text1, text2):
    text1 = clean_text(text1)
    text2 = clean_text(text2)

    if not text1 or not text2:
        return 0.0

    model = load_model()

    embeddings = model.encode(
        [text1, text2],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    score = np.dot(embeddings[0], embeddings[1])
    return float(np.clip(score, -1.0, 1.0))


def compare_texts(text1, text2):
    fingerprint_score = fingerprint_similarity(text1, text2)
    semantic_score = semantic_similarity(text1, text2)

    return {
        "fingerprint_similarity": round(fingerprint_score, 4),
        "semantic_similarity": round(semantic_score, 4),
    }

