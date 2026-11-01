
import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from collections import Counter

DetectorFactory.seed = 42

CSV_PATH = "data/raw/processed/pan_pairs.csv"

df = pd.read_csv(CSV_PATH)

def detect_language(text):
    if pd.isna(text):
        return "unknown"

    text = str(text).strip()

    if len(text) < 100:
        return "too_short"

    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


columns = ["suspicious_passage", "source_passage"]

for column in columns:
    print(f"\n{'=' * 55}")
    print(f"LANGUAGE DISTRIBUTION: {column}")
    print("=" * 55)

    # Sample up to 5000 passages for initial analysis.
    sample = df[column].dropna().drop_duplicates()

    if len(sample) > 5000:
        sample = sample.sample(n=5000, random_state=42)

    languages = sample.apply(detect_language)
    counts = Counter(languages)

    for language, count in counts.most_common():
        percentage = count / len(sample) * 100
        print(
            f"{language:15} {count:6} "
            f"({percentage:.2f}%)"
        )

    print("Total passages analyzed:", len(sample))
