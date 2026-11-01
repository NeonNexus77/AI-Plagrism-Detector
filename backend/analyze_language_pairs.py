
import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from collections import Counter

DetectorFactory.seed = 42

df = pd.read_csv(
    "data/raw/processed/pan_pairs.csv",
    usecols=["suspicious_passage", "source_passage"]
)

# Sample rows while preserving the relationship between passages.
sample = df.dropna().sample(
    n=min(5000, len(df.dropna())),
    random_state=42
)

def detect_language(text):
    text = str(text).strip()

    if len(text) < 100:
        return "too_short"

    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

pair_counts = Counter()

for _, row in sample.iterrows():
    suspicious_lang = detect_language(row["suspicious_passage"])
    source_lang = detect_language(row["source_passage"])

    pair_counts[(suspicious_lang, source_lang)] += 1

print("\nLANGUAGE COMBINATIONS IN PLAGIARISM PAIRS")
print("=" * 60)

for (suspicious, source), count in pair_counts.most_common():
    percentage = count / len(sample) * 100

    print(
        f"{suspicious:12} -> {source:12} "
        f"{count:5} ({percentage:.2f}%)"
    )

print("\nTotal pairs analyzed:", len(sample))
