
from pathlib import Path
import csv
import xml.etree.ElementTree as ET

DATA_ROOT = (
    Path(__file__).resolve().parent.parent
    / "data" / "raw" / "pan-plagiarism-corpus-2011"
)

OUTPUT_DIR = DATA_ROOT.parent / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "pan_pairs.csv"


def main():
    print("Indexing text files...", flush=True)

    # Index documents by filename once.
    text_files = {}
    for path in DATA_ROOT.rglob("*.txt"):
        text_files.setdefault(path.name, path)

    print(f"Indexed {len(text_files)} text files.", flush=True)

    xml_files = list(DATA_ROOT.rglob("suspicious-document*.xml"))
    print(f"Found {len(xml_files)} XML files.", flush=True)

    rows = []
    missing_sources = 0
    missing_suspicious = 0
    invalid_offsets = 0

    for i, xml_file in enumerate(xml_files, 1):
        root = ET.parse(xml_file).getroot()
        suspicious_ref = root.get("reference")

        suspicious_path = text_files.get(suspicious_ref)
        if suspicious_path is None:
            missing_suspicious += 1
            continue

        suspicious_text = suspicious_path.read_text(
            encoding="utf-8", errors="replace"
        )

        for feature in root.findall("./feature[@name='plagiarism']"):
            source_ref = feature.get("source_reference")
            source_path = text_files.get(source_ref)

            if source_path is None:
                missing_sources += 1
                continue

            source_text = source_path.read_text(
                encoding="utf-8", errors="replace"
            )

            try:
                start = int(feature.get("this_offset"))
                length = int(feature.get("this_length"))
                source_start = int(feature.get("source_offset"))
                source_length = int(feature.get("source_length"))
            except (TypeError, ValueError):
                invalid_offsets += 1
                continue

            suspicious_passage = suspicious_text[start:start + length]
            source_passage = source_text[
                source_start:source_start + source_length
            ]

            if (
                start < 0 or source_start < 0
                or length <= 0 or source_length <= 0
                or start + length > len(suspicious_text)
                or source_start + source_length > len(source_text)
                or not suspicious_passage.strip()
                or not source_passage.strip()
            ):
                invalid_offsets += 1
                continue

            rows.append({
                "suspicious_document": suspicious_ref,
                "source_document": source_ref,
                "suspicious_passage": suspicious_passage,
                "source_passage": source_passage,
                "plagiarism_type": feature.get("type", "unknown"),
                "manual_obfuscation": feature.get(
                    "manual_obfuscation", ""
                ),
                "label": 1,
            })

        if i % 1000 == 0:
            print(f"Processed {i}/{len(xml_files)} XML files...", flush=True)

    fields = [
        "suspicious_document", "source_document",
        "suspicious_passage", "source_passage",
        "plagiarism_type", "manual_obfuscation", "label"
    ]

    print("Writing CSV...", flush=True)
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\nFinished!")
    print("Positive pairs:", len(rows))
    print("Missing source references:", missing_sources)
    print("Missing suspicious documents:", missing_suspicious)
    print("Invalid offsets:", invalid_offsets)
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
