
from pathlib import Path

import fitz
from docx import Document


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def extract_text(file_path: str) -> str:
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Only PDF, DOCX, and TXT files are supported.")

    if extension == ".pdf":
        pages = []

        with fitz.open(file_path) as pdf:
            for page in pdf:
                pages.append(page.get_text())

        text = "\n".join(pages)

    elif extension == ".docx":
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs]
        text = "\n".join(paragraphs)

    else:
        text = path.read_text(encoding="utf-8-sig", errors="replace")

    if not text.strip():
        raise ValueError(
            "No text was extracted. The PDF may be scanned and need OCR."
        )

    return text
