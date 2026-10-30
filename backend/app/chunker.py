
import re


def split_into_chunks(text, max_chars=1200, overlap=200):
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("Invalid chunk size or overlap.")

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        # Prefer a sentence or word boundary near the chunk end.
        if end < len(text):
            boundary = text.rfind(". ", start + max_chars // 2, end)
            if boundary != -1:
                end = boundary + 1
            else:
                boundary = text.rfind(" ", start + max_chars // 2, end)
                if boundary != -1:
                    end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks
