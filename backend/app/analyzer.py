
import re

from rank_bm25 import BM25Okapi

from app.chunker import split_into_chunks
from app.retriever import retrieve_sources, tokenize
from app.scoring import score_pairs


def analyze_text(text, top_k=5):
    chunks = split_into_chunks(text)

    if not chunks:
        return {
            "chunks_analyzed": 0,
            "sources_checked": 0,
            "matches": [],
        }

    all_matches = []
    sources_checked = set()

    for chunk_index, chunk in enumerate(chunks):
        candidates = retrieve_sources(chunk, top_k=top_k)

        for candidate in candidates:
            source_name = candidate["source_document"]
            sources_checked.add(source_name)

            source_chunks = split_into_chunks(candidate["source_text"])

            if not source_chunks:
                continue

            query_tokens = tokenize(chunk)
            if not query_tokens:
                continue

            tokenized_chunks = [
                tokenize(source_chunk) for source_chunk in source_chunks
            ]

            if not any(tokenized_chunks):
                continue

            index = BM25Okapi(tokenized_chunks)
            scores = index.get_scores(query_tokens)

            best_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:5]

            selected = [
                i for i in best_indices if scores[i] > 0
            ]

            if not selected:
                continue

            pairs = [
                (chunk, source_chunks[i])
                for i in selected
            ]
            pair_scores = score_pairs(pairs)

            best_position = max(
                range(len(pair_scores)),
                key=lambda i: pair_scores[i]["fusion_score"],
            )

            source_index = selected[best_position]

            all_matches.append({
                "chunk_index": chunk_index,
                "submitted_passage": chunk,
                "source_document": source_name,
                "source_passage": source_chunks[source_index],
                "bm25_score": candidate["bm25_score"],
                **pair_scores[best_position],
            })

    all_matches.sort(
        key=lambda item: item["fusion_score"],
        reverse=True,
    )

    unique_matches = []
    seen = set()

    for match in all_matches:
        key = (match["chunk_index"], match["source_document"])

        if key not in seen:
            seen.add(key)
            unique_matches.append(match)

    return {
        "chunks_analyzed": len(chunks),
        "sources_checked": len(sources_checked),
        "matches": unique_matches[:20],
    }
