from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Chunk:
    index: int
    text: str


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be at least 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        if end < len(normalized):
            whitespace = normalized.rfind(" ", start, end)
            if whitespace > start:
                end = whitespace

        piece = normalized[start:end].strip()
        if piece:
            chunks.append(Chunk(index=index, text=piece))
            index += 1

        if end >= len(normalized):
            break

        # A nearby word boundary can be shorter than the requested overlap.
        start = max(start + 1, end - chunk_overlap)

    return chunks
