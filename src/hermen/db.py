from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from typing import Iterable
from uuid import uuid4

import numpy as np


@dataclass(slots=True)
class SearchResult:
    chunk_id: str
    document_id: str
    source_path: str
    chunk_index: int
    text: str
    metadata: dict[str, object]
    score: float


class HermenDB:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._ensure_schema()

    def close(self) -> None:
        self.connection.close()

    def _ensure_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL UNIQUE,
                content_hash TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id TEXT PRIMARY KEY,
                dimensions INTEGER NOT NULL,
                vector BLOB NOT NULL,
                FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
            """
        )
        self.connection.commit()

    def upsert_document(
        self,
        *,
        source_path: str,
        content_hash: str,
        metadata: dict[str, object],
        chunks: Iterable[tuple[int, str, dict[str, object], np.ndarray]],
    ) -> tuple[str, int]:
        existing = self.connection.execute(
            "SELECT id, content_hash FROM documents WHERE source_path = ?",
            (source_path,),
        ).fetchone()

        if existing and existing["content_hash"] == content_hash:
            return str(existing["id"]), 0

        document_id = str(existing["id"]) if existing else str(uuid4())
        now = datetime.now(UTC).isoformat()

        with self.connection:
            if existing:
                self.connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                self.connection.execute(
                    """
                    UPDATE documents
                    SET content_hash = ?, metadata_json = ?, created_at = ?
                    WHERE id = ?
                    """,
                    (content_hash, json.dumps(metadata), now, document_id),
                )
            else:
                self.connection.execute(
                    """
                    INSERT INTO documents (id, source_path, content_hash, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (document_id, source_path, content_hash, json.dumps(metadata), now),
                )

            count = 0
            for chunk_index, text, chunk_metadata, vector in chunks:
                chunk_id = str(uuid4())
                self.connection.execute(
                    """
                    INSERT INTO chunks (id, document_id, chunk_index, text, metadata_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (chunk_id, document_id, chunk_index, text, json.dumps(chunk_metadata)),
                )
                self.connection.execute(
                    """
                    INSERT INTO embeddings (chunk_id, dimensions, vector)
                    VALUES (?, ?, ?)
                    """,
                    (chunk_id, int(vector.shape[0]), vector.astype(np.float32).tobytes()),
                )
                count += 1

        return document_id, count

    def search(self, query_vector: np.ndarray, top_k: int = 6) -> list[SearchResult]:
        rows = self.connection.execute(
            """
            SELECT
                chunks.id AS chunk_id,
                chunks.document_id AS document_id,
                documents.source_path AS source_path,
                chunks.chunk_index AS chunk_index,
                chunks.text AS text,
                chunks.metadata_json AS metadata_json,
                embeddings.vector AS vector
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            JOIN embeddings ON embeddings.chunk_id = chunks.id
            """
        ).fetchall()

        if not rows:
            return []

        query = _normalize(query_vector.astype(np.float32))
        results: list[SearchResult] = []

        for row in rows:
            vector = np.frombuffer(row["vector"], dtype=np.float32)
            score = float(np.dot(_normalize(vector), query))
            results.append(
                SearchResult(
                    chunk_id=str(row["chunk_id"]),
                    document_id=str(row["document_id"]),
                    source_path=str(row["source_path"]),
                    chunk_index=int(row["chunk_index"]),
                    text=str(row["text"]),
                    metadata=json.loads(str(row["metadata_json"])),
                    score=score,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def stats(self) -> dict[str, int]:
        documents = self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"documents": int(documents), "chunks": int(chunks)}


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / norm
