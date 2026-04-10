from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from hermen.chunking import chunk_text
from hermen.config import ProjectConfig, default_config
from hermen.db import HermenDB, SearchResult
from hermen.embeddings import build_embedder
from hermen.models import build_query_model


SUPPORTED_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".csv",
    ".go",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".py",
    ".rb",
    ".rs",
    ".rst",
    ".sql",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass(slots=True)
class AskResponse:
    answer: str
    context: list[SearchResult]


class HermenProject:
    def __init__(self, root: Path, config: ProjectConfig) -> None:
        self.root = root
        self.config = config
        self.config_path = root / "hermen.toml"
        self.db = HermenDB(config.database_file(root))
        self._embedder = None
        self._query_model = None

    @classmethod
    def init(cls, root: Path, config: ProjectConfig | None = None) -> "HermenProject":
        root.mkdir(parents=True, exist_ok=True)
        project_config = config or default_config("")
        project = cls(root, project_config)
        project_config.save(project.config_path)
        return project

    @classmethod
    def open(cls, root: Path) -> "HermenProject":
        config_path = root / "hermen.toml"
        if not config_path.exists():
            raise FileNotFoundError(f"No hermen project found at {config_path}")
        return cls(root, ProjectConfig.load(config_path))

    def close(self) -> None:
        self.db.close()

    def ingest_paths(
        self,
        paths: list[Path],
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> dict[str, int]:
        embedder = self._get_embedder()
        files = list(_iter_files(paths))
        indexed_documents = 0
        indexed_chunks = 0

        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not chunks:
                continue

            vectors = embedder.embed_texts([chunk.text for chunk in chunks])
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            relative = _display_path(path, self.root)
            metadata = {
                "source_path": relative,
                "size_bytes": path.stat().st_size,
            }
            packed_chunks = [
                (
                    chunk.index,
                    chunk.text,
                    {
                        "source_path": relative,
                        "chunk_index": chunk.index,
                        "character_count": len(chunk.text),
                    },
                    vector,
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            _, inserted = self.db.upsert_document(
                source_path=relative,
                content_hash=content_hash,
                metadata=metadata,
                chunks=packed_chunks,
            )
            if inserted > 0:
                indexed_documents += 1
                indexed_chunks += inserted

        return {"documents": indexed_documents, "chunks": indexed_chunks}

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        embedder = self._get_embedder()
        vector = embedder.embed_query(query)
        return self.db.search(vector, top_k=top_k or self.config.default_top_k)

    def ask(self, question: str, top_k: int | None = None) -> AskResponse:
        results = self.search(question, top_k=top_k)
        model = self._get_query_model()
        answer = model.answer(question, results)
        return AskResponse(answer=answer, context=results)

    def stats(self) -> dict[str, int]:
        return self.db.stats()

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = build_embedder(self.config.embedding)
        return self._embedder

    def _get_query_model(self):
        if self._query_model is None:
            self._query_model = build_query_model(self.config.query_model)
        return self._query_model


def _iter_files(paths: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            discovered.append(path)
            continue

        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                    discovered.append(candidate)

    return discovered


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def serialize_search_results(results: list[SearchResult]) -> str:
    payload = [
        {
            "source_path": item.source_path,
            "chunk_index": item.chunk_index,
            "score": item.score,
            "text": item.text,
            "metadata": item.metadata,
        }
        for item in results
    ]
    return json.dumps(payload, indent=2)
