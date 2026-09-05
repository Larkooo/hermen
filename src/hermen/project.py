from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from hermen.chunking import chunk_text
from hermen.config import ProjectConfig, default_config
from hermen.db import HermenDB, SearchResult
from hermen.embeddings import build_embedder
from hermen.models import RetrievalPlan, build_query_model


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
    ".jpg",
    ".jpeg",
    ".md",
    ".pdf",
    ".png",
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
    ".webp",
    ".yaml",
    ".yml",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(slots=True)
class AskResponse:
    answer: str
    context: list[SearchResult]
    plan: RetrievalPlan


@dataclass(slots=True)
class IngestRecord:
    text: str
    metadata: dict[str, object]


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
        if (root / "hermen.toml").exists():
            raise FileExistsError(f"A hermen project already exists at {root}")
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
        query_model = self._get_query_model()
        files = list(_iter_files(paths))
        indexed_documents = 0
        indexed_chunks = 0

        for path in files:
            relative = _display_path(path, self.root)
            records = self._build_ingest_records(
                path,
                relative,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                query_model=query_model,
            )
            if not records:
                continue

            vectors = embedder.embed_texts([record.text for record in records])
            content_hash = _hash_file(path)
            metadata = {
                "source_path": relative,
                "size_bytes": path.stat().st_size,
                "vision_enabled": query_model.capabilities().vision,
            }
            packed_chunks = [
                (
                    chunk_index,
                    record.text,
                    {
                        "source_path": relative,
                        "chunk_index": chunk_index,
                        "character_count": len(record.text),
                        **record.metadata,
                    },
                    vector,
                )
                for chunk_index, (record, vector) in enumerate(zip(records, vectors, strict=True))
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

    def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> tuple[RetrievalPlan, list[SearchResult]]:
        desired_top_k = top_k or self.config.default_top_k
        model = self._get_query_model()
        plan = model.plan_retrieval(question, history, desired_top_k)
        results = self._execute_retrieval_plan(plan)
        return plan, results

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> AskResponse:
        plan, results = self.retrieve(question, top_k=top_k, history=history)
        model = self._get_query_model()
        answer = model.answer(question, results, history=history)
        return AskResponse(answer=answer, context=results, plan=plan)

    def stats(self) -> dict[str, int]:
        return self.db.stats()

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = build_embedder(self.config.embedding)
        return self._embedder

    def _get_query_model(self):
        if self._query_model is None:
            self._query_model = build_query_model(
                self.config.query_model,
                self.config.query_model_capabilities,
            )
        return self._query_model

    def _execute_retrieval_plan(self, plan: RetrievalPlan) -> list[SearchResult]:
        fused_scores: dict[str, float] = {}
        best_results: dict[str, SearchResult] = {}
        per_query_limit = max(plan.top_k, self.config.default_top_k) * 2

        for search_query in plan.search_queries:
            results = self.search(search_query, top_k=per_query_limit)
            for rank, item in enumerate(results, start=1):
                fused_scores[item.chunk_id] = fused_scores.get(item.chunk_id, 0.0) + 1.0 / (20 + rank)
                current = best_results.get(item.chunk_id)
                if current is None or item.score > current.score:
                    best_results[item.chunk_id] = item

        if plan.keywords:
            lexical_results = self.db.keyword_search(plan.keywords, top_k=per_query_limit)
            for rank, item in enumerate(lexical_results, start=1):
                fused_scores[item.chunk_id] = fused_scores.get(item.chunk_id, 0.0) + 1.25 / (20 + rank)
                current = best_results.get(item.chunk_id)
                if current is None:
                    best_results[item.chunk_id] = item

        ordered = sorted(
            best_results.values(),
            key=lambda item: (fused_scores.get(item.chunk_id, 0.0), item.score),
            reverse=True,
        )
        return ordered[: plan.top_k]

    def _build_ingest_records(
        self,
        path: Path,
        source_path: str,
        *,
        chunk_size: int,
        chunk_overlap: int,
        query_model,
    ) -> list[IngestRecord]:
        records: list[IngestRecord] = []
        suffix = path.suffix.lower()

        if suffix in IMAGE_EXTENSIONS:
            if query_model.capabilities().vision:
                records.append(
                    _describe_image_record(
                        query_model,
                        path,
                        source_path,
                        asset_type="image",
                    )
                )
            return records

        text = extract_text_from_file(path)
        records.extend(
            _chunk_records(
                text,
                source_path=source_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata={"asset_type": "document", "record_type": "text_chunk"},
            )
        )

        if suffix == ".pdf" and query_model.capabilities().vision:
            records.extend(
                _extract_pdf_image_records(
                    root=self.root,
                    pdf_path=path,
                    source_path=source_path,
                    query_model=query_model,
                )
            )

        return records


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


def extract_text_from_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _extract_text_from_pdf(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_text_from_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF ingestion requires pypdf. Install with: uv pip install -e '.[local]'"
        ) from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages)


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


def _chunk_records(
    text: str,
    *,
    source_path: str,
    chunk_size: int,
    chunk_overlap: int,
    metadata: dict[str, object],
) -> list[IngestRecord]:
    records: list[IngestRecord] = []
    for chunk in chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
        records.append(
            IngestRecord(
                text=chunk.text,
                metadata={
                    "source_path": source_path,
                    **metadata,
                },
            )
        )
    return records


def _extract_pdf_image_records(root: Path, pdf_path: Path, source_path: str, query_model) -> list[IngestRecord]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF image ingestion requires pypdf. Install with: uv pip install -e '.[local]'"
        ) from exc

    reader = PdfReader(str(pdf_path))
    records: list[IngestRecord] = []
    asset_dir = root / ".hermen" / "assets" / _safe_stem(source_path)
    asset_dir.mkdir(parents=True, exist_ok=True)

    for page_number, page in enumerate(reader.pages, start=1):
        for image_index, image in enumerate(getattr(page, "images", []), start=1):
            asset_path = asset_dir / f"page-{page_number:03d}-image-{image_index:03d}.png"
            image.image.save(asset_path, format="PNG")
            records.append(
                _describe_image_record(
                    query_model,
                    asset_path,
                    source_path,
                    asset_type="pdf_image",
                    page_number=page_number,
                    image_index=image_index,
                    asset_path_metadata=str(asset_path.relative_to(root)),
                )
            )
    return records


def _describe_image_record(
    query_model,
    image_path: Path,
    source_path: str,
    *,
    asset_type: str,
    page_number: int | None = None,
    image_index: int | None = None,
    asset_path_metadata: str | None = None,
) -> IngestRecord:
    prompt = (
        "Describe this image for retrieval. Mention visible text, labels, charts, diagrams, "
        "tables, and the main semantic content."
    )
    text = query_model.describe_image(str(image_path), prompt=prompt).strip()
    metadata: dict[str, object] = {
        "source_path": source_path,
        "asset_type": asset_type,
        "record_type": "image_semantic",
        "image_path": asset_path_metadata or str(image_path),
    }
    if page_number is not None:
        metadata["page_number"] = page_number
    if image_index is not None:
        metadata["image_index"] = image_index
    return IngestRecord(text=text, metadata=metadata)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_stem(path_text: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in path_text).strip("-") or "asset"
