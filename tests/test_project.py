from pathlib import Path
import sys
from types import SimpleNamespace

from hermen.config import ProjectConfig
from hermen.project import HermenProject, extract_text_from_file


def test_ingest_search_and_ask_with_hash_and_echo(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "handbook.md").write_text(
        "# Handbook\n\nOnboarding requires security training and payroll setup.",
        encoding="utf-8",
    )
    (docs / "benefits.md").write_text(
        "# Benefits\n\nHealth coverage starts on the first day of the month.",
        encoding="utf-8",
    )

    config = ProjectConfig()
    config.embedding.provider = "hash"
    config.query_model.provider = "echo"

    project = HermenProject.init(tmp_path, config)
    try:
        ingest_stats = project.ingest_paths([docs])
        assert ingest_stats["documents"] == 2
        assert ingest_stats["chunks"] >= 2

        results = project.search("What does onboarding require?", top_k=2)
        assert results
        assert results[0].source_path == "docs/handbook.md"

        answer = project.ask("What does onboarding require?", top_k=2)
        assert "handbook.md" in answer.answer
        assert answer.plan.search_queries

        _plan, _results, stream = project.stream_ask("What does onboarding require?", top_k=2)
        streamed_text = "".join(stream)
        assert "handbook.md" in streamed_text

        follow_up = project.ask("What about benefits?", top_k=2, history=[
            {"role": "user", "content": "What does onboarding require?"},
            {"role": "assistant", "content": answer.answer},
        ])
        assert follow_up.plan.search_queries
    finally:
        project.close()


def test_ingest_image_with_vision_enabled(tmp_path: Path) -> None:
    from PIL import Image

    image_path = tmp_path / "safety-training.png"
    Image.new("RGB", (32, 24), color="white").save(image_path)

    config = ProjectConfig()
    config.embedding.provider = "hash"
    config.query_model.provider = "echo"
    config.query_model_capabilities.vision = True

    project = HermenProject.init(tmp_path, config)
    try:
        ingest_stats = project.ingest_paths([image_path])
        assert ingest_stats["documents"] == 1
        assert ingest_stats["chunks"] == 1

        results = project.search("safety training image", top_k=1)
        assert results
        assert results[0].metadata["record_type"] == "image_semantic"
    finally:
        project.close()


def test_extract_text_from_pdf_uses_pypdf(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-pretend")

    class FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class FakePdfReader:
        def __init__(self, path: str) -> None:
            assert path.endswith("sample.pdf")
            self.pages = [FakePage("First page"), FakePage("Second page")]

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=FakePdfReader))

    text = extract_text_from_file(pdf_path)

    assert text == "First page\n\nSecond page"
