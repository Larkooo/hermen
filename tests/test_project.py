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
