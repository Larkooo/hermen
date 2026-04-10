from pathlib import Path

from hermen.config import ProjectConfig
from hermen.project import HermenProject


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
