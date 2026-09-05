import json
from pathlib import Path

from rich.text import Text
from typer.testing import CliRunner

from hermen.cli import app
from hermen.config import ProjectConfig


def test_offline_cli_workflow_and_json(tmp_path: Path) -> None:
    runner = CliRunner()
    root = tmp_path / "index"
    result = runner.invoke(app, ["init", "--root", str(root),
                                "--query-provider", "echo", "--embedding-provider", "hash"])
    assert result.exit_code == 0, result.output
    document = tmp_path / "notes.md"
    content = "backups retention 30 days " + "[bold]literal[/bold] " * 10
    document.write_text(content)
    result = runner.invoke(app, ["ingest", str(document), "--root", str(root)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["search", "backups retention", "--root", str(root), "--json"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert rows[0]["text"] == content.strip()
    result = runner.invoke(app, ["ask", "backups retention", "--root", str(root)])
    assert result.exit_code == 0 and "notes.md" in result.output


def test_init_requires_model_only_for_local_inference(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["init", "--root", str(tmp_path)])
    assert result.exit_code == 2
    assert "--model-path is required" in Text.from_ansi(result.output).plain
    assert not (tmp_path / "hermen.toml").exists()


def test_init_preserves_existing_config(tmp_path: Path) -> None:
    original = 'database_path = "keep.db"'
    (tmp_path / "hermen.toml").write_text(original)
    result = CliRunner().invoke(app, ["init", "--root", str(tmp_path),
                                    "--query-provider", "echo"])
    assert result.exit_code != 0
    assert (tmp_path / "hermen.toml").read_text() == original


def test_config_round_trip_with_escaped_strings(tmp_path: Path) -> None:
    config = ProjectConfig()
    config.database_path = 'notes "quoted"\\database.db'
    config.query_model.model_path = "model\nwith newline 🧪.gguf"
    path = tmp_path / "hermen.toml"
    config.save(path)
    assert ProjectConfig.load(path) == config
