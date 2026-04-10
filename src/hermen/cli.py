from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from hermen.config import DEFAULT_EMBEDDING_MODEL, default_config
from hermen.models import infer_query_model_capabilities
from hermen.project import HermenProject, serialize_search_results


app = typer.Typer(help="A local-first vector database with a built-in model query engine.")
console = Console()


@app.command()
def init(
    root: Path = typer.Option(Path("."), help="Project directory."),
    model_path: Path = typer.Option(..., exists=True, dir_okay=False, help="Path to a GGUF model."),
    clip_model_path: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Optional multimodal projector path for local vision support.",
    ),
    embedding_provider: str = typer.Option("sentence_transformers", help="Embedding provider."),
    embedding_model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, help="Embedding model name."),
    query_provider: str = typer.Option("llama_cpp", help="Query model provider."),
    default_top_k: int = typer.Option(6, min=1, help="Default retrieval count."),
    n_ctx: int = typer.Option(8192, min=512, help="Model context window."),
    temperature: float = typer.Option(0.1, min=0.0, max=2.0, help="Sampling temperature."),
) -> None:
    config = default_config(str(model_path))
    config.embedding.provider = embedding_provider
    config.embedding.model = embedding_model
    config.query_model.provider = query_provider
    config.query_model.model_path = str(model_path)
    config.query_model.clip_model_path = str(clip_model_path) if clip_model_path else ""
    config.default_top_k = default_top_k
    config.query_model.n_ctx = n_ctx
    config.query_model.temperature = temperature
    config.query_model_capabilities = infer_query_model_capabilities(config.query_model)

    project = HermenProject.init(root, config)
    project.close()
    console.print(f"Initialized hermen project at [bold]{root.resolve()}[/bold]")
    console.print(f"Database: {config.database_path}")
    console.print(f"Query model: {config.query_model.model_path}")
    console.print(f"Vision enabled: {config.query_model_capabilities.vision}")


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(..., exists=True, help="Files or directories to ingest."),
    root: Path = typer.Option(Path("."), help="Project directory."),
    chunk_size: int = typer.Option(1000, min=100, help="Chunk size in characters."),
    chunk_overlap: int = typer.Option(200, min=0, help="Chunk overlap in characters."),
) -> None:
    project = HermenProject.open(root)
    try:
        stats = project.ingest_paths(paths, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    finally:
        project.close()

    console.print(
        f"Indexed [bold]{stats['documents']}[/bold] documents and [bold]{stats['chunks']}[/bold] chunks"
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    root: Path = typer.Option(Path("."), help="Project directory."),
    top_k: int = typer.Option(6, min=1, help="Number of matches to return."),
    json_output: bool = typer.Option(False, "--json", help="Print results as JSON."),
) -> None:
    project = HermenProject.open(root)
    try:
        results = project.search(query, top_k=top_k)
    finally:
        project.close()

    if json_output:
        console.print(serialize_search_results(results))
        return

    table = Table(title="Search Results")
    table.add_column("Score", justify="right")
    table.add_column("Source")
    table.add_column("Chunk")
    table.add_column("Preview")

    for item in results:
        table.add_row(
            f"{item.score:.4f}",
            item.source_path,
            str(item.chunk_index),
            item.text[:160].replace("\n", " "),
        )

    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer."),
    root: Path = typer.Option(Path("."), help="Project directory."),
    top_k: int = typer.Option(6, min=1, help="Number of chunks to retrieve."),
) -> None:
    project = HermenProject.open(root)
    try:
        response = project.ask(question, top_k=top_k)
    finally:
        project.close()

    sources = "\n".join(f"- {item.source_path}#{item.chunk_index}" for item in response.context)
    console.print(Panel(Text(response.answer), title="Answer"))
    console.print(Panel(Text(sources or "No sources retrieved."), title="Sources"))


@app.command()
def chat(
    root: Path = typer.Option(Path("."), help="Project directory."),
    top_k: int = typer.Option(6, min=1, help="Number of chunks to retrieve per turn."),
) -> None:
    project = HermenProject.open(root)
    console.print("Type a question. Type 'exit' or 'quit' to stop.")
    history: list[dict[str, str]] = []
    try:
        while True:
            question = typer.prompt("hermen")
            if question.strip().lower() in {"exit", "quit"}:
                break
            response = project.ask(question, top_k=top_k, history=history)
            console.print(Panel(Text(response.answer), title="Answer"))
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": response.answer})
    finally:
        project.close()


@app.command()
def stats(
    root: Path = typer.Option(Path("."), help="Project directory."),
) -> None:
    project = HermenProject.open(root)
    try:
        snapshot = project.stats()
    finally:
        project.close()

    table = Table(title="Database Stats")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in snapshot.items():
        table.add_row(key, str(value))
    console.print(table)
