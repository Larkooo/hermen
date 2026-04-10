from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(slots=True)
class EmbeddingConfig:
    provider: str = "sentence_transformers"
    model: str = DEFAULT_EMBEDDING_MODEL
    dimensions: int = 384


@dataclass(slots=True)
class QueryModelConfig:
    provider: str = "llama_cpp"
    model: str = "gemma-4-E2B-i1-Q4_K_M.gguf"
    model_path: str = ""
    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    n_ctx: int = 8192
    n_gpu_layers: int = -1
    temperature: float = 0.1
    max_tokens: int = 512


@dataclass(slots=True)
class ProjectConfig:
    schema_version: int = 1
    database_path: str = ".hermen/hermen.db"
    default_top_k: int = 6
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    query_model: QueryModelConfig = field(default_factory=QueryModelConfig)

    @classmethod
    def load(cls, path: Path) -> "ProjectConfig":
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        embedding = data.get("embedding", {})
        query_model = data.get("query_model", {})
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            database_path=str(data.get("database_path", ".hermen/hermen.db")),
            default_top_k=int(data.get("default_top_k", 6)),
            embedding=EmbeddingConfig(
                provider=str(embedding.get("provider", "sentence_transformers")),
                model=str(embedding.get("model", DEFAULT_EMBEDDING_MODEL)),
                dimensions=int(embedding.get("dimensions", 384)),
            ),
            query_model=QueryModelConfig(
                provider=str(query_model.get("provider", "llama_cpp")),
                model=str(query_model.get("model", "gemma-4-E2B-i1-Q4_K_M.gguf")),
                model_path=str(query_model.get("model_path", "")),
                base_url=str(query_model.get("base_url", "")),
                api_key_env=str(query_model.get("api_key_env", "OPENAI_API_KEY")),
                n_ctx=int(query_model.get("n_ctx", 8192)),
                n_gpu_layers=int(query_model.get("n_gpu_layers", -1)),
                temperature=float(query_model.get("temperature", 0.1)),
                max_tokens=int(query_model.get("max_tokens", 512)),
            ),
        )

    def save(self, path: Path) -> None:
        path.write_text(self.to_toml(), encoding="utf-8")

    def database_file(self, root: Path) -> Path:
        return root / self.database_path

    def to_toml(self) -> str:
        query_base = self.query_model.base_url.replace("\\", "\\\\").replace('"', '\\"')
        query_model_path = self.query_model.model_path.replace("\\", "\\\\").replace('"', '\\"')
        query_model_name = self.query_model.model.replace("\\", "\\\\").replace('"', '\\"')
        embedding_model = self.embedding.model.replace("\\", "\\\\").replace('"', '\\"')
        return (
            f"schema_version = {self.schema_version}\n"
            f'database_path = "{self.database_path}"\n'
            f"default_top_k = {self.default_top_k}\n\n"
            "[embedding]\n"
            f'provider = "{self.embedding.provider}"\n'
            f'model = "{embedding_model}"\n'
            f"dimensions = {self.embedding.dimensions}\n\n"
            "[query_model]\n"
            f'provider = "{self.query_model.provider}"\n'
            f'model = "{query_model_name}"\n'
            f'model_path = "{query_model_path}"\n'
            f'base_url = "{query_base}"\n'
            f'api_key_env = "{self.query_model.api_key_env}"\n'
            f"n_ctx = {self.query_model.n_ctx}\n"
            f"n_gpu_layers = {self.query_model.n_gpu_layers}\n"
            f"temperature = {self.query_model.temperature}\n"
            f"max_tokens = {self.query_model.max_tokens}\n"
        )


def default_config(model_path: str) -> ProjectConfig:
    config = ProjectConfig()
    config.query_model.model_path = model_path
    if model_path:
        config.query_model.model = Path(model_path).name
    return config
