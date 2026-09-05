from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
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
    clip_model_path: str = ""
    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    n_ctx: int = 8192
    n_gpu_layers: int = -1
    temperature: float = 0.1
    max_tokens: int = 512


@dataclass(slots=True)
class QueryModelCapabilities:
    text: bool = True
    vision: bool = False
    audio: bool = False


@dataclass(slots=True)
class ProjectConfig:
    schema_version: int = 1
    database_path: str = ".hermen/hermen.db"
    default_top_k: int = 6
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    query_model: QueryModelConfig = field(default_factory=QueryModelConfig)
    query_model_capabilities: QueryModelCapabilities = field(default_factory=QueryModelCapabilities)

    @classmethod
    def load(cls, path: Path) -> "ProjectConfig":
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        embedding = data.get("embedding", {})
        query_model = data.get("query_model", {})
        capabilities = query_model.get("capabilities", {})
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
                clip_model_path=str(query_model.get("clip_model_path", "")),
                base_url=str(query_model.get("base_url", "")),
                api_key_env=str(query_model.get("api_key_env", "OPENAI_API_KEY")),
                n_ctx=int(query_model.get("n_ctx", 8192)),
                n_gpu_layers=int(query_model.get("n_gpu_layers", -1)),
                temperature=float(query_model.get("temperature", 0.1)),
                max_tokens=int(query_model.get("max_tokens", 512)),
            ),
            query_model_capabilities=QueryModelCapabilities(
                text=bool(capabilities.get("text", True)),
                vision=bool(capabilities.get("vision", False)),
                audio=bool(capabilities.get("audio", False)),
            ),
        )

    def save(self, path: Path) -> None:
        path.write_text(self.to_toml(), encoding="utf-8")

    def database_file(self, root: Path) -> Path:
        return root / self.database_path

    def to_toml(self) -> str:
        return (
            f"schema_version = {self.schema_version}\n"
            f"database_path = {json.dumps(self.database_path, ensure_ascii=False)}\n"
            f"default_top_k = {self.default_top_k}\n"
            "\n[embedding]\n"
            f"provider = {json.dumps(self.embedding.provider, ensure_ascii=False)}\n"
            f"model = {json.dumps(self.embedding.model, ensure_ascii=False)}\n"
            f"dimensions = {self.embedding.dimensions}\n"
            "\n[query_model]\n"
            f"provider = {json.dumps(self.query_model.provider, ensure_ascii=False)}\n"
            f"model = {json.dumps(self.query_model.model, ensure_ascii=False)}\n"
            f"model_path = {json.dumps(self.query_model.model_path, ensure_ascii=False)}\n"
            f"clip_model_path = {json.dumps(self.query_model.clip_model_path, ensure_ascii=False)}\n"
            f"base_url = {json.dumps(self.query_model.base_url, ensure_ascii=False)}\n"
            f"api_key_env = {json.dumps(self.query_model.api_key_env, ensure_ascii=False)}\n"
            f"n_ctx = {self.query_model.n_ctx}\n"
            f"n_gpu_layers = {self.query_model.n_gpu_layers}\n"
            f"temperature = {self.query_model.temperature}\n"
            f"max_tokens = {self.query_model.max_tokens}\n"
            "\n[query_model.capabilities]\n"
            f"text = {str(self.query_model_capabilities.text).lower()}\n"
            f"vision = {str(self.query_model_capabilities.vision).lower()}\n"
            f"audio = {str(self.query_model_capabilities.audio).lower()}\n"
        )


def default_config(model_path: str) -> ProjectConfig:
    config = ProjectConfig()
    config.query_model.model_path = model_path
    if model_path:
        config.query_model.model = Path(model_path).name
    return config
