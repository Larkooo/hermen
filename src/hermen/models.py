from __future__ import annotations

import os
from typing import Protocol

from hermen.config import QueryModelConfig
from hermen.db import SearchResult


SYSTEM_PROMPT = """You answer questions using only the supplied company knowledge.
If the answer is not in the retrieved context, say so directly.
Always cite the source paths you used in a short Sources section."""


class QueryModel(Protocol):
    def answer(self, question: str, context: list[SearchResult]) -> str:
        ...


class EchoQueryModel:
    def answer(self, question: str, context: list[SearchResult]) -> str:
        sources = ", ".join(sorted({item.source_path for item in context})) or "none"
        preview = "\n\n".join(item.text[:300] for item in context[:3]) or "No context found."
        return f"Question: {question}\n\nContext preview:\n{preview}\n\nSources: {sources}"


class LlamaCppQueryModel:
    def __init__(self, config: QueryModelConfig) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install with: uv pip install -e '.[local]'"
            ) from exc

        if not config.model_path:
            raise ValueError("query_model.model_path must be set for llama_cpp provider")

        self._llama = Llama(
            model_path=config.model_path,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            verbose=False,
        )
        self._temperature = config.temperature
        self._max_tokens = config.max_tokens

    def answer(self, question: str, context: list[SearchResult]) -> str:
        prompt = _render_completion_prompt(question, context)
        response = self._llama(
            prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stop=["\nQuestion:", "\nContext:", "\nSource:", "<end_of_turn>", "</s>"],
        )
        choice = response["choices"][0]
        return _clean_answer(str(choice["text"]).strip())


class OpenAICompatibleQueryModel:
    def __init__(self, config: QueryModelConfig) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "httpx is not installed. Install with: uv pip install -e '.[remote]'"
            ) from exc

        if not config.base_url:
            raise ValueError("query_model.base_url must be set for openai_compatible provider")

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Environment variable {config.api_key_env} is not set")

        self._httpx = httpx
        self._base_url = config.base_url.rstrip("/")
        self._api_key = api_key
        self._model = config.model
        self._temperature = config.temperature
        self._max_tokens = config.max_tokens

    def answer(self, question: str, context: list[SearchResult]) -> str:
        response = self._httpx.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _render_context_prompt(question, context)},
                ],
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"]).strip()


def build_query_model(config: QueryModelConfig) -> QueryModel:
    if config.provider == "echo":
        return EchoQueryModel()
    if config.provider == "llama_cpp":
        return LlamaCppQueryModel(config)
    if config.provider == "openai_compatible":
        return OpenAICompatibleQueryModel(config)
    raise ValueError(f"Unsupported query model provider: {config.provider}")


def _render_context_prompt(question: str, context: list[SearchResult]) -> str:
    sections: list[str] = []
    for item in context:
        sections.append(
            f"Source: {item.source_path}#chunk-{item.chunk_index}\n"
            f"Score: {item.score:.4f}\n"
            f"{item.text}"
        )
    rendered_context = "\n\n---\n\n".join(sections) if sections else "No context retrieved."
    return f"Question:\n{question}\n\nContext:\n{rendered_context}"


def _render_completion_prompt(question: str, context: list[SearchResult]) -> str:
    rendered_context = "\n\n---\n\n".join(
        f"Source: {item.source_path}#chunk-{item.chunk_index}\n{item.text}" for item in context
    ) or "No context retrieved."
    return (
        "Use only the provided context to answer the question in 1-3 sentences. "
        "If the answer is not present in the context, say so.\n\n"
        f"Context:\n{rendered_context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _clean_answer(text: str) -> str:
    cleaned = text.replace("<end_of_turn>", "").replace("</s>", "").strip()
    return cleaned
