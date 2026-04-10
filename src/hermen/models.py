from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Protocol

from hermen.config import QueryModelCapabilities, QueryModelConfig
from hermen.db import SearchResult


SYSTEM_PROMPT = """You answer questions using only the supplied company knowledge.
If the answer is not in the retrieved context, say so directly.
Always cite the source paths you used in a short Sources section."""


@dataclass(slots=True)
class RetrievalPlan:
    search_queries: list[str]
    keywords: list[str]
    top_k: int
    answer_strategy: str


class QueryModel(Protocol):
    def capabilities(self) -> QueryModelCapabilities:
        ...

    def plan_retrieval(
        self,
        question: str,
        history: list[dict[str, str]] | None,
        default_top_k: int,
    ) -> RetrievalPlan:
        ...

    def answer(
        self,
        question: str,
        context: list[SearchResult],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        ...

    def describe_image(self, image_path: str, prompt: str | None = None) -> str:
        ...


class EchoQueryModel:
    def __init__(self, config: QueryModelConfig, capabilities: QueryModelCapabilities) -> None:
        self._config = config
        self._capabilities = capabilities

    def capabilities(self) -> QueryModelCapabilities:
        return self._capabilities

    def plan_retrieval(
        self,
        question: str,
        history: list[dict[str, str]] | None,
        default_top_k: int,
    ) -> RetrievalPlan:
        rewritten = question
        if history:
            last_user_turn = next(
                (turn["content"] for turn in reversed(history) if turn["role"] == "user"),
                "",
            )
            if last_user_turn and question.lower().startswith(("what about", "and ", "how about")):
                rewritten = f"{last_user_turn}\nFollow-up: {question}"
        return RetrievalPlan(
            search_queries=[rewritten, question],
            keywords=[],
            top_k=default_top_k,
            answer_strategy="direct",
        )

    def answer(
        self,
        question: str,
        context: list[SearchResult],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        sources = ", ".join(sorted({item.source_path for item in context})) or "none"
        preview = "\n\n".join(item.text[:300] for item in context[:3]) or "No context found."
        return f"Question: {question}\n\nContext preview:\n{preview}\n\nSources: {sources}"

    def describe_image(self, image_path: str, prompt: str | None = None) -> str:
        if not self._capabilities.vision:
            raise RuntimeError("Vision is not enabled for the current query model")

        summary = f"Image asset from {Path(image_path).name}."
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                summary += f" Dimensions: {image.width}x{image.height}."
        except Exception:
            pass

        if prompt:
            summary += f" Prompt: {prompt}"
        return summary


class LlamaCppQueryModel:
    def __init__(self, config: QueryModelConfig, capabilities: QueryModelCapabilities) -> None:
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install with: uv pip install -e '.[local]'"
            ) from exc

        if not config.model_path:
            raise ValueError("query_model.model_path must be set for llama_cpp provider")

        self._capabilities = capabilities
        chat_handler = None
        if capabilities.vision and config.clip_model_path:
            chat_handler = Llava15ChatHandler(config.clip_model_path, verbose=False)

        with _silence_native_stderr():
            self._llama = Llama(
                model_path=config.model_path,
                n_ctx=config.n_ctx,
                n_gpu_layers=config.n_gpu_layers,
                chat_handler=chat_handler,
                verbose=False,
            )
        self._temperature = config.temperature
        self._max_tokens = config.max_tokens

    def capabilities(self) -> QueryModelCapabilities:
        return self._capabilities

    def plan_retrieval(
        self,
        question: str,
        history: list[dict[str, str]] | None,
        default_top_k: int,
    ) -> RetrievalPlan:
        standalone_question = _rewrite_followup_with_history(question, history)
        prompt = (
            "Rewrite this user request into one concise semantic search query for a vector database.\n"
            "Return only the rewritten query text.\n\n"
            f"User request: {standalone_question}\n"
            "Vector search query:"
        )
        with _silence_native_stderr():
            response = self._llama(
                prompt,
                temperature=0.2,
                max_tokens=64,
                stop=["\n", "</s>"],
            )
        generated = _clean_search_query(str(response["choices"][0]["text"]).strip())
        if not generated:
            generated = standalone_question

        keywords = _extract_keywords(question)
        if not keywords and standalone_question != question:
            keywords = _extract_keywords(standalone_question)
        keyword_query = " ".join(keywords[:4]) if keywords else ""
        search_queries = _dedupe_queries([generated, standalone_question, question])
        if keyword_query:
            search_queries = _dedupe_queries([*search_queries, keyword_query])
        return RetrievalPlan(
            search_queries=search_queries,
            keywords=keywords,
            top_k=default_top_k,
            answer_strategy="llm_query_rewrite",
        )

    def answer(
        self,
        question: str,
        context: list[SearchResult],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        prompt = _render_completion_prompt(question, context)
        with _silence_native_stderr():
            response = self._llama(
                prompt,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stop=["\nQuestion:", "\nContext:", "\nSource:", "<end_of_turn>", "</s>"],
            )
        choice = response["choices"][0]
        return _clean_answer(str(choice["text"]).strip())

    def describe_image(self, image_path: str, prompt: str | None = None) -> str:
        if not self._capabilities.vision:
            raise RuntimeError("Vision is not enabled for the current query model")

        content = [
            {"type": "text", "text": prompt or DEFAULT_IMAGE_PROMPT},
            {"type": "image_url", "image_url": {"url": Path(image_path).resolve().as_uri()}},
        ]
        with _silence_native_stderr():
            response = self._llama.create_chat_completion(
                messages=[{"role": "user", "content": content}],
                temperature=0.1,
                max_tokens=min(self._max_tokens, 384),
            )
        message = response["choices"][0]["message"]["content"]
        return _clean_answer(str(message).strip())


class OpenAICompatibleQueryModel:
    def __init__(self, config: QueryModelConfig, capabilities: QueryModelCapabilities) -> None:
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
        self._capabilities = capabilities

    def capabilities(self) -> QueryModelCapabilities:
        return self._capabilities

    def plan_retrieval(
        self,
        question: str,
        history: list[dict[str, str]] | None,
        default_top_k: int,
    ) -> RetrievalPlan:
        response = self._httpx.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "temperature": 0,
                "max_tokens": 256,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Return only JSON. Plan retrieval queries for a vector database."
                        ),
                    },
                    {
                        "role": "user",
                        "content": _render_retrieval_plan_prompt(question, history, default_top_k),
                    },
                ],
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        text = str(data["choices"][0]["message"]["content"]).strip()
        return _parse_retrieval_plan(text, question, default_top_k)

    def answer(
        self,
        question: str,
        context: list[SearchResult],
        history: list[dict[str, str]] | None = None,
    ) -> str:
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

    def describe_image(self, image_path: str, prompt: str | None = None) -> str:
        if not self._capabilities.vision:
            raise RuntimeError("Vision is not enabled for the current query model")

        response = self._httpx.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "temperature": 0.1,
                "max_tokens": min(self._max_tokens, 384),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt or DEFAULT_IMAGE_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": Path(image_path).resolve().as_uri()},
                            },
                        ],
                    }
                ],
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"]).strip()


def build_query_model(
    config: QueryModelConfig,
    capabilities: QueryModelCapabilities | None = None,
) -> QueryModel:
    resolved_capabilities = capabilities or infer_query_model_capabilities(config)
    if config.provider == "echo":
        return EchoQueryModel(config, resolved_capabilities)
    if config.provider == "llama_cpp":
        return LlamaCppQueryModel(config, resolved_capabilities)
    if config.provider == "openai_compatible":
        return OpenAICompatibleQueryModel(config, resolved_capabilities)
    raise ValueError(f"Unsupported query model provider: {config.provider}")


def infer_query_model_capabilities(config: QueryModelConfig) -> QueryModelCapabilities:
    if config.provider == "llama_cpp":
        return QueryModelCapabilities(
            text=True,
            vision=bool(config.clip_model_path and Path(config.clip_model_path).exists()),
            audio=False,
        )

    return QueryModelCapabilities(text=True, vision=False, audio=False)


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


DEFAULT_IMAGE_PROMPT = (
    "Describe this image for retrieval in a company knowledge base. "
    "Mention visible text, labels, tables, charts, diagrams, entities, and the main semantic content. "
    "Be concise but information-dense."
)


def _render_retrieval_plan_prompt(
    question: str,
    history: list[dict[str, str]] | None,
    default_top_k: int,
) -> str:
    history_text = _render_history(history)
    return (
        "You are planning retrieval for a vector database.\n"
        "Rewrite the user's request into semantic search queries that will help find the best chunks.\n"
        "If the request is a follow-up, use the conversation history to make the search queries standalone.\n"
        "Return only plain text in exactly this format:\n"
        "SEARCH_QUERY_1: ...\n"
        "SEARCH_QUERY_2: ...\n"
        "KEYWORDS: keyword 1, keyword 2\n"
        f"TOP_K: {default_top_k}\n"
        "ANSWER_STRATEGY: direct\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"User question:\n{question}\n\n"
        f"Preferred top_k: {default_top_k}\n"
    )


def _clean_answer(text: str) -> str:
    cleaned = text.replace("<end_of_turn>", "").replace("</s>", "").strip()
    return cleaned


def _clean_search_query(text: str) -> str:
    cleaned = _clean_answer(text).strip()
    prefixes = [
        "vector search query:",
        "search query:",
        "query:",
        "rewrite:",
    ]
    lowered = cleaned.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            lowered = cleaned.lower()
    return cleaned.strip(" \"'")


def _render_history(history: list[dict[str, str]] | None) -> str:
    if not history:
        return "No prior conversation."
    rendered: list[str] = []
    for turn in history[-6:]:
        role = turn.get("role", "user")
        content = turn.get("content", "").strip()
        if content:
            rendered.append(f"{role}: {content}")
    return "\n".join(rendered) or "No prior conversation."


def _parse_retrieval_plan(text: str, question: str, default_top_k: int) -> RetrievalPlan:
    line_plan = _parse_line_retrieval_plan(text, question, default_top_k)
    if line_plan is not None:
        return line_plan

    try:
        data = json.loads(_extract_json_object(text))
    except Exception:
        return RetrievalPlan(
            search_queries=[question],
            keywords=[],
            top_k=default_top_k,
            answer_strategy="fallback_raw_query",
        )

    search_queries = [str(item).strip() for item in data.get("search_queries", []) if str(item).strip()]
    keywords = [str(item).strip() for item in data.get("keywords", []) if str(item).strip()]
    top_k = int(data.get("top_k", default_top_k))
    top_k = max(1, min(top_k, 12))
    answer_strategy = str(data.get("answer_strategy", "direct")).strip() or "direct"

    if not search_queries:
        search_queries = [question]

    return RetrievalPlan(
        search_queries=search_queries[:4],
        keywords=keywords[:12],
        top_k=top_k,
        answer_strategy=answer_strategy,
    )


def _parse_line_retrieval_plan(
    text: str,
    question: str,
    default_top_k: int,
) -> RetrievalPlan | None:
    search_queries: list[str] = []
    keywords: list[str] = []
    top_k = default_top_k
    answer_strategy = "direct"

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip().upper()
        value = value.strip()
        if key.startswith("SEARCH_QUERY") and value:
            search_queries.append(value)
        elif key == "KEYWORDS" and value:
            keywords = [item.strip() for item in value.split(",") if item.strip()]
        elif key == "TOP_K" and value.isdigit():
            top_k = max(1, min(int(value), 12))
        elif key == "ANSWER_STRATEGY" and value:
            answer_strategy = value

    if not search_queries:
        return None

    return RetrievalPlan(
        search_queries=search_queries[:4],
        keywords=keywords[:12],
        top_k=top_k,
        answer_strategy=answer_strategy,
    )


def _rewrite_followup_with_history(question: str, history: list[dict[str, str]] | None) -> str:
    if not history:
        return question

    lowered = question.lower().strip()
    looks_follow_up = lowered.startswith(
        ("what about", "how about", "and ", "what else", "those", "them", "it")
    )
    if not looks_follow_up:
        return question

    prior_user_turns = [turn["content"].strip() for turn in history if turn.get("role") == "user"]
    if not prior_user_turns:
        return question

    previous = prior_user_turns[-1]
    return f"{previous} {question}".strip()


def _extract_keywords(text: str) -> list[str]:
    stopwords = {
        "about",
        "a",
        "an",
        "and",
        "are",
        "data",
        "for",
        "how",
        "in",
        "is",
        "many",
        "much",
        "of",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "were",
        "which",
        "used",
    }
    keywords: list[str] = []
    for token in text.replace("?", " ").replace(",", " ").split():
        normalized = token.strip().lower()
        if len(normalized) < 3 or normalized in stopwords:
            continue
        if normalized not in keywords:
            keywords.append(normalized)
    return keywords[:8]


def _dedupe_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        normalized = " ".join(query.lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(query.strip())
    return unique[:4]


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("Unterminated JSON object")


@contextmanager
def _silence_native_stderr():
    stderr_fd = None
    devnull_fd = None
    try:
        stderr_fd = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        if stderr_fd is not None:
            os.dup2(stderr_fd, 2)
            os.close(stderr_fd)
        if devnull_fd is not None:
            os.close(devnull_fd)
