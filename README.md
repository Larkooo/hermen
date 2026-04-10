# hermen

`hermen` is a local-first vector database with a built-in model query engine.
It stores documents, chunks, embeddings, and metadata in SQLite, then layers
retrieval and generation on top so the database can answer questions directly.

The default local query path is a GGUF model through `llama.cpp`. In this repo,
the intended base model is Gemma 4 E2B.

## What it does

- Index plain-text company knowledge into a self-contained database
- Run semantic retrieval over stored chunks
- Answer questions with retrieved context through a local or remote model
- Keep model selection configurable per project instead of hard-coding one model

## Architecture

- Storage: SQLite for documents, chunks, metadata, and embeddings
- Retrieval: cosine similarity over persisted embeddings
- Query engine: pluggable providers with `llama.cpp` local GGUF support by default
- CLI: `init`, `ingest`, `search`, `ask`, `chat`, `stats`

This is intentionally local-first and simple. For very large corpora, you would
replace the brute-force vector scan with an ANN index later, but the product
surface does not need to change.

## Quick start

`llama-cpp-python` and `sentence-transformers` are best used with Python 3.11 or
3.12 today, so this project targets `>=3.11,<3.14`.

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[local,dev]"
```

Initialize a database with your local Gemma GGUF model:

```bash
hermen init \
  --model-path "/Users/nas/Library/Application Support/app.cotypist.Cotypist/Models/gemma-4-E2B-i1-Q4_K_M.gguf"
```

Index a directory:

```bash
hermen ingest ./docs ./handbook.md
```

Run retrieval only:

```bash
hermen search "What is our onboarding process?"
```

Run retrieval plus generation:

```bash
hermen ask "What is our onboarding process?"
```

Open an interactive session:

```bash
hermen chat
```

## Model selection

`hermen` treats the database and the query model as separate concerns:

- The database always owns persistence, chunking, embeddings, and retrieval
- The query model synthesizes an answer over retrieved context

Supported query providers in this repo:

- `llama_cpp`: local GGUF models through `llama.cpp`
- `openai_compatible`: any OpenAI-compatible chat endpoint
- `echo`: a built-in debug provider for tests and smoke checks

Supported embedding providers:

- `sentence_transformers`: local embedding models
- `hash`: deterministic no-dependency fallback for tests and quick demos

Users can swap models by editing `hermen.toml` or overriding flags on `init`.

## Config

`hermen init` writes `hermen.toml` and `.hermen/hermen.db`.

Example:

```toml
schema_version = 1
database_path = ".hermen/hermen.db"
default_top_k = 6

[embedding]
provider = "sentence_transformers"
model = "sentence-transformers/all-MiniLM-L6-v2"
dimensions = 384

[query_model]
provider = "llama_cpp"
model = "gemma-4-E2B-i1-Q4_K_M.gguf"
model_path = "/absolute/path/to/model.gguf"
base_url = ""
api_key_env = "OPENAI_API_KEY"
n_ctx = 8192
n_gpu_layers = -1
temperature = 0.1
max_tokens = 512
```

## Development

Run tests:

```bash
pytest
```

Run lint:

```bash
ruff check .
```
