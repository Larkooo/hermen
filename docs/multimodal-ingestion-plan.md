# Multimodal Ingestion Plan

## Goal

Extend `hermen` so projects initialized with a vision-capable model can ingest
images and PDF-embedded images as first-class knowledge sources. Hermen should
use the configured model during ingest to generate semantic records for visual
content, then store those records in the vector database for later retrieval.

The database remains text-and-metadata centric:

- raw assets are referenced, not embedded directly into the main chunk table
- image semantics are persisted as text records plus structured metadata
- retrieval can combine text chunks and image-derived semantic chunks

## Product Behavior

### Init

`hermen init` should detect or accept query-model capabilities and persist them
in `hermen.toml`.

Example:

```toml
[query_model]
provider = "llama_cpp"
model = "gemma-4-E2B-i1-Q4_K_M.gguf"
model_path = "/absolute/path/to/model.gguf"

[query_model.capabilities]
text = true
vision = true
audio = false
```

Rules:

- `vision=false`: Hermen skips multimodal inference during ingest
- `vision=true`: Hermen enables image-aware ingestion
- capability detection must be based on the runtime/provider path, not just the
  model name
- users must be able to override auto-detection manually

### Ingest

When `vision=true`, `hermen ingest` should:

- ingest standalone image files such as `png`, `jpg`, `jpeg`, `webp`
- extract embedded images from PDFs
- optionally rasterize PDF pages for page-level visual summaries later
- run multimodal inference during ingest to create semantic records
- store those records alongside normal text chunks

When `vision=false`, Hermen should still:

- ingest normal document text
- optionally extract image assets and metadata for future re-processing
- avoid pretending that visual understanding occurred

### Querying

`ask` and `chat` should retrieve from both:

- text chunks
- image-derived semantic chunks

The user should not need a separate query mode. If an answer depends on a chart,
diagram, or scanned figure, Hermen should be able to retrieve the semantic
record created during ingest.

## Ingestion Architecture

### 1. Capability Detection

Add a capability probe layer at project initialization.

Responsibilities:

- inspect configured provider and runtime
- decide whether text, vision, or audio inputs are actually supported
- persist capability flags in config

Suggested interface:

```python
@dataclass(slots=True)
class ModelCapabilities:
    text: bool
    vision: bool
    audio: bool
```

Detection should be provider-specific:

- `llama_cpp`: check whether the configured model/runtime can handle images for
  that exact model path
- `openai_compatible`: optionally probe model metadata if the endpoint exposes
  it, otherwise require explicit user configuration
- fallback: default to conservative values

### 2. Asset Extraction

Introduce an asset extraction layer before chunking.

Asset types:

- document text
- standalone image files
- PDF embedded images
- optionally rendered PDF pages

Output:

```python
@dataclass(slots=True)
class ExtractedAsset:
    asset_id: str
    source_path: str
    asset_type: str  # text, image, pdf_image, pdf_page
    page_number: int | None
    image_index: int | None
    mime_type: str | None
    file_path: str | None
    text: str | None
    metadata: dict[str, object]
```

### 3. Multimodal Semantic Extraction

For image-bearing assets, Hermen should run an ingest-time semantic pass.

Expected outputs:

- concise caption
- OCR text if present
- figure/table/chart summary
- detected entities or labels
- keywords
- confidence or extraction mode metadata

Persist these as semantic records, for example:

```python
@dataclass(slots=True)
class SemanticRecord:
    source_asset_id: str
    record_type: str  # raw_text, caption, ocr, figure_summary, page_summary
    text: str
    metadata: dict[str, object]
```

This is the core design choice:

- Hermen stores multimodal understanding as retrievable semantic text
- retrieval stays language-based
- the original image remains referenceable for citations and future upgrades

### 4. Embedding Strategy

Embed all semantic records, not just raw extracted text.

For example:

- raw PDF text chunk
- image OCR chunk
- image caption chunk
- page summary chunk

This allows one user question to match:

- literal text from a page
- language generated from a chart or figure
- abstract summaries created during ingest

### 5. Storage Changes

Add explicit asset and semantic-record tables rather than overloading the
existing chunk schema.

Suggested tables:

- `documents`
- `assets`
- `semantic_records`
- `embeddings`

Possible schema:

```sql
CREATE TABLE assets (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL,
  asset_type TEXT NOT NULL,
  source_path TEXT NOT NULL,
  page_number INTEGER,
  image_index INTEGER,
  mime_type TEXT,
  file_path TEXT,
  metadata_json TEXT NOT NULL,
  FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE semantic_records (
  id TEXT PRIMARY KEY,
  asset_id TEXT NOT NULL,
  record_type TEXT NOT NULL,
  text TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
);
```

The current chunk table can remain temporarily for backward compatibility, but
the medium-term goal should be one unified retrieval record layer.

## Retrieval Behavior

Raw `search` can remain the low-level retrieval primitive.

`ask` and `chat` should retrieve across all semantic record types and expose
their origin in metadata:

- source file path
- page number
- image index
- asset type
- record type

Ranking should favor:

- semantically relevant matches
- diversity across source assets
- text and image-derived evidence together

## Provider Responsibilities

### Llama/GGUF Path

If the configured local runtime truly supports vision, Hermen should use it for
multimodal extraction during ingest.

If it does not, Hermen must:

- mark `vision=false`
- skip multimodal inference
- optionally preserve extracted assets for future re-ingest with a better
  runtime

### OpenAI-Compatible Path

If the endpoint accepts image inputs, Hermen can use it for captioning and page
understanding during ingest. This should be behind the same capability contract.

## Phased Rollout

### Phase 1: Capability and Schema

- add provider capability detection
- persist model capabilities in config
- add asset and semantic-record schema
- keep existing text-only behavior intact

Acceptance:

- a project can declare whether vision is available
- Hermen can store extracted non-text assets without retrieval regressions

### Phase 2: Image and PDF Asset Extraction

- ingest standalone images
- extract embedded images from PDFs
- persist asset metadata and filesystem references

Acceptance:

- `hermen ingest some.pdf image.png` creates asset rows and references
- no semantic inference required yet

### Phase 3: Multimodal Semantic Ingest

- run model inference over visual assets during ingest
- create captions, OCR text, page summaries, and figure summaries
- embed and store semantic records

Acceptance:

- image-derived records appear in retrieval results
- answers can cite PDF pages and images

### Phase 4: Retrieval Fusion

- retrieve across text chunks and visual semantic records
- tune fusion and ranking
- surface record-type/source metadata in CLI outputs

Acceptance:

- `ask` can answer questions that depend on figure/chart content
- search results clearly identify whether evidence came from text or image

### Phase 5: Re-index and Upgrade Tooling

- add `hermen reindex --with-vision`
- allow upgrading old projects after switching to a vision-capable model
- optionally cache expensive multimodal ingest outputs

Acceptance:

- existing projects can be upgraded without rebuilding everything manually

## Operational Considerations

### Cost and Latency

Multimodal ingest will be much slower than plain text ingest. Hermen should:

- support batching where the provider allows it
- cache semantic outputs by asset hash
- skip unchanged assets on re-ingest

### Determinism

Ingest-time semantic extraction affects retrieval quality. Hermen should store:

- the provider
- the model identifier
- extraction settings
- asset hash

This makes results explainable and reindexing reproducible.

### Citations

Every semantic record should preserve provenance:

- original file path
- page number
- image index
- extraction type

This is required so answers can cite the correct source even when the matched
text was generated during ingest rather than extracted verbatim.

## Open Questions

- which exact local runtime path should be treated as vision-capable for the
  current Gemma GGUF setup
- whether page rasterization should be on by default for PDFs
- whether OCR should be a separate fallback pipeline or always included in the
  multimodal pass
- whether to unify `chunks` and `semantic_records` immediately or migrate in two
  steps

## Recommendation

Implement this in phases, starting with capability detection and explicit asset
storage. The key product win is not “store images in the DB”; it is “turn visual
content into retrievable semantic knowledge during ingest.” That keeps Hermen’s
database semantics clean while making multimodal models materially useful.
