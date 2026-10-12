# RAG FastAPI

A complete **Retrieval-Augmented Generation** API built with **FastAPI** and **ChromaDB**.

Ingest documents, then ask questions. Answers are generated *only* from the text
retrieved from your own documents, and every answer comes back with the sources
it was built from.

---

## Table of contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [The pipeline, stage by stage](#the-pipeline-stage-by-stage)
- [API reference](#api-reference)
- [Choosing the LLM](#choosing-the-llm)
- [Configuration](#configuration)
- [Testing](#testing)
- [What to learn next](#what-to-learn-next)
- [Troubleshooting](#troubleshooting)

---

## Quick start

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create the sample documents into the vector store
python -m scripts.ingest

# 3. Start the API
python main.py
# or: uvicorn app.main:app --reload
```

Then open <http://127.0.0.1:8000/docs> for interactive API documentation
(Swagger UI), and try:

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/query `
  -H "Content-Type: application/json" `
  -d '{\"question\": \"What does the Team plan cost?\", \"top_k\": 3}'
```

> **Note:** the query endpoint needs an LLM. The retrieval half works standalone, but
> the generation half will return `503` until a model is reachable. See
> [Choosing the LLM](#choosing-the-llm).

---

## How it works

Without RAG, a language model can only answer from what it memorised during
training. It cannot know your internal docs, and its knowledge has a cutoff date.

RAG fixes this by fetching the relevant text **first** and handing it to the
model as context:

```
                     ┌──────────────────────────────────────────────┐
                     │            INGESTION (once)                  │
                     │                                              │
  documents ───────► │  chunk  ──►  embed  ──►  store  ──►  ChromaDB│
                     └──────────────────────────────────────────────┘

                     ┌──────────────────────────────────────────────┐
                     │             QUERY (per question)             │
                     │                                              │
  question ────────► │  embed  ──►  search  ──►  prompt ──► LLM ►  │  answer
                     │              ChromaDB          (context)      │  + sources
                     └──────────────────────────────────────────────┘
```

The critical idea: **the model never invents facts, it summarises text you
gave it.** That is why every response also returns the chunks that were used.

---

## Project structure

```
rag-fastapi/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app factory + startup (ASGI entrypoint)
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py           # HTTP endpoints: thin, validation only
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py           # Settings, loaded from .env / environment
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── rag.py              # Pydantic request/response models
│   └── services/
│       ├── __init__.py
│       ├── chunking.py         # Text -> overlapping chunks
│       ├── embeddings.py       # Text -> vectors (sentence-transformers)
│       ├── vectorstore.py      # ChromaDB: add / query / delete / list
│       ├── llm.py              # OpenAI-compatible chat completions
│       └── pipeline.py         # Orchestrates the 5 stages
├── data/
│   └── documents/              # Sample documents (.txt / .md)
├── scripts/
│   ├── __init__.py
│   └── ingest.py               # CLI: python -m scripts.ingest
├── tests/
│   └── test_pipeline.py        # Unit + API tests (all offline-safe)
├── .env.example                # Copy to .env and edit
├── main.py                     # Convenience launcher: python main.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

### Why this layout

This is called **layered architecture**, and it is the most common structure for
production FastAPI services. Each layer has one job:

| Layer | Responsibility | Depends on |
|---|---|---|
| `schemas/` | The API contract. Validates input, shapes output. | nothing |
| `core/` | Cross-cutting config. | nothing |
| `services/` | All the business logic. | `core`, `schemas` |
| `api/` | HTTP concerns only. | everything above |

The important rule: **dependencies point downwards, never upwards.** `services/`
never imports from `api/`. That is what makes the code testable without HTTP.

Two consequences worth internalising:

1. **Routes stay thin.** A route should validate input, call a service, and
   shape a response. If a route contains an `if` about *how* something works,
   that logic belongs in a service.
2. **Services are reusable.** Because `pipeline.py` does not know HTTP exists,
   `scripts/ingest.py` reuses the exact same ingestion code path as the API.
   One implementation, two entrypoints.

---

## The pipeline, stage by stage

### 1. Chunking — `app/services/chunking.py`

Raw text is too large to embed well, so it is split into **chunks** of ~500
characters. A good chunk is small enough to embed precisely but large enough to
carry meaning on its own.

The splitter is **recursive**: it tries natural boundaries first, in order —
paragraph, newline, sentence, whitespace, and only then a hard character cut:

```
"\n\n"  ──►  "\n"  ──►  ". "  ──►  " "  ──►  hard cut
```

This keeps sentences intact instead of slicing mid-word, which is what a
naive fixed-width splitter does.

**Overlap** repeats the tail of one chunk at the start of the next. If a
sentence that answers a question is cut in half across two chunks, both halves
get embedded without their other half and neither retrieves well. Overlapping
by ~10-20% guarantees the sentence appears complete in at least one chunk.

The overlap tail is snapped to a word or sentence boundary, so chunks never
begin with a fragment like `"les. Plain values"`.

### 2. Embedding — `app/services/embeddings.py`

An **embedding model** converts text into a vector of numbers, here 384 floats
from `all-MiniLM-L6-v2`. The key property: **texts with similar meaning get
vectors pointing in similar directions.**

Two things matter here:

- The **same model** must embed the documents and the question. A vector from
  model A means nothing to model B.
- The model is loaded once and cached with `@lru_cache`. Loading a transformer
  takes seconds and hundreds of MB of RAM, so doing it per request would be
  painfully slow.

### 3. Storage — `app/services/vectorstore.py`

**ChromaDB** stores each vector alongside its original text and metadata, and
persists to disk so the index survives a restart.

Two design points that are easy to get wrong:

**We compute embeddings ourselves** and pass them in, rather than letting
ChromaDB embed. This means we control which model is used, which matters
because you must use the same one at query time.

**The collection uses cosine distance.** We explicitly create the collection
with `{"hnsw:space": "cosine"}`. ChromaDB's *default* is L2, which returns
*squared* Euclidean distance. For unit vectors that equals `2 - 2Â·cosine`,
so converting it with `1 - distance` produces meaningless negative scores.
With cosine distance, `1 - distance` is a true cosine similarity.

> `get_or_create_collection` never changes the settings of an existing
> collection. If a collection already exists with a different metric, the code
> detects it and rebuilds it, so scores stay correct.

**Metadata** (`source`, `document_id`, `chunk_index`, `characters`,
`ingested_at`) is stored with every chunk. This is what enables filtering to
one document, showing citations, and deleting a single document without
rebuilding the index.

### 4. Retrieval — `similarity_search()`

The question is embedded with the same model, and ChromaDB returns the chunks
whose vectors are closest.

**Cosine distance** ranges from 0 (identical) to 2 (opposite), so:

```
similarity = 1 - distance
```

Empirically, scores above ~0.7 mean the chunk is genuinely on topic; below ~0.4
means it probably should not have been retrieved.

**`top_k`** controls how many chunks are sent to the model. Too few and the
answer is grounded in incomplete information. Too many and the model is
distracted by irrelevant text, the prompt grows costly, and the answer gets
vaguer. **Three to six is the usual sweet spot.**

### 5. Generation — `app/services/llm.py`

The retrieved chunks are inserted into a prompt and the LLM is told to answer
*using only that context*:

```
CONTEXT:
[1] The Team plan costs 45 USD per month...
[2] Private networking through a VPC is...

QUESTION: How much does the Team plan cost?
```

The system prompt is what makes this RAG rather than just search:

```
1. Answer using ONLY the information contained in the CONTEXT section.
2. If the context does not contain the answer, reply exactly:
   "I don't have enough information..."
3. Be concise and factual. Do not speculate or add outside knowledge.
4. Cite information with [1], [2], ...
```

Rule 2 is the important one: it lets the system **admit ignorance** instead of
hallucinating. Rules 1 and 4 force grounding and traceability.

The client speaks the OpenAI `/chat/completions` protocol, which is a de-facto
standard. Changing provider is a config change, not a code change.

---

## API reference

Base prefix: `/api/v1` (set by `API_PREFIX`)

### `GET /health`

Service status and what is currently indexed.

```json
{
  "status": "ok",
  "app_name": "RAG FastAPI",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "llm_model": "llama3.2",
  "chroma_collection": "rag_documents",
  "documents_indexed": 2
}
```

### `POST /query`

Answer a question with retrieval-augmented generation.

```json
// request
{
  "question": "What does the Team plan cost?",
  "top_k": 3,
  "source_filter": null,
  "temperature": 0.2,
  "include_sources": true
}
```

```json
// response
{
  "question": "What does the Team plan cost?",
  "answer": "The Team plan costs 45 USD per month [1].",
  "model": "llama3.2",
  "sources": [
    {
      "id": "a1b2c3:0",
      "text": "The Team plan costs 45 USD per month...",
      "score": 0.6217,
      "source": "acme_cloud_guide.txt",
      "chunk_index": 0,
      "metadata": {}
    }
  ],
  "context_used": "CONTEXT:\n[1] The Team plan costs...",
  "latency_ms": 339.07
}
```

Returns `503` with a clear message if the LLM is unreachable.

### `POST /documents`

Ingest raw text. Re-ingesting the same `source` replaces the old chunks rather
than duplicating them.

```json
// request
{
  "text": "The Team plan costs 45 USD per month...",
  "source": "pricing.txt",
  "metadata": {"team": "docs"}
}
```

```json
// response (201)
{
  "source": "pricing.txt",
  "chunks_created": 3,
  "characters_processed": 1840,
  "chunk_ids": ["d58e9aaf:0", "d58e9aaf:1", "d58e9aaf:2"],
  "ingested_at": "2026-10-12T06:08:16.659318Z"
}
```

### `POST /documents/upload`

Same, but as `multipart/form-data` with a `.txt` or `.md` file (max 5 MB):

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/documents/upload `
  -F "file=@data\documents\acme_cloud_guide.txt"
```

### `GET /documents`

List indexed documents with chunk counts.

### `DELETE /documents/{source}`

Delete one document and all of its chunks. `404` if it does not exist.

---

## Choosing the LLM

The default config expects **Ollama**, which runs models locally for free.

### Option A: Ollama (local, free, recommended for learning)

```powershell
# 1. Install from https://ollama.com

# 2. Pull a model
ollama pull llama3.2

# 3. Make sure it is running (starts automatically after install)
ollama serve
```

The defaults in `.env.example` already point at Ollama, so nothing else is needed.
Smaller models work better on limited hardware: `llama3.2` (3B), `qwen2.5:3b`.

### Option B: OpenAI (cloud, paid)

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

### Option C: LM Studio

```env
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL=<the model you loaded>
```

### Option D: vLLM (self-hosted)

```env
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=EMPTY
LLM_MODEL=<served model name>
```

Anything speaking the OpenAI protocol works.

### Free vs paid trade-off

| | Ollama | OpenAI |
|---|---|---|
| Cost | Free | Per token |
| Privacy | Fully local | Data leaves your machine |
| Speed | Depends on your hardware | Fast, consistent |
| Setup | Install + ~4 GB download | API key |

For learning, **Ollama is the better choice**: you can experiment freely and see
exactly what the model receives.

---

## Configuration

Every setting is an environment variable that can also live in a `.env` file:

```powershell
Copy-Item .env.example .env
```

`.env` is gitignored, so secrets never get committed.

| Variable | Default | Meaning |
|---|---|---|
| `API_PREFIX` | `/api/v1` | URL prefix for all routes |
| `CORS_ORIGINS` | `*` | Allowed origins, comma-separated |
| `CHROMA_PERSIST_DIRECTORY` | `./chroma_store` | Where the index lives on disk |
| `CHROMA_COLLECTION` | `rag_documents` | Collection name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` |
| `CHUNK_SIZE` | `500` | Target characters per chunk |
| `CHUNK_OVERLAP` | `50` | Characters repeated between chunks |
| `RETRIEVAL_TOP_K` | `4` | Chunks retrieved by default |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `LLM_API_KEY` | `ollama` | API key for the LLM |
| `LLM_MODEL` | `llama3.2` | Model name |
| `LLM_TEMPERATURE` | `0.2` | Low = focused, high = creative |
| `LLM_MAX_TOKENS` | `1024` | Max tokens in the answer |
| `LLM_TIMEOUT_SECONDS` | `120` | Request timeout |

**Changing `CHUNK_SIZE`, `CHUNK_OVERLAP` or `EMBEDDING_MODEL` invalidates the
existing index.** Delete `chroma_store/` and re-ingest:

```powershell
Remove-Item -Recurse chroma_store; python -m scripts.ingest
```

### Tuning tip

Absolute scores are meaningless across models; what matters is *relative* order.
Ingest a document, run a few questions with `top_k: 5`, and **actually read the
returned chunks**. If the answer is not in the retrieved text, retune chunking
before touching anything else.

---

## Testing

```powershell
$env:PYTHONPATH = "D:\Projects\rag-fastapi"   # Windows PowerShell
python -m pytest -q
```

The suite covers all 28 cases without needing an LLM:

- **Chunking** — size limits, sequential indices, metadata, boundary snapping,
  empty input
- **Ingest + retrieve** — relevance, sorting, replacement, source filtering,
  delete
- **Prompt building** — numbering, empty context
- **LLM client** — response parsing, request payload, offline error handling
- **HTTP API** — health, CRUD, validation errors (`422`), upload rejection,

  end-to-end query with a stubbed LLM

Tests use a throwaway ChromaDB collection in the OS temp directory via a
`monkeypatch` fixture, so they never touch the real index.

Lint and format:

```powershell
python -m ruff check app scripts tests
python -m ruff format app scripts tests
```

---

## What to learn next

This is a working baseline. The usual next steps, roughly in order of how much
improvement you get per unit of effort:

1. **Add a relevance threshold.** Drop chunks scoring below ~0.3 before calling
   the LLM. Otherwise irrelevant context dilutes the answer. *Easiest win.*
2. **Evaluate properly.** Build ~50 question/expected-answer pairs and measure
   **retrieval recall**: is the correct chunk in the top *k*? Without numbers
   you cannot tell if a change helped.
3. **Hybrid search.** Combine vector similarity with keyword search (BM25).
   Vector search is good at meaning but weak on rare terms, IDs and product
   names; keyword search is the opposite. Usually meaningfully better recall.
4. **Reranking.** Retrieve 50 chunks, then score each (question, chunk) pair
   with a cross-encoder, keep the best 5. Slower but materially more accurate
   than a bi-encoder.
5. **Better embeddings.** `all-MiniLM-L6-v2` is small and fast. `bge-large`
   or `gte-large` are stronger. Trade speed and RAM for accuracy.
6. **Streaming.** Return the answer token by token with a `StreamingResponse`,
   so the user sees output immediately.
7. **Async ingestion.** Embedding is CPU-bound. Move it to a background task
   (`BackgroundTasks`, then Celery) so a big document does not block the
   event loop.
8. **Conversational history.** Pass previous turns in `messages` so follow-up
   questions like "how much does *that* plan cost?" resolve correctly.

### The most important lesson

**In practice, most bad answers come from retrieval, not the language model.**
Before swapping models, print the retrieved chunks and read them. If the answer
is not in that text, no amount of prompt engineering will help — you need better
chunking, better embeddings, or more documents.

---

## Troubleshooting

**`503` on `/api/v1/query` with "Cannot reach the LLM"**
No LLM is listening. Start Ollama, or set `LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL`
in `.env`. Retrieval still works, so you can test with `GET /api/v1/documents`.

**`ModuleNotFoundError: No module named 'app'`**
Run commands from the project root, or set the path:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

**Similarity scores are low or negative**
The collection is using L2 instead of cosine distance. Delete `chroma_store/`
and re-ingest so the collection is rebuilt with cosine.

**`ImportError: cannot import name 'CollectionNotFoundError'`**
ChromaDB 1.x removed that exception. The code uses `list_collections()` to check
existence instead, which works across versions.

**First request is slow**
The embedding model downloads (~90 MB) on first use, then warms up during app
startup (`lifespan`). Subsequent starts read from the HuggingFace cache.

**HF symlink warnings on Windows**
Harmless — Windows without Developer Mode uses file copies instead of symlinks.
Set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to silence.

**500 on `/api/v1/documents/upload`**
Only `.txt` and `.md`, UTF-8 encoded, max 5 MB.
