"""Tests for the RAG pipeline.

These tests use a throwaway ChromaDB collection in the OS temp directory so
they never touch the real index.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from app.services import pipeline
from app.services.chunking import chunk_text

build_prompt = pipeline.build_prompt
count_chunks = pipeline.count_stored_chunks

TEXT = """Nimbus plans.

The Starter plan costs 5 USD per month and includes 2 vCPU and 20 GB of storage.
The Team plan costs 45 USD per month and includes 8 vCPU, 32 GB of memory and
200 GB of storage. The Enterprise plan is billed annually and includes a
99.99% uptime SLA.

Manual scaling is done with the command nimbus scale my-app --replicas 6.
Autoscaling is enabled per service with a target CPU utilisation and a range of
replicas.
"""


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch, tmp_path):
    """Point the vector store at a temporary collection for every test."""
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", "test_rag")
    from app.core.config import get_settings

    get_settings.cache_clear()
    pipeline.reset_store()
    yield
    pipeline.reset_store()
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunk_text_respects_size():
    chunks = chunk_text(TEXT, chunk_size=120, chunk_overlap=20, source="t")
    assert chunks
    for chunk in chunks:
        assert len(chunk.text) <= 200, "chunks should stay near the target size"


def test_chunk_text_indices_are_sequential():
    chunks = chunk_text(TEXT, chunk_size=100, chunk_overlap=15, source="t")
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_text_carries_source_metadata():
    chunks = chunk_text(TEXT, chunk_size=100, source="guide.txt")
    assert all(c.metadata["source"] == "guide.txt" for c in chunks)


def test_chunk_text_does_not_start_mid_word():
    chunks = chunk_text(TEXT, chunk_size=80, chunk_overlap=40, source="t")
    for chunk in chunks:
        first = chunk.text.split()[0]
        # The overlap tail must begin at a word boundary, so the first token
        # should be a whole word rather than a fragment like "les.".
        assert first in TEXT, f"chunk starts mid-word: {chunk.text[:40]!r}"


def test_chunk_text_handles_empty_and_short_input():
    assert chunk_text("", source="empty") == []
    assert chunk_text("   \n\n  ", source="blank") == []
    short = chunk_text("Just one sentence.", source="t")
    assert len(short) == 1 and short[0].text == "Just one sentence."


# ---------------------------------------------------------------------------
# Ingest + retrieve
# ---------------------------------------------------------------------------


def test_ingest_then_retrieve():
    result = pipeline.ingest_document(text=TEXT, source="guide.txt")
    assert result.chunks_created > 0
    assert len(result.chunk_ids) == result.chunks_created

    hits = pipeline.similarity_search("What does the Team plan cost?", top_k=3)
    assert hits
    assert "Team plan" in hits[0].text
    assert hits[0].score >= hits[-1].score
    assert 0.0 <= hits[0].score <= 1.0


def test_ingest_replaces_previous_version():
    first = pipeline.ingest_document(
        text=TEXT, source="guide.txt", chunk_size=100, chunk_overlap=20
    )
    assert first.chunks_created > 1

    second = pipeline.ingest_document(text="Only this sentence.", source="guide.txt")
    assert second.chunks_created == 1
    assert pipeline.similarity_search("Team plan", top_k=10)[0].text == "Only this sentence."


def test_retrieve_on_empty_store_returns_nothing():
    assert pipeline.similarity_search("anything") == []


def test_search_respects_source_filter():
    pipeline.ingest_document(text=TEXT, source="guide.txt")
    pipeline.ingest_document(
        text="A completely unrelated note about gardening.",
        source="garden.txt",
    )

    hits = pipeline.similarity_search("Team plan", top_k=5, source_filter="garden.txt")
    assert all(hit.source == "garden.txt" for hit in hits)


def test_delete_document():
    pipeline.ingest_document(text=TEXT, source="guide.txt")
    assert pipeline.delete_document("guide.txt") > 0
    assert pipeline.similarity_search("Team plan") == []


def test_list_documents_groups_by_source():
    pipeline.ingest_document(text=TEXT, source="guide.txt")
    pipeline.ingest_document(text="Small note.", source="note.txt")

    documents = pipeline.list_indexed_documents()
    assert {d["source"] for d in documents} == {"guide.txt", "note.txt"}
    assert sum(d["chunks"] for d in documents) == count_chunks()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_build_prompt_numbers_context():
    prompt = build_prompt("Who pays?", ["first block", "second block"])
    assert "[1] first block" in prompt
    assert "[2] second block" in prompt
    assert "QUESTION: Who pays?" in prompt


def test_build_prompt_without_context():
    prompt = build_prompt("Who pays?", [])
    assert "no relevant documents found" in prompt


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


class FakeResponse:
    """Stand-in for an httpx response."""

    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _patch_llm(monkeypatch, payload):
    monkeypatch.setattr(
        httpx.Client, "post", lambda self, url, **kwargs: FakeResponse(payload)
    )


def test_generate_returns_answer_and_model(monkeypatch):
    from app.services import llm

    _patch_llm(
        monkeypatch,
        {"choices": [{"message": {"content": "45 USD per month."}}], "model": "llama"},
    )
    answer, model = llm.generate("Cost?", ["Team plan costs 45 USD."])
    assert answer == "45 USD per month."
    assert model == "llama"


def test_generate_includes_context_and_question(monkeypatch):
    from app.services import llm

    captured = {}

    def fake_post(self, url, **kwargs):
        captured.update(kwargs["json"])
        return FakeResponse({"choices": [{"message": {"content": "ok"}}], "model": "m"})

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    llm.generate("Cost?", ["Team plan costs 45 USD."])

    messages = captured["messages"]
    assert messages[0]["role"] == "system"
    assert "Team plan costs 45 USD." in messages[1]["content"]
    assert "Cost?" in messages[1]["content"]


def test_generate_raises_actionable_error_when_llm_is_offline(monkeypatch):
    from app.services import llm

    def offline(self, url, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx.Client, "post", offline)

    with pytest.raises(RuntimeError, match="Cannot reach the LLM"):
        llm.generate("Cost?", ["ctx"])


def test_generate_raises_on_empty_answer(monkeypatch):
    from app.services import llm

    _patch_llm(monkeypatch, {"choices": [{"message": {"content": "  "}}], "model": "m"})

    with pytest.raises(RuntimeError, match="empty answer"):
        llm.generate("Cost?", ["ctx"])


def test_query_returns_503_when_llm_is_offline(client, monkeypatch):
    """A dead LLM becomes a clear 503 instead of an unhandled crash."""

    def offline(**kwargs):
        raise RuntimeError("Cannot reach the LLM at 'http://localhost:11434/v1'.")

    # Patching httpx.Client globally would also break TestClient's transport,
    # so patch our own wrapper instead.
    monkeypatch.setattr("app.services.llm.generate", offline)

    pipeline.ingest_document(text=TEXT, source="guide.txt", chunk_size=100)
    response = client.post("/api/v1/query", json={"question": "Cost?", "top_k": 2})

    assert response.status_code == 503
    assert "Cannot reach the LLM" in response.json()["detail"]


def test_query_end_to_end_with_mocked_llm(client, monkeypatch):
    """The whole retrieve-then-generate path, with the LLM stubbed out."""
    monkeypatch.setattr(
        "app.services.llm.generate",
        lambda **kwargs: ("The Team plan costs 45 USD per month [1].", "test-model"),
    )

    client.post(
        "/api/v1/documents",
        json={"text": TEXT, "source": "guide.txt", "metadata": {"team": "docs"}},
    )

    response = client.post(
        "/api/v1/query",
        json={"question": "What does the Team plan cost?", "top_k": 3},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["answer"] == "The Team plan costs 45 USD per month [1]."
    assert payload["model"] == "test-model"
    assert payload["sources"], "sources should be returned by default"
    assert all("Team plan" in s["text"] for s in payload["sources"][:2])
    assert payload["latency_ms"] >= 0
    assert "[1] " in payload["context_used"]


def test_query_can_hide_sources(client, monkeypatch):
    monkeypatch.setattr(
        "app.services.llm.generate",
        lambda **kwargs: ("A short answer.", "test-model"),
    )
    client.post("/api/v1/documents", json={"text": TEXT, "source": "guide.txt"})

    response = client.post(
        "/api/v1/query",
        json={"question": "Anything?", "include_sources": False},
    )
    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert response.json()["context_used"] is None


def test_query_respects_source_filter(client, monkeypatch):
    monkeypatch.setattr(
        "app.services.llm.generate",
        lambda **kwargs: ("A short answer.", "test-model"),
    )
    client.post("/api/v1/documents", json={"text": TEXT, "source": "guide.txt"})
    client.post(
        "/api/v1/documents",
        json={"text": "Gardening notes about tomatoes.", "source": "garden.txt"},
    )

    response = client.post(
        "/api/v1/query",
        json={
            "question": "What does the Team plan cost?",
            "top_k": 5,
            "source_filter": "garden.txt",
        },
    )
    assert response.status_code == 200
    assert [s["source"] for s in response.json()["sources"]] == ["garden.txt"]


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(isolated_store):
    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["embedding_model"]


def test_ingest_and_list_documents(client):
    response = client.post(
        "/api/v1/documents",
        json={"text": TEXT, "source": "guide.txt", "metadata": {"team": "docs"}},
    )
    assert response.status_code == 201
    assert response.json()["chunks_created"] > 0

    listing = client.get("/api/v1/documents").json()
    assert listing["total_documents"] == 1
    assert listing["documents"][0]["metadata"] == {"team": "docs"}


def test_query_rejects_empty_question(client):
    response = client.post("/api/v1/query", json={"question": ""})
    assert response.status_code == 422


def test_query_rejects_out_of_range_top_k(client):
    response = client.post("/api/v1/query", json={"question": "hi", "top_k": 999})
    assert response.status_code == 422


def test_delete_missing_document_returns_404(client):
    response = client.delete("/api/v1/documents/does-not-exist.txt")
    assert response.status_code == 404


def test_upload_rejects_unsupported_extension(client):
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("notes.pdf", b"binary", "application/pdf")},
    )
    assert response.status_code == 400


def test_upload_txt_file(client):
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("note.txt", TEXT.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 201
    assert response.json()["source"] == "note.txt"
