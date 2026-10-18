import hashlib

import pytest
from agents import set_tracing_disabled
from langchain_core.embeddings import Embeddings

from tenerife_rag.chunking import Chunk
from tenerife_rag.config import LLMConfig
from tenerife_rag.store import build_store

set_tracing_disabled(True)


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings: texts sharing words end up close."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [1e-6] * 256
        for word in text.lower().split():
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1
        return vector


class FakeWeather:
    def get_weather(self, fecha: str, lugar: str | None = None) -> dict:
        return {"ok": True, "fecha": fecha, "lugar_id": lugar, "estado": "despejado", "temp_max_c": 12, "temp_min_c": 5}


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """Relative paths (logs/turns.jsonl, logs/tool_calls.log) land in a temp dir, not in the repo."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def config() -> LLMConfig:
    return LLMConfig()


@pytest.fixture
def store(embeddings):
    chunks = [
        Chunk(
            id="benijo",
            text="Playa de Benijo, arena negra y atardeceres.",
            path=["Zona Norte", "Anaga", "Playa de Benijo"],
            pages=[8],
        ),
    ]
    return build_store(chunks, embeddings)
