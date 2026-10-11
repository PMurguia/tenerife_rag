import hashlib

import pytest
from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings: texts sharing words end up close."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [1e-6] * 256
        for word in text.lower().split():
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1
        return vector


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()
