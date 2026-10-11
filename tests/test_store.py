from tenerife_rag.chunking import Chunk
from tenerife_rag.store import build_store, search

CHUNKS = [
    Chunk(id="1", text="pizzeria en la orotava", path=["Zona Norte"]),
    Chunk(id="2", text="playa de arena negra", path=["Zona Norte"]),
    Chunk(id="3", text="playa de arena dorada", path=["Zona Sur"]),
]


def test_search_returns_chunks_with_scores(embeddings):
    store = build_store(CHUNKS, embeddings)

    hits = search(store, "playa")

    assert all(isinstance(chunk, Chunk) for chunk, _ in hits)
    assert all(isinstance(score, float) for _, score in hits)


def test_search_ranks_relevant_chunk_first(embeddings):
    store = build_store(CHUNKS, embeddings)

    hits = search(store, "arena negra", top_k=1)

    assert hits[0][0].id == "2"


def test_search_limits_results_to_top_k(embeddings):
    store = build_store(CHUNKS, embeddings)

    assert len(search(store, "playa", top_k=2)) == 2


def test_search_filters_with_where(embeddings):
    store = build_store(CHUNKS, embeddings)

    hits = search(store, "playa", top_k=3, where=lambda c: "Zona Sur" in c.path)

    assert [chunk.id for chunk, _ in hits] == ["3"]


def test_search_returns_original_chunk_not_embedded_text(embeddings):
    store = build_store(CHUNKS, embeddings)

    chunk, _ = search(store, "arena negra", top_k=1)[0]

    assert chunk == CHUNKS[1]


def test_build_store_uses_custom_text_fn(embeddings):
    chunks = [
        Chunk(id="1", text="sin palabras clave"),
        Chunk(id="2", text="otro texto"),
    ]
    store = build_store(chunks, embeddings, text_fn=lambda c: "volcan" if c.id == "2" else c.text)

    hits = search(store, "volcan", top_k=1)

    assert hits[0][0].id == "2"
