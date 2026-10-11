from tenerife_rag.chunking import Chunk

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import Embeddings

def build_store(chunks: list[Chunk], embeddings: Embeddings, text_fn=Chunk.embed_text) -> InMemoryVectorStore:
    store = InMemoryVectorStore(embeddings)
    store.add_documents([c.to_document(text_fn) for c in chunks])
    return store

def search(store: InMemoryVectorStore, query: str, top_k: int=3, where=None) -> list[tuple[Chunk, float]]:
    doc_filter = None if where is None else (lambda doc: where(Chunk.from_document(doc)))
    hits = store.similarity_search_with_score(query, k=top_k, filter=doc_filter)
    return [(Chunk.from_document(doc), score) for doc, score in hits]