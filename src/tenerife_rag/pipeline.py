"""Índice de la guía, el mismo para el notebook y la app."""

from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from tenerife_rag.chunking import structural_chunks
from tenerife_rag.config import LLMConfig
from tenerife_rag.extraction import extract_lines
from tenerife_rag.store import build_store

GUIDE_PDF = Path(__file__).parent / "data" / "TENERIFE.pdf"


def build_index(config: LLMConfig, embeddings: Embeddings | None = None, pdf: Path = GUIDE_PDF) -> InMemoryVectorStore:
    """Chunks estructurales de la guía embebidos con su cabecera."""
    chunks = structural_chunks(extract_lines(str(pdf)))
    return build_store(chunks, embeddings or config.embeddings())
