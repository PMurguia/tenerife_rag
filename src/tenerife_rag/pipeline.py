"""Índice de la guía, el mismo para el notebook y la app."""

import tempfile
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from tenerife_rag.chunking import structural_chunks
from tenerife_rag.config import LLMConfig
from tenerife_rag.extraction import extract_images, extract_lines, save_images
from tenerife_rag.store import build_store

GUIDE_PDF = Path(__file__).parent / "data" / "TENERIFE.pdf"
IMAGE_DIR = Path(tempfile.gettempdir()) / "tenerife_rag" / "images"  # Gradio serves files from the temp dir


def build_index(config: LLMConfig, embeddings: Embeddings | None = None, pdf: Path = GUIDE_PDF) -> InMemoryVectorStore:
    """Chunks estructurales de la guía, con sus fotos, embebidos con su cabecera."""
    chunks = structural_chunks(extract_lines(str(pdf)), extract_images(str(pdf)))
    return build_store(chunks, embeddings or config.embeddings())


def export_images(pdf: Path = GUIDE_PDF, out_dir: Path = IMAGE_DIR) -> dict[int, Path]:
    """Fotos de la guía guardadas en disco, por xref, para mostrarlas junto a las respuestas."""
    return save_images(str(pdf), out_dir)
