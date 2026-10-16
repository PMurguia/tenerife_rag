import pytest

from tenerife_rag.pipeline import GUIDE_PDF, build_index
from tenerife_rag.store import search


def test_build_index_indexes_the_structural_chunks_of_the_guide(config, embeddings):
    if not GUIDE_PDF.exists():
        pytest.skip("TENERIFE.pdf not available")

    store = build_index(config, embeddings)

    chunk, _ = search(store, "Zona Norte Parque rural de Anaga Playa de Benijo", top_k=1)[0]
    assert chunk.header == "Zona Norte > Parque rural de Anaga > Playa de Benijo"
