from tenerife_rag.chunking import Chunk


def test_header_joins_path_with_separator():
    chunk = Chunk(id="1", text="Arena negra.", path=["Zona Norte", "Playa de Benijo"])

    assert chunk.header == "Zona Norte > Playa de Benijo"


def test_header_is_empty_without_path():
    chunk = Chunk(id="1", text="Arena negra.")

    assert chunk.header == ""


def test_embed_text_prepends_header_when_path_exists():
    chunk = Chunk(id="1", text="Arena negra.", path=["Norte", "Benijo"])

    assert chunk.embed_text() == "Norte > Benijo\nArena negra."


def test_embed_text_is_plain_text_without_path():
    chunk = Chunk(id="1", text="Arena negra.")

    assert chunk.embed_text() == "Arena negra."


def test_citation_shows_single_page():
    chunk = Chunk(id="1", text="...", path=["Benijo"], pages=[8])

    assert chunk.citation == "Benijo (p. 8)"


def test_citation_shows_page_range():
    chunk = Chunk(id="1", text="...", path=["Benijo"], pages=[8, 9])

    assert chunk.citation == "Benijo (p. 8-9)"


def test_citation_falls_back_to_id_and_unknown_page():
    chunk = Chunk(id="7", text="...")

    assert chunk.citation == "Fragmento 7 (p. ?)"


def test_document_round_trip_preserves_chunk():
    chunk = Chunk(id="1", text="Arena negra.", kind="seccion", path=["Norte"], pages=[8])

    assert Chunk.from_document(chunk.to_document()) == chunk


def test_to_document_embeds_header_but_keeps_raw_text_in_metadata():
    chunk = Chunk(id="1", text="Arena negra.", path=["Norte"])

    doc = chunk.to_document()

    assert doc.page_content == "Norte\nArena negra."
    assert doc.metadata["text"] == "Arena negra."


def test_to_document_uses_custom_text_fn():
    chunk = Chunk(id="1", text="Arena negra.", path=["Norte"])

    doc = chunk.to_document(lambda c: c.text.upper())

    assert doc.page_content == "ARENA NEGRA."
