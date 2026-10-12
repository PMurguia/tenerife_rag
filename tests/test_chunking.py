from tenerife_rag.chunking import Chunk, _clean, _title, structural_chunks
from tenerife_rag.extraction import Line


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


# --- structural chunker -----------------------------------------------------


def bullet(marker: str, text: str, x: float, page: int = 1, bold: bool = False) -> Line:
    return Line(page, 0.0, x, f"{marker} {text}", bold, marker)


def plain(text: str, x: float, page: int = 1, bold: bool = False) -> Line:
    return Line(page, 0.0, x, text, bold, None)


ZONE = plain("ZONA NORTE", 73, bold=True)


def test_title_cuts_at_first_punctuation_and_drops_noise():
    assert _title("Mirador de La Jardina (diría que es mi favorito) [ubicación]") == "Mirador de La Jardina"
    assert _title("Santa Cruz de Tenerife:") == "Santa Cruz de Tenerife"


def test_clean_removes_link_noise_and_space_before_punctuation():
    assert _clean("Aparcar cerca [ubicación], donde está la iglesia [vídeo - ubicación].") == (
        "Aparcar cerca, donde está la iglesia."
    )


def test_bullets_build_zone_locality_and_place_path():
    chunks = structural_chunks([
        ZONE,
        bullet("•", "Parque rural de Anaga", 91, bold=True),
        bullet("o", "Playa de Benijo [vídeo - ubicación]", 127),
        plain("Playa de arena negra.", 145),
    ])

    assert [(c.kind, c.header) for c in chunks] == [
        ("seccion", "Zona Norte > Parque rural de Anaga"),
        ("poi", "Zona Norte > Parque rural de Anaga > Playa de Benijo"),
    ]
    assert chunks[1].text == "Playa de Benijo Playa de arena negra."


def test_route_steps_are_detected_by_their_verb():
    chunks = structural_chunks([
        ZONE,
        bullet("•", "San Cristóbal de La Laguna", 91, bold=True),
        bullet("o", "Aparcar cerca de la Plaza de la Concepción", 127),
    ])

    assert chunks[1].kind == "ruta"


def test_curiosities_are_grouped_in_one_chunk_until_text_goes_back_to_section():
    chunks = structural_chunks([
        ZONE,
        bullet("•", "La Orotava", 91, bold=True),
        plain("Curiosidades:", 109, bold=True),
        bullet("o", "Es el municipio más alto de España.", 127),
        bullet("o", "Tiene la alfombra de tierra más grande.", 127),
        plain("A La Orotava pertenecen tres playas:", 109),
        bullet("o", "Playa del Ancón", 127),
    ])

    assert [c.kind for c in chunks] == ["seccion", "curiosidades", "poi"]
    assert chunks[1].text == "Es el municipio más alto de España. Tiene la alfombra de tierra más grande."
    assert chunks[0].text.endswith("A La Orotava pertenecen tres playas:")


def test_less_indented_text_after_a_place_goes_back_to_section():
    chunks = structural_chunks([
        ZONE,
        bullet("•", "Costa Adeje", 91, bold=True),
        bullet("o", "Playa del Camisón", 127),
        plain("Esta es una playa pequeñita.", 145),
        plain("Debéis ir al Siam Park.", 109, page=2),
    ])

    assert chunks[1].text == "Playa del Camisón Esta es una playa pequeñita."
    assert chunks[0].text.endswith("Debéis ir al Siam Park.")
    assert chunks[0].pages == [1, 2]


def test_restaurants_hang_from_the_place_above():
    chunks = structural_chunks([
        plain("OTRAS VAINAS", 73, bold=True),
        bullet("•", "Sitios para comer", 91),
        bullet("o", "Zona Norte", 127, bold=True),
        bullet("▪", "Pizzería Gioconda (zona de La Orotava [ubicación]) [Tripadvisor]", 163),
    ])

    assert chunks[-1].kind == "restaurante"
    assert chunks[-1].header == "Otras vainas > Sitios para comer > Zona Norte > Pizzería Gioconda"
    assert chunks[-1].text == "Pizzería Gioconda (zona de La Orotava)"


def test_lines_before_first_zone_are_ignored():
    chunks = structural_chunks([plain("TENERIFE – LUGARES DE INTERÉS", 73, bold=True), ZONE])

    assert chunks == []
