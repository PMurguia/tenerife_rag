from datetime import date
from pathlib import Path

import httpx
import openai
from agents.testing import ModelStep, ScriptedModel, assistant_message, function_call
from conftest import FakeWeather

from app import (
    MAX_IMAGES,
    MISSING_KEY,
    THINKING,
    TRACE_EMPTY,
    cited_images,
    format_sources,
    make_respond,
    missing_key,
    new_conversation,
    photos_step,
    render_steps,
)
from tenerife_rag.assistant import TenerifeAssistant
from tenerife_rag.chunking import Chunk
from tenerife_rag.store import build_store
from tenerife_rag.trace import Step

BENIJO = Chunk(
    id="benijo",
    text="Playa de Benijo, arena negra.",
    path=["Zona Norte", "Anaga", "Playa de Benijo"],
    pages=[8],
    images=[58],
)
JARDINA = Chunk(
    id="jardina",
    text="Mirador de La Jardina, playa a la vista.",
    path=["Zona Norte", "Anaga", "Mirador de La Jardina"],
    pages=[8],
    images=[50, 51],
)
IMAGE_PATHS = {58: Path("58.jpeg"), 50: Path("50.jpeg"), 51: Path("51.jpeg")}


async def last(respond, message, chat=None, assistant=None):
    """Last (textbox, chat, assistant, photos, trace) yielded by the handler."""
    outputs = [output async for output in respond(message, chat or [], assistant)]
    return outputs[-1]


def test_format_sources_lists_only_cited_labels():
    sources = format_sources("Benijo [S2].", {"S1": JARDINA, "S2": BENIJO})

    assert "S2 · Zona Norte > Anaga > Playa de Benijo (p. 8)" in sources
    assert "S1" not in sources


def test_format_sources_is_empty_without_citations():
    assert format_sources("No lo sé.", {"S1": BENIJO}) == ""


def test_cited_images_show_only_photos_of_cited_places():
    images = cited_images("Benijo [S2].", {"S1": JARDINA, "S2": BENIJO}, IMAGE_PATHS)

    assert images == [("58.jpeg", "Playa de Benijo")]


def test_cited_images_are_capped():
    many = Chunk(id="troya", text="...", path=["Playa de Troya"], images=list(range(10)))
    paths = {xref: Path(f"{xref}.jpeg") for xref in range(10)}

    assert len(cited_images("Troya [S1].", {"S1": many}, paths)) == MAX_IMAGES


def test_cited_images_skip_photos_without_a_file():
    assert cited_images("Benijo [S1].", {"S1": BENIJO}, {}) == []


def test_photos_step_names_the_places_and_the_cap():
    many = Chunk(id="troya", text="...", path=["Playa de Troya"], images=list(range(6)))
    paths = {xref: Path(f"{xref}.jpeg") for xref in range(6)}

    step = photos_step("Troya [S1].", {"S1": many}, paths)

    assert step.details == ["de: Playa de Troya", f"se muestran {MAX_IMAGES} de 6 (máximo {MAX_IMAGES})"]


def test_photos_step_says_when_no_cited_place_has_photos():
    assert photos_step("No lo sé.", {"S1": BENIJO}, IMAGE_PATHS).details == ["ningún lugar citado tiene fotos"]


def test_render_steps_escapes_untrusted_text():
    html = render_steps([Step("rag", "Búsqueda", ["«<script>alert(1)</script>»"], ms=12.3)])

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "12 ms" in html


def test_render_steps_without_steps_shows_the_placeholder():
    assert render_steps([]) == TRACE_EMPTY


async def test_photos_of_cited_places_fill_the_gallery(config, embeddings):
    store = build_store([BENIJO], embeddings)
    model = ScriptedModel(
        [
            [
                function_call(
                    "search_tenerife_guide",
                    {"query": "benijo", "zona": None, "categoria": None, "k": 1},
                    call_id="call_1",
                )
            ],
            [assistant_message("Benijo tiene arena negra [S1].")],
        ]
    )
    respond = make_respond(lambda: TenerifeAssistant(config, store, FakeWeather(), model=model), IMAGE_PATHS)

    _, chat, _, photos, trace = await last(respond, "¿Cómo es Benijo?")

    assert chat[-1]["content"].startswith("Benijo tiene arena negra [S1].")
    assert photos == [("58.jpeg", "Playa de Benijo")]
    for kind in ["Memoria", "LLM", "RAG", "Respuesta", "Fotos"]:
        assert f'<span class="step-kind">{kind}</span>' in trace


async def test_grouped_citations_list_every_source(config, embeddings):
    store = build_store([BENIJO, JARDINA], embeddings)
    model = ScriptedModel(
        [
            [
                function_call(
                    "search_tenerife_guide",
                    {"query": "playa", "zona": None, "categoria": None, "k": 2},
                    call_id="call_1",
                )
            ],
            [assistant_message("Para el atardecer, Benijo [S1, S2].")],
        ]
    )
    respond = make_respond(
        lambda: TenerifeAssistant(config, store, FakeWeather(), model=model, today_fn=lambda: date(2026, 10, 9))
    )

    box, chat, _, _, _ = await last(respond, "¿Dónde ver el atardecer?")

    assert box == ""
    assert "- S1 · " in chat[-1]["content"]
    assert "- S2 · " in chat[-1]["content"]


async def test_respond_keeps_one_assistant_per_session(config, store):
    model = ScriptedModel([[assistant_message("Hola.")], [assistant_message("Adiós.")]])
    respond = make_respond(lambda: TenerifeAssistant(config, store, FakeWeather(), model=model))

    _, chat, assistant, _, _ = await last(respond, "Hola")
    _, chat, again, _, _ = await last(respond, "Adiós", chat, assistant)

    assert again is assistant
    assert [m["content"] for m in chat] == ["Hola", "Hola.", "Adiós", "Adiós."]


async def test_question_is_shown_before_the_model_answers(config, store):
    class Silent:
        async def ask_events(self, question):
            raise AssertionError("the question must be shown before the model is called")
            yield

    respond = make_respond(Silent)

    box, chat, _, photos, _ = await anext(respond("¿Qué ver en Anaga?", [], None))

    assert box == ""
    assert chat == [{"role": "user", "content": "¿Qué ver en Anaga?"}, {"role": "assistant", "content": THINKING}]
    assert photos == []


async def test_connection_error_is_explained_and_not_remembered(config, store):
    error = openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    model = ScriptedModel([ModelStep.raise_error(error)])
    respond = make_respond(lambda: TenerifeAssistant(config, store, FakeWeather(), model=model))

    _, chat, assistant, _, _ = await last(respond, "¿Qué ver en Anaga?")

    assert chat[-1]["content"] == "No hay conexión con el servicio del modelo. Comprueba tu red."
    assert assistant.memory.turns == []


async def test_unexpected_error_shows_no_traceback(config, store):
    class Broken:
        async def ask_events(self, question):
            raise ValueError("boom")
            yield

    _, chat, _, _, trace = await last(make_respond(Broken), "Hola")

    assert chat[-1]["content"] == "Ha ocurrido un error inesperado. Inténtalo de nuevo."
    assert "step-failed" in trace


async def test_missing_key_answers_with_instructions():
    _, chat, _, _, _ = await last(missing_key, "Hola")

    assert chat[-1] == {"role": "assistant", "content": MISSING_KEY}


def test_new_conversation_clears_everything_and_drops_the_assistant():
    assert new_conversation() == ("", [], None, [], TRACE_EMPTY)
