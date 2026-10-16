from datetime import date

import httpx
import openai
from agents.testing import ModelStep, ScriptedModel, assistant_message, function_call

from app import MISSING_KEY, format_sources, make_respond, missing_key
from conftest import FakeWeather
from tenerife_rag.assistant import TenerifeAssistant
from tenerife_rag.chunking import Chunk
from tenerife_rag.store import build_store

BENIJO = Chunk(id="benijo", text="Playa de Benijo, arena negra.", path=["Zona Norte", "Anaga", "Playa de Benijo"], pages=[8])
JARDINA = Chunk(id="jardina", text="Mirador de La Jardina, playa a la vista.", path=["Zona Norte", "Anaga", "Mirador de La Jardina"], pages=[8])


async def last(respond, message, chat=None, assistant=None):
    """Last (textbox, chat, assistant) yielded by the handler."""
    output = None
    async for output in respond(message, chat or [], assistant):
        pass
    return output


def test_format_sources_lists_only_cited_labels():
    sources = format_sources("Benijo [S2].", {"S1": JARDINA, "S2": BENIJO})

    assert "S2 · Zona Norte > Anaga > Playa de Benijo (p. 8)" in sources
    assert "S1" not in sources


def test_format_sources_is_empty_without_citations():
    assert format_sources("No lo sé.", {"S1": BENIJO}) == ""


async def test_grouped_citations_list_every_source(config, embeddings):
    store = build_store([BENIJO, JARDINA], embeddings)
    model = ScriptedModel([
        [function_call("search_tenerife_guide", {"query": "playa", "zona": None, "categoria": None, "k": 2}, call_id="call_1")],
        [assistant_message("Para el atardecer, Benijo [S1, S2].")],
    ])
    respond = make_respond(lambda: TenerifeAssistant(config, store, FakeWeather(), model=model, today_fn=lambda: date(2026, 10, 9)))

    box, chat, _ = await last(respond, "¿Dónde ver el atardecer?")

    assert box == ""
    assert "- S1 · " in chat[-1]["content"] and "- S2 · " in chat[-1]["content"]


async def test_respond_keeps_one_assistant_per_session(config, store):
    model = ScriptedModel([[assistant_message("Hola.")], [assistant_message("Adiós.")]])
    respond = make_respond(lambda: TenerifeAssistant(config, store, FakeWeather(), model=model))

    _, chat, assistant = await last(respond, "Hola")
    _, chat, again = await last(respond, "Adiós", chat, assistant)

    assert again is assistant
    assert [m["content"] for m in chat] == ["Hola", "Hola.", "Adiós", "Adiós."]


async def test_connection_error_is_explained_and_not_remembered(config, store):
    error = openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    model = ScriptedModel([ModelStep.raise_error(error)])
    respond = make_respond(lambda: TenerifeAssistant(config, store, FakeWeather(), model=model))

    _, chat, assistant = await last(respond, "¿Qué ver en Anaga?")

    assert chat[-1]["content"] == "No hay conexión con el servicio del modelo. Comprueba tu red."
    assert assistant.memory.turns == []


async def test_unexpected_error_shows_no_traceback(config, store):
    class Broken:
        async def ask_stream(self, question):
            raise ValueError("boom")
            yield

    _, chat, _ = await last(make_respond(Broken), "Hola")

    assert chat[-1]["content"] == "Ha ocurrido un error inesperado. Inténtalo de nuevo."


async def test_missing_key_answers_with_instructions():
    _, chat, _ = await last(missing_key, "Hola")

    assert chat[-1] == {"role": "assistant", "content": MISSING_KEY}
