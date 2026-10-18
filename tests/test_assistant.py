import json
from datetime import date

import httpx
import openai
from agents import Usage
from agents.testing import ModelStep, ScriptedModel, assistant_message, function_call
from conftest import FakeWeather

from tenerife_rag.assistant import TenerifeAssistant, friendly_error, instructions_for
from tenerife_rag.trace import Step


async def test_tool_output_keeps_call_id(config, store):
    model = ScriptedModel(
        [
            [function_call("get_weather", {"fecha": "2026-10-10", "lugar": "teide"}, call_id="call_1")],
            [assistant_message("Mañana, 12 °C en el Teide.")],
        ]
    )
    assistant = TenerifeAssistant(config, store, FakeWeather(), model=model, today_fn=lambda: date(2026, 10, 9))

    await assistant.ask("¿Qué tiempo hará mañana en el Teide?")

    outputs = [i for i in model.last_call.input if i.get("type") == "function_call_output"]
    assert outputs[0]["call_id"] == "call_1"
    assert "2026-10-09" in model.first_call.system_instructions


async def test_ask_stream_yields_the_answer_in_pieces(config, store):
    model = ScriptedModel([[assistant_message("Benijo tiene arena negra.")]])
    assistant = TenerifeAssistant(config, store, FakeWeather(), model=model)

    deltas = [delta async for delta in assistant.ask_stream("¿Cómo es Benijo?")]

    assert "".join(deltas) == "Benijo tiene arena negra."
    assert assistant.memory.turns[0].assistant == "Benijo tiene arena negra."


async def test_each_turn_is_logged_as_one_json_line(config, store, tmp_path):
    model = ScriptedModel(
        [
            [function_call("get_weather", {"fecha": "2026-10-10", "lugar": "teide"}, call_id="call_1")],
            [assistant_message("Mañana, 12 °C en el Teide.")],
        ],
        default_usage=Usage(requests=1, input_tokens=100, output_tokens=10, total_tokens=110),
    )
    log = tmp_path / "logs" / "turns.jsonl"
    assistant = TenerifeAssistant(config, store, FakeWeather(), model=model, turn_log=log)

    await assistant.ask("¿Qué tiempo hará mañana en el Teide?")

    [record] = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert record["herramientas"] == ["get_weather"]
    assert (record["input_tokens"], record["output_tokens"]) == (200, 20)  # summed over both rounds
    assert record["memoria_tokens"] == assistant.memory.tokens()
    assert record.keys() >= {"ts", "pregunta", "latencia_s"}


async def test_ask_events_traces_every_step_of_the_turn(config, store):
    model = ScriptedModel(
        [
            [
                function_call(
                    "search_tenerife_guide",
                    {"query": "benijo", "zona": "Zona Norte", "categoria": None, "k": 1},
                    call_id="call_1",
                )
            ],
            [function_call("get_weather", {"fecha": "2026-10-10", "lugar": "anaga"}, call_id="call_2")],
            [assistant_message("Benijo tiene arena negra [S1] y mañana estará despejado.")],
        ],
        default_usage=Usage(requests=1, input_tokens=100, output_tokens=10, total_tokens=110),
    )
    assistant = TenerifeAssistant(config, store, FakeWeather(), model=model, today_fn=lambda: date(2026, 10, 9))

    events = [event async for event in assistant.ask_events("¿Vale la pena ir a Benijo mañana?")]

    steps = [event for event in events if isinstance(event, Step)]
    assert [step.kind for step in steps] == ["memoria", "llm", "rag", "llm", "tiempo", "llm", "respuesta", "memoria"]
    assert steps[1].details[1:] == ["100 tokens de entrada · 10 de salida", "pide search_tenerife_guide"]
    assert steps[2].details[0] == "«benijo» · zona Zona Norte · categoría — · k=1"
    assert steps[2].details[1].startswith("S1 · Zona Norte > Anaga > Playa de Benijo (p. 8) · score ")
    assert steps[4].details[:2] == ["fecha 2026-10-10 · lugar anaga", "despejado, 5–12 °C"]
    assert steps[5].details[-1] == "escribe la respuesta"
    assert steps[6].details[:2] == ["cita: S1", "recuperados sin citar: ninguno"]
    assert steps[7].details == [f"1 turno · {assistant.memory.tokens()} tokens de {assistant.memory.budget}"]
    assert steps == assistant.turn.steps


async def test_ask_events_traces_a_failed_tool(config, store):
    class DownWeather:
        def get_weather(self, fecha, lugar=None):
            return {"ok": False, "error": "timeout", "message": "Open-Meteo no respondió a tiempo."}

    model = ScriptedModel(
        [
            [function_call("get_weather", {"fecha": "2026-10-10", "lugar": None}, call_id="call_1")],
            [assistant_message("No tengo la predicción.")],
        ]
    )
    assistant = TenerifeAssistant(config, store, DownWeather(), model=model)

    await assistant.ask("¿Qué tiempo hará?")

    [weather] = [step for step in assistant.turn.steps if step.kind == "tiempo"]
    assert not weather.ok
    assert weather.details[1] == "timeout: Open-Meteo no respondió a tiempo."


async def test_ask_events_traces_a_model_error(config, store):
    error = openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    model = ScriptedModel([ModelStep.raise_error(error)])
    assistant = TenerifeAssistant(config, store, FakeWeather(), model=model)

    events = [event async for event in assistant.ask_events("Hola")]

    assert events[-1] == friendly_error(error)
    assert events[-2].kind == "error"
    assert not events[-2].ok


def test_scope_block_goes_first_in_the_instructions():
    instructions = instructions_for(date(2026, 10, 9))

    assert instructions.startswith("ALCANCE (prioridad máxima)")
    assert "administrador" in instructions


def test_friendly_error_never_shows_the_exception():
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")

    assert friendly_error(openai.APIConnectionError(request=request)).startswith("No hay conexión")
    assert friendly_error(
        openai.RateLimitError("429", response=httpx.Response(429, request=request), body=None)
    ).startswith("Se ha superado")
    assert friendly_error(
        openai.AuthenticationError("401", response=httpx.Response(401, request=request), body=None)
    ).startswith("La API key")
    assert "boom" not in friendly_error(ValueError("boom"))
