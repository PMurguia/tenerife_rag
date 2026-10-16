import json
from datetime import date

import httpx
import openai
from agents import Usage
from agents.testing import ScriptedModel, assistant_message, function_call

from conftest import FakeWeather
from tenerife_rag.assistant import TenerifeAssistant, friendly_error, instructions_for

async def test_tool_output_keeps_call_id(config, store):
    model = ScriptedModel([
        [function_call("get_weather", {"fecha": "2026-10-10", "lugar": "teide"}, call_id="call_1")],
        [assistant_message("Mañana, 12 °C en el Teide.")],
    ])
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
    model = ScriptedModel([
        [function_call("get_weather", {"fecha": "2026-10-10", "lugar": "teide"}, call_id="call_1")],
        [assistant_message("Mañana, 12 °C en el Teide.")],
    ], default_usage=Usage(requests=1, input_tokens=100, output_tokens=10, total_tokens=110))
    log = tmp_path / "logs" / "turns.jsonl"
    assistant = TenerifeAssistant(config, store, FakeWeather(), model=model, turn_log=log)

    await assistant.ask("¿Qué tiempo hará mañana en el Teide?")

    [record] = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert record["herramientas"] == ["get_weather"]
    assert (record["input_tokens"], record["output_tokens"]) == (200, 20)  # summed over both rounds
    assert record["memoria_tokens"] == assistant.memory.tokens()
    assert record.keys() >= {"ts", "pregunta", "latencia_s"}


def test_scope_block_goes_first_in_the_instructions():
    instructions = instructions_for(date(2026, 10, 9))

    assert instructions.startswith("ALCANCE (prioridad máxima)")
    assert "administrador" in instructions


def test_friendly_error_never_shows_the_exception():
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")

    assert friendly_error(openai.APIConnectionError(request=request)).startswith("No hay conexión")
    assert friendly_error(openai.RateLimitError("429", response=httpx.Response(429, request=request), body=None)).startswith("Se ha superado")
    assert friendly_error(openai.AuthenticationError("401", response=httpx.Response(401, request=request), body=None)).startswith("La API key")
    assert "boom" not in friendly_error(ValueError("boom"))
