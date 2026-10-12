from datetime import date

from agents.testing import ScriptedModel, assistant_message, function_call

from conftest import FakeWeather
from tenerife_rag.assistant import TenerifeAssistant

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
