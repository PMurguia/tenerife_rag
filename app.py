"""Interfaz web de la Guía Tenerife. Uso: uv run python app.py"""

from collections.abc import AsyncIterator, Callable

import gradio as gr
import truststore

from tenerife_rag.assistant import TenerifeAssistant, friendly_error
from tenerife_rag.chunking import Chunk
from tenerife_rag.citations import cited_labels
from tenerife_rag.config import LLMConfig
from tenerife_rag.pipeline import build_index
from tenerife_rag.tools import setup_logging
from tenerife_rag.weather import WeatherService

EXAMPLES = [
    "¿Merece la pena ver el atardecer en Benijo mañana?",
    "¿Qué playa del sur es nudista?",
    "¿Dónde puedo comer en el norte de la isla?",
    "¿Qué ver en La Orotava?",
    "¿Qué tiempo hará el sábado en el Teide?",
]
MISSING_KEY = "Falta OPENAI_API_KEY en el archivo .env."

Chat = list[dict]
Respond = Callable[[str, Chat, TenerifeAssistant | None], AsyncIterator[tuple[str, Chat, TenerifeAssistant | None]]]


def format_sources(answer: str, chunks_by_label: dict[str, Chunk]) -> str:
    """Fuentes citadas en la respuesta, en el orden en que se recuperaron."""
    cited = cited_labels(answer)
    lines = [f"- {label} · {chunk.citation}" for label, chunk in chunks_by_label.items() if label in cited]
    return "\n\n**Fuentes**\n" + "\n".join(lines) if lines else ""


def make_respond(new_assistant: Callable[[], TenerifeAssistant]) -> Respond:
    """Manejador del chat: un asistente (con su memoria) por sesión de navegador."""

    async def respond(message: str, chat: Chat, assistant: TenerifeAssistant | None):
        if not message.strip():
            yield "", chat, assistant
            return
        assistant = assistant or new_assistant()
        chat = [*chat, {"role": "user", "content": message}, {"role": "assistant", "content": ""}]
        try:
            async for delta in assistant.ask_stream(message):
                chat[-1]["content"] += delta
                yield "", chat, assistant
        except Exception as exc:
            chat[-1]["content"] = friendly_error(exc)
            yield "", chat, assistant
            return
        chat[-1]["content"] += format_sources(chat[-1]["content"], assistant.chunks_by_label)
        yield "", chat, assistant

    return respond


async def missing_key(message: str, chat: Chat, assistant: None):
    """Sin API key no hay índice ni modelo: solo se explica qué falta."""
    yield "", [*chat, {"role": "user", "content": message}, {"role": "assistant", "content": MISSING_KEY}], assistant


def build_demo(respond: Respond) -> gr.Blocks:
    """La interfaz, sin lanzarla."""
    with gr.Blocks(title="Guía Tenerife") as demo:
        gr.Markdown("# Guía Tenerife\nPregunta por lugares, planes, comida o el tiempo. Las respuestas citan la guía.")
        state = gr.State(None)
        chatbot = gr.Chatbot(label="Conversación", height=520)
        box = gr.Textbox(placeholder="Pregunta sobre Tenerife…", show_label=False)
        gr.Examples(EXAMPLES, inputs=box)
        box.submit(respond, [box, chatbot, state], [box, chatbot, state])
    return demo


def main() -> None:
    """Todo lo que tiene efectos está aquí: importar app no hace nada."""
    truststore.inject_into_ssl()
    config = LLMConfig.from_env()
    setup_logging()
    if not config.api_key:
        build_demo(missing_key).launch()
        return
    store = build_index(config)
    weather = WeatherService()
    build_demo(make_respond(lambda: TenerifeAssistant(config, store, weather))).launch()


if __name__ == "__main__":
    main()
