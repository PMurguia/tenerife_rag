"""Interfaz web de la Guía Tenerife. Uso: uv run python app.py"""

from collections.abc import AsyncIterator, Callable
from html import escape
from pathlib import Path

import gradio as gr
import truststore

from tenerife_rag.assistant import TenerifeAssistant, friendly_error
from tenerife_rag.chunking import Chunk
from tenerife_rag.citations import cited_labels
from tenerife_rag.config import LLMConfig
from tenerife_rag.pipeline import build_index, export_images
from tenerife_rag.tools import setup_logging
from tenerife_rag.trace import Step
from tenerife_rag.weather import WeatherService

EXAMPLES = [
    "¿Merece la pena ver el atardecer en Benijo mañana?",
    "¿Qué playa del sur es nudista?",
    "¿Dónde puedo comer en el norte de la isla?",
    "¿Qué ver en La Orotava?",
    "¿Qué tiempo hará el sábado en el Teide?",
]
MISSING_KEY = "Falta OPENAI_API_KEY en el archivo .env."
MAX_IMAGES = 4

CSS = Path(__file__).with_name("app.css")
THEME = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="indigo",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)
HERO = """
<div class="hero">
  <h1>Guía Tenerife</h1>
  <p>Tu asistente para descubrir la isla: playas, miradores, rutas, dónde comer y qué tiempo hará,
  con lo que cuenta la guía.</p>
  <div class="badges">
    <span>📖 Respuestas con citas</span><span>⛅ Tiempo de Open-Meteo</span>
    <span>📷 Fotos de la guía</span><span>💬 Recuerda la conversación</span>
  </div>
</div>
"""
WELCOME = """<div class="welcome">
  <div class="icon">🌋</div>
  <h3>¡Hola! Soy tu guía de Tenerife</h3>
  <p>Pregúntame por lugares, planes o comida y te respondo con lo que dice la guía, citando cada fragmento.
  Si el plan depende del tiempo, consulto la predicción.</p>
</div>"""
THINKING = '<span class="thinking">Pensando<span class="dots"><i></i><i></i><i></i></span></span>'
TRACE_EMPTY = (
    '<div class="trace-empty">Aquí verás cada paso de la respuesta: la memoria, las rondas del modelo, '
    "las búsquedas en la guía, el tiempo y las fotos.</div>"
)
STEP_LABELS = {
    "memoria": "Memoria",
    "llm": "LLM",
    "rag": "RAG",
    "tiempo": "Tiempo",
    "respuesta": "Respuesta",
    "fotos": "Fotos",
    "error": "Error",
}
FOOTER = '<div class="footer">Fuentes: guía TENERIFE.pdf · Predicción: Open-Meteo · Modelo: OpenAI</div>'

Chat = list[dict]
Photos = list[tuple[str, str]]  # (path, caption) for gr.Gallery
Outputs = tuple[str, Chat, TenerifeAssistant | None, Photos, str]  # textbox, chat, state, gallery, trace HTML
Respond = Callable[[str, Chat, TenerifeAssistant | None], AsyncIterator[Outputs]]


def format_sources(answer: str, chunks_by_label: dict[str, Chunk]) -> str:
    """Fuentes citadas en la respuesta, en el orden en que se recuperaron."""
    cited = cited_labels(answer)
    lines = [f"- {label} · {chunk.citation}" for label, chunk in chunks_by_label.items() if label in cited]
    return "\n\n**Fuentes**\n" + "\n".join(lines) if lines else ""


def _cited_photos(answer: str, chunks_by_label: dict[str, Chunk], image_paths: dict[int, Path]) -> Photos:
    cited = cited_labels(answer)
    return [
        (str(image_paths[xref]), chunk.path[-1])
        for label, chunk in chunks_by_label.items()
        if label in cited
        for xref in chunk.images
        if xref in image_paths
    ]


def cited_images(answer: str, chunks_by_label: dict[str, Chunk], image_paths: dict[int, Path]) -> Photos:
    """Fotos de la guía de los lugares citados (como mucho MAX_IMAGES), con el lugar como pie de foto."""
    return _cited_photos(answer, chunks_by_label, image_paths)[:MAX_IMAGES]


def photos_step(answer: str, chunks_by_label: dict[str, Chunk], image_paths: dict[int, Path]) -> Step:
    """Paso de la traza con las fotos que van a la galería y de qué lugares salen."""
    photos = _cited_photos(answer, chunks_by_label, image_paths)
    if not photos:
        return Step("fotos", "Fotos de la guía", ["ningún lugar citado tiene fotos"])
    places = ", ".join(dict.fromkeys(place for _, place in photos))
    details = [f"de: {places}", f"se muestran {min(len(photos), MAX_IMAGES)} de {len(photos)} (máximo {MAX_IMAGES})"]
    return Step("fotos", "Fotos de la guía", details)


def render_steps(steps: list[Step]) -> str:
    """La traza del turno como HTML; todo el texto se escapa porque viene del usuario, la guía o el modelo."""
    if not steps:
        return TRACE_EMPTY
    items = []
    for number, step in enumerate(steps, 1):
        ms = f'<span class="step-ms">{step.ms:.0f} ms</span>' if step.ms is not None else ""
        details = "".join(f"<li>{escape(detail)}</li>" for detail in step.details)
        items.append(
            f'<li class="step step-{step.kind}{"" if step.ok else " step-failed"}">'
            f'<div class="step-head"><span class="step-n">{number}</span>'
            f'<span class="step-kind">{STEP_LABELS[step.kind]}</span>'
            f'<span class="step-title">{escape(step.title)}</span>{ms}</div>'
            f'<ul class="step-details">{details}</ul></li>'
        )
    return f'<ol class="trace">{"".join(items)}</ol>'


def make_respond(new_assistant: Callable[[], TenerifeAssistant], image_paths: dict[int, Path] | None = None) -> Respond:
    """Manejador del chat: un asistente (con su memoria) por sesión de navegador."""

    async def respond(message: str, chat: Chat, assistant: TenerifeAssistant | None):
        if not message.strip():
            yield "", chat, assistant, gr.skip(), gr.skip()
            return
        assistant = assistant or new_assistant()
        # The question shows up at once, with a thinking bubble until the first streamed text replaces it
        chat = [*chat, {"role": "user", "content": message}, {"role": "assistant", "content": THINKING}]
        steps: list[Step] = []
        yield "", chat, assistant, [], render_steps(steps)
        answer = ""
        try:
            async for event in assistant.ask_events(message):
                if isinstance(event, Step):
                    steps.append(event)
                    yield "", chat, assistant, [], render_steps(steps)
                else:
                    answer += event
                    chat[-1]["content"] = answer
                    yield "", chat, assistant, [], gr.skip()
        except Exception as exc:
            chat[-1]["content"] = friendly_error(exc)
            steps.append(Step("error", "Error inesperado", [type(exc).__name__], ok=False))
            yield "", chat, assistant, [], render_steps(steps)
            return
        chat[-1]["content"] = answer + format_sources(answer, assistant.chunks_by_label)
        steps.append(photos_step(answer, assistant.chunks_by_label, image_paths or {}))
        photos = cited_images(answer, assistant.chunks_by_label, image_paths or {})
        yield "", chat, assistant, photos, render_steps(steps)

    return respond


async def missing_key(message: str, chat: Chat, assistant: None):
    """Sin API key no hay índice ni modelo: solo se explica qué falta."""
    chat = [*chat, {"role": "user", "content": message}, {"role": "assistant", "content": MISSING_KEY}]
    yield "", chat, assistant, [], TRACE_EMPTY


def new_conversation() -> Outputs:
    """Vacía la caja, el chat, la galería y la traza; sin asistente, el siguiente mensaje empieza con memoria nueva."""
    return "", [], None, [], TRACE_EMPTY


def build_demo(respond: Respond) -> gr.Blocks:
    """La interfaz, sin lanzarla (el tema y el CSS se aplican en launch)."""
    with gr.Blocks(title="Guía Tenerife") as demo:
        gr.HTML(HERO)
        state = gr.State(None)
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, elem_classes="card"):
                chatbot = gr.Chatbot(
                    elem_id="chat",
                    show_label=False,
                    height=540,
                    placeholder=WELCOME,
                    layout="bubble",
                )
                with gr.Row(elem_id="composer"):
                    box = gr.Textbox(
                        elem_id="ask",
                        placeholder="Pregunta sobre Tenerife…",
                        show_label=False,
                        submit_btn=True,
                        autofocus=True,
                        scale=5,
                    )
                    # Icon-only button: the hover tooltip with its description lives in app.css
                    clear = gr.Button("+", elem_id="new-chat", variant="secondary", scale=0, min_width=52)
                gr.Examples(EXAMPLES, inputs=box, label="Prueba con…")
            with gr.Column(scale=2):
                with gr.Column(elem_classes="card"):
                    gr.Markdown("### 📷 Fotos de la guía", elem_classes="card-title")
                    photos = gr.Gallery(
                        elem_id="photos",
                        show_label=False,
                        columns=2,
                        height=260,
                        object_fit="cover",
                        interactive=False,
                    )
                    gr.Markdown(
                        "Las fotos de los lugares que cita la respuesta. Haz clic en una para ampliarla.",
                        elem_classes="hint",
                    )
                with gr.Column(elem_classes="card"):
                    gr.Markdown("### 🔎 Paso a paso", elem_classes="card-title")
                    trace = gr.HTML(TRACE_EMPTY, elem_id="trace")
        gr.HTML(FOOTER)
        box.submit(respond, [box, chatbot, state], [box, chatbot, state, photos, trace])
        clear.click(new_conversation, outputs=[box, chatbot, state, photos, trace])
    return demo


def launch(demo: gr.Blocks) -> None:
    """Lanza la interfaz con el tema y la hoja de estilos de la app."""
    demo.launch(theme=THEME, css_paths=[CSS])


def main() -> None:
    """Todo lo que tiene efectos está aquí: importar app no hace nada."""
    truststore.inject_into_ssl()
    config = LLMConfig.from_env()
    setup_logging()
    if not config.api_key:
        launch(build_demo(missing_key))
        return
    store = build_index(config)
    weather = WeatherService()
    image_paths = export_images()
    launch(build_demo(make_respond(lambda: TenerifeAssistant(config, store, weather), image_paths)))


if __name__ == "__main__":
    main()
