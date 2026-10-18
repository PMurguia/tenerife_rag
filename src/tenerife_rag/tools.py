"""Herramientas del agente (búsqueda en la guía y tiempo) y su registro en el log."""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from agents import ModelBehaviorError, function_tool
from agents.tool_context import ToolContext
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import Field

from tenerife_rag.chunking import Chunk
from tenerife_rag.store import search
from tenerife_rag.trace import Step, elapsed_ms
from tenerife_rag.weather import PlaceId, WeatherService

logger = logging.getLogger("tenerife_rag.tools")
package_logger = logging.getLogger("tenerife_rag")  # parent of tools and weather: receives calls and retries

Zone = Literal["Zona Norte", "Zona Sur"]
Category = Literal["comer", "visitar"]


def log_tool_call(name: str, arguments: str, result: dict) -> None:
    """Registra una llamada a herramienta en logs/tool_calls.log."""
    logger.info("tool=%s args=%s ok=%s error=%s", name, arguments, result.get("ok"), result.get("error"))


def setup_logging(log_dir="logs") -> None:
    """Configura logs/tool_calls.log: cada llamada a herramienta y cada reintento fallido de Open-Meteo.

    Llamarla dos veces no duplica el handler.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "tool_calls.log"
    if any(
        isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.absolute()
        for h in package_logger.handlers
    ):
        return
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO)


@dataclass
class TurnContext:
    """Estado de un turno: índice, servicio del tiempo y etiquetas de los chunks citables."""

    store: InMemoryVectorStore
    weather_service: WeatherService
    sources: dict[str, str] = field(default_factory=dict)
    chunks_by_label: dict[str, Chunk] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)


def _reply(ctx: ToolContext[TurnContext], result: dict) -> str:
    log_tool_call(ctx.tool_name, ctx.tool_arguments, result)
    return json.dumps(result, ensure_ascii=False)


def tool_failure(ctx: ToolContext[TurnContext], error: Exception) -> str:
    """Respuesta de error para el modelo cuando una herramienta falla."""
    code = "argumentos_invalidos" if isinstance(error, ModelBehaviorError) else "error_interno"
    ctx.context.steps.append(Step("error", f"{ctx.tool_name} falló", [f"{code}: {error}"], ok=False))
    return _reply(ctx, {"ok": False, "error": code, "message": str(error)})


def weather_step(fecha: date, lugar: str | None, result: dict, ms: float) -> Step:
    """Paso de la traza para una consulta del tiempo: la predicción o el error."""
    asked = f"fecha {fecha.isoformat()} · lugar {lugar or 'isla (por defecto)'}"
    if not result.get("ok"):
        return Step("tiempo", "Open-Meteo · get_weather", [asked, f"{result['error']}: {result['message']}"], ms, False)
    forecast = f"{result['estado']}, {result['temp_min_c']}–{result['temp_max_c']} °C"
    source = f"fuente: {result.get('fuente', 'Open-Meteo')}"
    return Step("tiempo", "Open-Meteo · get_weather", [asked, forecast, source], ms)


def guide_filter(zona: Zone | None, categoria: Category | None):
    """Filtro de chunks por zona y categoría, o None si no hay filtro."""
    if zona is None and categoria is None:
        return None

    def where(chunk: Chunk) -> bool:
        if zona is not None and chunk.path[:1] != [zona]:
            return False
        return categoria is None or (chunk.kind == "restaurante") == (categoria == "comer")

    return where


@function_tool(failure_error_function=tool_failure)
def search_tenerife_guide(
    ctx: ToolContext[TurnContext],
    query: str,
    zona: Zone | None,
    categoria: Category | None,
    k: Annotated[int, Field(ge=1, le=8)],
) -> str:
    """Busca en la guía de Tenerife los fragmentos más relevantes para la consulta.

    Args:
        query: Qué buscar, con las palabras del usuario o el nombre del lugar.
        zona: "Zona Norte" o "Zona Sur" si la pregunta la deja clara; si no, null.
        categoria: "comer" para restaurantes, "visitar" para el resto; null si no está clara.
        k: Número de fragmentos que devolver.
    """
    turn = ctx.context
    start = time.perf_counter()
    results, found = [], []
    for chunk, score in search(turn.store, query, k, where=guide_filter(zona, categoria)):
        label = turn.sources.setdefault(chunk.id, f"S{len(turn.sources) + 1}")
        turn.chunks_by_label[label] = chunk
        results.append({"id": label, "texto": chunk.text, "cita": chunk.citation})
        found.append(f"{label} · {chunk.citation} · score {score:.3f}")
    asked = f"«{query}» · zona {zona or '—'} · categoría {categoria or '—'} · k={k}"
    details = [asked, *(found or ["sin resultados"])]
    turn.steps.append(Step("rag", "Búsqueda en la guía · search_tenerife_guide", details, elapsed_ms(start)))
    return _reply(ctx, {"ok": True, "resultados": results})


@function_tool(failure_error_function=tool_failure)
def get_weather(ctx: ToolContext[TurnContext], fecha: date, lugar: PlaceId | None) -> str:
    """Predicción del tiempo en Tenerife para un día.

    Args:
        fecha: Día de la predicción, en formato AAAA-MM-DD.
        lugar: Zona de la isla; null para la isla en general.
    """
    start = time.perf_counter()
    result = ctx.context.weather_service.get_weather(fecha.isoformat(), lugar)
    ctx.context.steps.append(weather_step(fecha, lugar, result, elapsed_ms(start)))
    return _reply(ctx, result)
