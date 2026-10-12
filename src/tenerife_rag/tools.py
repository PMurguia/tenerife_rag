import logging
from pathlib import Path
import json
from dataclasses import dataclass, field

from agents import ModelBehaviorError
from agents.tool_context import ToolContext
from langchain_core.vectorstores import InMemoryVectorStore

from tenerife_rag.chunking import Chunk
from tenerife_rag.weather import WeatherService

from datetime import date
from typing import Annotated, Literal

from agents import function_tool
from pydantic import Field

from tenerife_rag.store import search
from tenerife_rag.weather import PlaceId

logger = logging.getLogger("tenerife_rag.tools")

Zone = Literal["Zona Norte", "Zona Sur"]
Category = Literal["comer", "visitar"]

def log_tool_call(name: str, arguments: str, result: dict) -> None:
    logger.info("tool=%s args=%s ok=%s error=%s", name, arguments, result.get("ok"), result.get("error"))
    
def setup_logging(log_dir="logs") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "tool_calls.log"
    # Idempotent: re-running a notebook cell must not attach a second handler to the same file
    if any(
        isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.absolute()
        for h in logger.handlers
    ):
        return
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

@dataclass
class TurnContext:
    store: InMemoryVectorStore
    weather_service: WeatherService
    sources: dict[str, str] = field(default_factory=dict)
    chunks_by_label: dict[str, Chunk] = field(default_factory=dict)

def _reply(ctx: ToolContext[TurnContext], result: dict) -> str:
    log_tool_call(ctx.tool_name, ctx.tool_arguments, result)
    return json.dumps(result, ensure_ascii=False)

def tool_failure(ctx: ToolContext[TurnContext], error: Exception) -> str:
    code = "argumentos_invalidos" if isinstance(error, ModelBehaviorError) else "error_interno"
    return _reply(ctx, {"ok": False, "error": code, "message": str(error)})

def guide_filter(zona: Zone | None, categoria: Category | None):
    if zona is None and categoria is None:
        return None

    def where(chunk: Chunk) -> bool:
        if zona is not None and zona not in chunk.path:
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
    turn = ctx.context
    results = []
    for chunk, _ in search(turn.store, query, k, where=guide_filter(zona, categoria)):
        label = turn.sources.setdefault(chunk.id, f"S{len(turn.sources)+1}")
        turn.chunks_by_label[label] = chunk
        results.append({"id": label, "texto": chunk.text, "cita": chunk.citation})
    return _reply(ctx, {"ok": True, "resultados": results})

@function_tool(failure_error_function=tool_failure)
def get_weather(ctx: ToolContext[TurnContext], fecha: date, lugar: PlaceId | None) -> str:
    return _reply(ctx, ctx.context.weather_service.get_weather(fecha.isoformat(), lugar))