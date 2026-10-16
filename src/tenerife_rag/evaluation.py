"""Evaluación reproducible: recuperación (hit@k, MRR) y recorrido del asistente."""

import json
import re
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

from langchain_core.vectorstores import InMemoryVectorStore

from tenerife_rag.assistant import TenerifeAssistant, tool_calls
from tenerife_rag.chunking import Chunk
from tenerife_rag.citations import has_citation
from tenerife_rag.store import search
from tenerife_rag.weather import canary_today

EVAL_DIR = Path(__file__).parent / "eval"
LIMIT_RE = re.compile(
    r"no (aparece|tengo|encuentro|dispongo|puedo|contiene|incluye)|s[oó]lo puedo|[uú]nicamente puedo",
    re.IGNORECASE,
)


def load_cases(name: str) -> list[dict]:
    """Lee un conjunto de eval/ (questions.json, assistant_cases.json)."""
    return json.loads((EVAL_DIR / name).read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def first_relevant_rank(chunks: list[Chunk], expected: str) -> int | None:
    """Posición (desde 1) del primer chunk que contiene la frase esperada, o None."""
    phrase = _normalize(expected)
    return next((i for i, chunk in enumerate(chunks, 1) if phrase in _normalize(chunk.text)), None)


def evaluate(store: InMemoryVectorStore, questions: list[dict], ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, float]:
    """hit@k para cada k y MRR sobre las preguntas."""
    hits = dict.fromkeys(ks, 0)
    reciprocal = 0.0
    for q in questions:
        chunks = [chunk for chunk, _ in search(store, q["pregunta"], max(ks))]
        rank = first_relevant_rank(chunks, q["esperado"])
        for k in ks:
            hits[k] += rank is not None and rank <= k
        reciprocal += 1 / rank if rank else 0
    n = len(questions)
    return {**{f"hit@{k}": hits[k] / n for k in ks}, "MRR": reciprocal / n}


def expected_date(spec: str, today: date) -> str:
    """"+1" -> mañana respecto a today; una fecha ISO se deja tal cual."""
    return (today + timedelta(days=int(spec))).isoformat() if spec.startswith(("+", "-")) else spec


def score_case(case: dict, answer: str, calls: list[tuple[str, dict]], today: date) -> dict[str, bool]:
    """Una comprobación por cada campo del caso; los campos ausentes no se puntúan."""
    checks = {}
    if "herramientas" in case:
        checks["herramientas"] = {name for name, _ in calls} == set(case["herramientas"])
    if "fecha" in case:
        fecha = expected_date(case["fecha"], today)
        checks["fecha"] = any(name == "get_weather" and args.get("fecha") == fecha for name, args in calls)
    if "palabras" in case:
        checks["palabras"] = all(word.lower() in answer.lower() for word in case["palabras"])
    if "cita" in case:
        checks["cita"] = has_citation(answer) == case["cita"]
    if "limite" in case:
        checks["limite"] = (LIMIT_RE.search(answer) is not None) == case["limite"]
    return checks


async def run_case(make_assistant: Callable[[], TenerifeAssistant], case: dict, today: date) -> dict:
    """Ejecuta un caso en una conversación nueva y lo puntúa."""
    assistant = make_assistant()
    answer = await assistant.ask(case["pregunta"])
    checks = score_case(case, answer, tool_calls(assistant.last_result), today)
    return {"id": case["id"], "respuesta": answer, **checks, "ok": all(checks.values())}


async def evaluate_assistant(
    make_assistant: Callable[[], TenerifeAssistant],
    cases: list[dict],
    runs: int = 3,
    today_fn: Callable[[], date] = canary_today,
) -> list[dict]:
    """Cada caso `runs` veces: con temperatura > 0, una sola ejecución no dice nada."""
    return [await run_case(make_assistant, case, today_fn()) for case in cases for _ in range(runs)]


def pass_rates(results: list[dict]) -> dict[str, float]:
    """Proporción de ejecuciones correctas por caso."""
    by_case: dict[str, list[bool]] = {}
    for result in results:
        by_case.setdefault(result["id"], []).append(result["ok"])
    return {case_id: sum(oks) / len(oks) for case_id, oks in by_case.items()}
