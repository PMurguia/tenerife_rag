"""Traza de un turno: cada paso que se activa (memoria, modelo, herramientas, respuesta, fotos)."""

import time
from dataclasses import dataclass, field
from typing import Literal

StepKind = Literal["memoria", "llm", "rag", "tiempo", "respuesta", "fotos", "error"]


@dataclass
class Step:
    """Un paso del turno, con sus detalles y lo que tardó (si aplica)."""

    kind: StepKind
    title: str
    details: list[str] = field(default_factory=list)
    ms: float | None = None
    ok: bool = True


def elapsed_ms(start: float) -> float:
    """Milisegundos desde start (un valor de time.perf_counter)."""
    return round((time.perf_counter() - start) * 1000, 1)
