"""Memoria de la conversación con un presupuesto de tokens."""

from dataclasses import dataclass

import tiktoken

from tenerife_rag.citations import strip_citations


@dataclass
class Turn:
    """Una pregunta y su respuesta."""

    user: str
    assistant: str


class ConversationMemory:
    """Turnos anteriores; descarta los más antiguos si se pasan del presupuesto de tokens."""

    def __init__(self, budget_tokens: int = 3000):
        self.turns = []
        self.budget = budget_tokens
        self.dropped = 0
        self._encoding = tiktoken.get_encoding("o200k_base")

    def add(self, user: str, assistant: str) -> None:
        """Guarda el turno sin las citas y recorta si hace falta."""
        self.turns.append(Turn(user, strip_citations(assistant)))
        self._trim()

    def as_input(self) -> list[dict]:
        """Los turnos como mensajes de entrada para el modelo."""
        return [
            m
            for t in self.turns
            for m in ({"role": "user", "content": t.user}, {"role": "assistant", "content": t.assistant})
        ]

    def tokens(self) -> int:
        """Tokens que ocupan los turnos guardados."""
        return sum(len(self._encoding.encode(t.user + t.assistant)) for t in self.turns)

    def _trim(self) -> None:
        while len(self.turns) > 1 and self.tokens() > self.budget:
            self.turns.pop(0)
            self.dropped += 1
