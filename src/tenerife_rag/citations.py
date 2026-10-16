"""Una sola gramática para las citas [S1], [S1][S2], [S1, S2] y [S1; S2]."""

import re

_ITEM = r"S\d+"
CITATION_GROUP_RE = re.compile(rf"\[\s*{_ITEM}(?:\s*[,;]\s*{_ITEM})*\s*\]")
_STRIP_RE = re.compile(rf"\s*{CITATION_GROUP_RE.pattern}")


def cited_labels(text: str) -> set[str]:
    """Ids citados en el texto: "[S1, S2]" -> {"S1", "S2"}."""
    return {label for group in CITATION_GROUP_RE.findall(text) for label in re.findall(_ITEM, group)}


def has_citation(text: str) -> bool:
    """True si el texto cita al menos un fragmento."""
    return CITATION_GROUP_RE.search(text) is not None


def strip_citations(text: str) -> str:
    """El texto sin citas ni el espacio que las precede."""
    return _STRIP_RE.sub("", text)
