from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import re

from langchain_core.documents import Document

from tenerife_rag.extraction import Line

ZONES = {"ZONA NORTE": "Zona Norte", "ZONA SUR": "Zona Sur", "OTRAS VAINAS": "Otras vainas"}
ROUTE_RE = re.compile(r"Aparcar|Caminar|Callejear|Al bajar|Al ir|En |Si vais")
CURIOSITIES_RE = re.compile(r"Curiosidades:?$")
LINK_NOISE = ["[vídeo - ubicación]", "[ubicación]", "[vídeo]", "[web]", "[Tripadvisor]", "(ubicación)"]
TITLE_END_RE = re.compile(r"[(\[:,.]")

@dataclass
class Chunk:
    """Representa un fragmento de texto con metadatos asociados."""
    id: str
    text: str
    kind: str = "ventana"
    path: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    
    @property
    def header(self) -> str:
        return " > ".join(self.path)
    
    def embed_text(self) -> str:
        return f"{self.header}\n{self.text}" if self.header else self.text
    
    @property
    def citation(self) -> str:
        pages = "-".join(str(p) for p in sorted({self.pages[0], self.pages[-1]})) if self.pages else "?"
        
        return f"{self.header or 'Fragmento ' + self.id} (p. {pages})"
    
    def to_document(self, text_fn: Callable[["Chunk"], str] = embed_text) -> Document:
        return Document(id=self.id, page_content=text_fn(self), metadata=asdict(self))
    
    @classmethod
    def from_document(cls, doc: Document) -> "Chunk":
        return cls(**doc.metadata)


def _clean(text: str) -> str:
    for noise in LINK_NOISE:
        text = text.replace(noise, "")
    text = " ".join(text.split())
    return re.sub(r" ([.,;)])", r"\1", text)


def _title(body: str) -> str:
    return TITLE_END_RE.split(_clean(body), maxsplit=1)[0].strip()


@dataclass
class _Chunker:

    chunks: list[Chunk] = field(default_factory=list)
    zone: str | None = None
    l1: str | None = None
    l2: str | None = None
    section: Chunk | None = None
    current: Chunk | None = None
    indent: float = 0.0
    in_curiosities: bool = False

    def feed(self, line: Line) -> None:
        body = line.text.partition(" ")[2] if line.marker else line.text
        if line.bold and line.text.upper() in ZONES:
            self._zone(line)
        elif line.marker == "•":
            self._locality(line, body)
        elif line.bold and CURIOSITIES_RE.match(line.text) and self.section is not None:
            self._open(line, "curiosidades", [self.zone, self.l1, "Curiosidades"], "")
            self.in_curiosities = True
        elif line.marker == "o":
            self._place(line, body)
        elif line.marker == "▪":
            self._open(line, "restaurante", [self.zone, self.l1, self.l2, _title(body)], body)
        elif self.current is not None:
            self._continue(line)

    def _zone(self, line: Line) -> None:
        self.zone, self.l1, self.l2 = ZONES[line.text.upper()], None, None
        self.section = self.current = None
        self.in_curiosities = False

    def _locality(self, line: Line, body: str) -> None:
        self.l1, self.l2, self.in_curiosities = _title(body), None, False
        self.section = self._open(line, "seccion", [self.zone, self.l1], body)

    def _place(self, line: Line, body: str) -> None:
        if self.in_curiosities:
            self._append(body, line.page)
            self.indent = line.x
            return
        self.l2 = _title(body)
        kind = "ruta" if ROUTE_RE.match(body) else "poi"
        self._open(line, kind, [self.zone, self.l1, self.l2], body)

    def _continue(self, line: Line) -> None:
        if self.current is not self.section and self.section is not None and line.x < self.indent:
            self.current, self.indent, self.in_curiosities = self.section, 0.0, False
        self._append(line.text, line.page)

    def _open(self, line: Line, kind: str, path: list[str | None], text: str) -> Chunk:
        self.current = Chunk(str(len(self.chunks)), text, kind, [p for p in path if p], [line.page])
        self.chunks.append(self.current)
        self.indent = line.x
        return self.current

    def _append(self, text: str, page: int) -> None:
        self.current.text += " " + text
        if page not in self.current.pages:
            self.current.pages.append(page)


def structural_chunks(lines: list[Line]) -> list[Chunk]:
    chunker = _Chunker()
    for line in lines:
        chunker.feed(line)
    for chunk in chunker.chunks:
        chunk.text = _clean(chunk.text)
    return chunker.chunks
        