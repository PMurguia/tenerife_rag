from dataclasses import asdict, dataclass, field
from typing import Callable
from langchain_core.documents import Document


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
        