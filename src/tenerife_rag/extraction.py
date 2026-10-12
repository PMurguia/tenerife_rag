from dataclasses import dataclass, field

import pymupdf

@dataclass
class Line:
    page: int
    y: float
    x: float
    text: str
    bold: bool
    marker: str | None

MARKER_X_RANGE = {"•": (85, 100), "-": (100, 118), "o": (118, 136), "▪": (154, 172)}
ROW_TOLERANCE = 4.0

def _detect_marker(text: str, x: float) -> str | None:
    head, _, rest = text.partition(" ")
    if head in MARKER_X_RANGE and rest:
        low, high = MARKER_X_RANGE[head]
        if low <= x < high:
            return head
    return None


def _is_bold(spans: list[dict]) -> bool:
    visible = [(len(s["text"].strip()), bool(s["flags"] & pymupdf.TEXT_FONT_BOLD)) for s in spans]
    total = sum(n for n, _ in visible)
    return total > 0 and sum(n for n, bold in visible if bold) * 2 >= total


@dataclass
class _Fragment:

    page: int
    y: float
    x: float
    spans: list[dict] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join("".join(s["text"] for s in self.spans).split())


def _merge_rows(fragments: list[_Fragment], tolerance: float = ROW_TOLERANCE) -> list[_Fragment]:
    rows: list[list[_Fragment]] = []
    for fragment in sorted(fragments, key=lambda f: (f.page, f.y)):
        row = rows[-1] if rows else None
        if row and row[0].page == fragment.page and abs(fragment.y - row[0].y) < tolerance:
            row.append(fragment)
        else:
            rows.append([fragment])
    merged = []
    for row in rows:
        row.sort(key=lambda f: f.x)
        spans = [s for f in row for s in (*f.spans, {"text": " ", "flags": 0})]  # space between fragments
        merged.append(_Fragment(row[0].page, min(f.y for f in row), row[0].x, spans))
    return merged


def extract_lines(path: str) -> list[Line]:
    fragments = []
    with pymupdf.open(path) as pdf:
        for page in pdf:
            for block in page.get_text("dict")["blocks"]:
                for raw in block.get("lines", []):
                    fragment = _Fragment(page.number + 1, raw["bbox"][1], raw["bbox"][0], raw["spans"])
                    if fragment.text:
                        fragments.append(fragment)
    return [
        Line(row.page, row.y, row.x, row.text, _is_bold(row.spans), _detect_marker(row.text, row.x))
        for row in _merge_rows(fragments)
    ]
