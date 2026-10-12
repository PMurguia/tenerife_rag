from dataclasses import dataclass

import pymupdf

@dataclass
class Line:
    page: int
    y: float
    x: float
    text: str
    bold: bool
    marker: str | None
    
MARKERS = {"•", "o", "▪", "-"}

def _detect_marker(text: str) -> str | None:
    head, _, rest = text.partition(" ")
    return head if head in MARKERS and rest else None


def _is_bold(spans: list[dict]) -> bool:
    visible = [(len(s["text"].strip()), bool(s["flags"] & pymupdf.TEXT_FONT_BOLD)) for s in spans]
    total = sum(n for n, _ in visible)
    return total > 0 and sum(n for n, bold in visible if bold) * 2 >= total


def extract_lines(path: str) -> list[Line]:
    lines = []
    with pymupdf.open(path) as pdf:
        for page in pdf:
            for block in page.get_text("dict")["blocks"]:
                for raw in block.get("lines", []):
                    text = " ".join("".join(s["text"] for s in raw["spans"]).split())
                    if not text:
                        continue
                    x0, y0, *_ = raw["bbox"]
                    lines.append(Line(page.number + 1, y0, x0, text, _is_bold(raw["spans"]), _detect_marker(text)))
    return _merge_orphan_markers(sorted(lines, key=lambda line: (line.page, line.y)))


def _merge_orphan_markers(lines: list[Line], max_gap: float = 5.0) -> list[Line]:
    merged: list[Line] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        if (
            line.text in MARKERS
            and nxt is not None
            and nxt.page == line.page
            and abs(nxt.y - line.y) <= max_gap
            and nxt.x > line.x
        ):
            # keep the marker's position: x is where the bullet sits, as in "• ..." lines
            merged.append(Line(line.page, line.y, line.x, f"{line.text} {nxt.text}", nxt.bold, line.text))
            i += 2
        else:
            merged.append(line)
            i += 1
    return merged