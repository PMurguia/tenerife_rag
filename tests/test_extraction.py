from tenerife_rag.extraction import Line, _detect_marker, _merge_orphan_markers


def line(text: str, y: float, x: float = 126.6, page: int = 1, bold: bool = False) -> Line:
    return Line(page, y, x, text, bold, _detect_marker(text))


def test_detect_marker_needs_text_after_bullet():
    assert _detect_marker("• Santa Cruz") == "•"
    assert _detect_marker("-") is None
    assert _detect_marker("ola de calor") is None


def test_merges_orphan_marker_with_next_line():
    merged = _merge_orphan_markers([line("-", 234.0, x=108.6), line("Aparcar en el Parque.", 235.8)])

    assert merged == [Line(1, 234.0, 108.6, "- Aparcar en el Parque.", False, "-")]


def test_does_not_merge_across_pages():
    lines = [line("-", 780.0, x=108.6), line("Texto.", 70.0, page=2)]

    assert _merge_orphan_markers(lines) == lines


def test_does_not_merge_when_next_line_is_far_below():
    lines = [line("-", 234.0, x=108.6), line("Otro párrafo.", 260.0)]

    assert _merge_orphan_markers(lines) == lines


def test_leaves_lines_with_inline_marker_untouched():
    lines = [line("o Playa de Benijo", 100.0), line("Arena negra.", 115.0, x=144.6)]

    assert _merge_orphan_markers(lines) == lines
