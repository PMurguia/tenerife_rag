from tenerife_rag.extraction import _detect_marker, _Fragment, _merge_rows


def fragment(text: str, y: float, x: float, page: int = 1) -> _Fragment:
    return _Fragment(page, y, x, [{"text": text, "flags": 0}])


def test_detect_marker_needs_text_after_bullet():
    assert _detect_marker("• Santa Cruz", 90.6) == "•"
    assert _detect_marker("-", 108.6) is None


def test_detect_marker_only_counts_inside_its_indent_range():
    assert _detect_marker("o Playa de Benijo", 126.6) == "o"
    assert _detect_marker("o buscad vosotros mismos", 108.6) is None  # conjunction at body indent


def test_merges_bullet_and_text_on_the_same_row():
    rows = _merge_rows([fragment("Aparcar en el Parque.", 235.8, 126.6), fragment("-", 234.0, 108.6)])

    assert [(r.text, r.x, r.y) for r in rows] == [("- Aparcar en el Parque.", 108.6, 234.0)]


def test_does_not_merge_rows_four_points_apart_or_more():
    rows = _merge_rows([fragment("-", 234.0, 108.6), fragment("Otro párrafo.", 238.0, 126.6)])

    assert [r.text for r in rows] == ["-", "Otro párrafo."]


def test_does_not_merge_across_pages():
    rows = _merge_rows([fragment("-", 100.0, 108.6), fragment("Texto.", 101.0, 126.6, page=2)])

    assert [r.page for r in rows] == [1, 2]
