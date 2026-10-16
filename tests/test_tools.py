import logging
from pathlib import Path
from typing import get_args

import pytest

from conftest import FakeEmbeddings
from tenerife_rag.chunking import Chunk, structural_chunks
from tenerife_rag.extraction import extract_lines
from tenerife_rag.store import build_store, search
from tenerife_rag.tools import Category, Zone, guide_filter, log_tool_call, logger, setup_logging


@pytest.fixture
def restore_logger():
    """Undo the handlers and level that setup_logging adds to the module logger."""
    handlers, level = list(logger.handlers), logger.level
    yield
    for handler in logger.handlers:
        if handler not in handlers:
            handler.close()
    logger.handlers = handlers
    logger.setLevel(level)


def test_log_tool_call_logs_success(caplog):
    caplog.set_level(logging.INFO, logger=logger.name)

    log_tool_call("get_weather", '{"fecha": "2026-10-13"}', {"ok": True})

    assert caplog.records[0].levelno == logging.INFO
    assert caplog.messages == ['tool=get_weather args={"fecha": "2026-10-13"} ok=True error=None']


def test_log_tool_call_logs_error_code(caplog):
    caplog.set_level(logging.INFO, logger=logger.name)

    log_tool_call("get_weather", '{"fecha": "2026-10-01"}', {"ok": False, "error": "fecha_pasada"})

    assert caplog.messages == ['tool=get_weather args={"fecha": "2026-10-01"} ok=False error=fecha_pasada']


def test_setup_logging_creates_missing_directory(tmp_path, restore_logger):
    log_dir = tmp_path / "nested" / "logs"

    setup_logging(log_dir)

    assert log_dir.is_dir()


def test_setup_logging_writes_tool_calls_to_file(tmp_path, restore_logger):
    setup_logging(tmp_path)

    log_tool_call("get_weather", "{}", {"ok": False, "error": "timeout"})

    lines = (tmp_path / "tool_calls.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert lines[0].endswith(" - INFO - tool=get_weather args={} ok=False error=timeout")


def test_setup_logging_twice_does_not_duplicate_lines(tmp_path, restore_logger):
    setup_logging(tmp_path)
    setup_logging(tmp_path)

    log_tool_call("get_weather", "{}", {"ok": True})

    lines = (tmp_path / "tool_calls.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_setup_logging_with_different_dirs_writes_to_both(tmp_path, restore_logger):
    setup_logging(tmp_path / "a")
    setup_logging(tmp_path / "b")

    log_tool_call("get_weather", "{}", {"ok": True})

    for name in ("a", "b"):
        assert len((tmp_path / name / "tool_calls.log").read_text(encoding="utf-8").splitlines()) == 1


# --- zone / category filter -------------------------------------------------

GUIDE = Path(__file__).parents[1] / "src" / "tenerife_rag" / "data" / "TENERIFE.pdf"


def test_no_filter_returns_none():
    assert guide_filter(None, None) is None


def test_zone_is_the_first_level_of_the_path():
    restaurant = Chunk(id="1", text="...", kind="restaurante", path=["Zona Norte", "Sitios para comer", "La Baranda"])

    assert guide_filter("Zona Norte", "comer")(restaurant)
    assert not guide_filter("Zona Sur", None)(restaurant)
    assert not guide_filter("Zona Norte", None)(Chunk(id="2", text="...", path=[]))


def test_category_splits_restaurants_from_the_rest():
    restaurant = Chunk(id="1", text="...", kind="restaurante", path=["Zona Norte"])
    beach = Chunk(id="2", text="...", kind="poi", path=["Zona Norte"])

    assert [guide_filter(None, "comer")(c) for c in (restaurant, beach)] == [True, False]
    assert [guide_filter(None, "visitar")(c) for c in (restaurant, beach)] == [False, True]


@pytest.fixture(scope="module")
def guide_store():
    if not GUIDE.exists():
        pytest.skip("TENERIFE.pdf not available")
    return build_store(structural_chunks(extract_lines(str(GUIDE))), FakeEmbeddings())


@pytest.mark.parametrize("zona", get_args(Zone))
def test_every_zone_filters_the_real_guide(guide_store, zona):
    hits = search(guide_store, "playa", top_k=50, where=guide_filter(zona, None))

    assert hits  # a renamed heading in the guide would silently empty the filter
    assert all(chunk.path[0] == zona for chunk, _ in hits)


@pytest.mark.parametrize("categoria", get_args(Category))
def test_every_category_filters_the_real_guide(guide_store, categoria):
    hits = search(guide_store, "sitio", top_k=50, where=guide_filter(None, categoria))

    assert hits
    assert all((chunk.kind == "restaurante") == (categoria == "comer") for chunk, _ in hits)


def test_northern_restaurants_are_found_by_zone_and_category(guide_store):
    hits = search(guide_store, "restaurante", top_k=50, where=guide_filter("Zona Norte", "comer"))

    assert len(hits) == 11
