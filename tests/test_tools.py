import logging

import pytest

from tenerife_rag.tools import log_tool_call, logger, setup_logging


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
