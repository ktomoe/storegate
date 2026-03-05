"""Unit tests for logger module."""
import logging

import pytest

import storegate.logger as logger


@pytest.fixture(autouse=True)
def _clean_file_handlers():
    """Remove any FileHandlers added during a test so they don't bleed into others."""
    yield
    for h in list(logger._logger.handlers):
        if isinstance(h, logging.FileHandler):
            h.close()
            logger._logger.removeHandler(h)


def _file_handler_count() -> int:
    return sum(1 for h in logger._logger.handlers if isinstance(h, logging.FileHandler))


# ---------------------------------------------------------------------------
# add_file_handler — basic behaviour
# ---------------------------------------------------------------------------

def test_add_file_handler_adds_one_handler(tmp_path):
    logger.add_file_handler(str(tmp_path / 'run.log'))
    assert _file_handler_count() == 1


def test_add_file_handler_same_path_is_noop(tmp_path):
    path = str(tmp_path / 'run.log')
    logger.add_file_handler(path)
    logger.add_file_handler(path)
    assert _file_handler_count() == 1


def test_add_file_handler_same_path_via_relative(tmp_path, monkeypatch):
    """Duplicate detection works even when the path form differs (abs vs rel)."""
    monkeypatch.chdir(tmp_path)
    logger.add_file_handler('run.log')
    logger.add_file_handler(str(tmp_path / 'run.log'))
    assert _file_handler_count() == 1


def test_add_file_handler_different_paths_adds_two(tmp_path):
    logger.add_file_handler(str(tmp_path / 'a.log'))
    logger.add_file_handler(str(tmp_path / 'b.log'))
    assert _file_handler_count() == 2


def test_add_file_handler_many_calls_still_one(tmp_path):
    path = str(tmp_path / 'run.log')
    for _ in range(10):
        logger.add_file_handler(path)
    assert _file_handler_count() == 1


# ---------------------------------------------------------------------------
# _header_line — long message does not produce negative padding
# ---------------------------------------------------------------------------

def test_header_line_long_message_does_not_raise():
    """Messages longer than 78 chars must not crash or produce garbage."""
    result = logger._header_line('x' * 100, '=')
    assert isinstance(result, str)


def test_header_line_long_message_contains_message():
    msg = 'a' * 100
    result = logger._header_line(msg, '=')
    assert msg in result


def test_header_line_long_message_no_negative_chars():
    """With pad clamped to 0, the result is just ' message '."""
    result = logger._header_line('x' * 100, '=')
    assert result == ' ' + 'x' * 100 + ' '


def test_header_line_normal_message_has_padding():
    result = logger._header_line('hello', '=')
    assert result.startswith('=')
    assert result.endswith('=')
