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


@pytest.fixture(autouse=True)
def _restore_log_level():
    """Restore the logger level after each test."""
    original = logger._logger.level
    yield
    logger._logger.setLevel(original)


class _ListHandler(logging.Handler):
    """Handler that appends formatted messages to a list."""
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(self.format(record))


@pytest.fixture
def capture():
    """Fixture that captures log output from the storegate logger."""
    handler = _ListHandler()
    handler.setLevel(logging.DEBUG)
    logger._logger.addHandler(handler)
    yield handler.messages
    logger._logger.removeHandler(handler)


def _file_handler_count() -> int:
    return sum(1 for h in logger._logger.handlers if isinstance(h, logging.FileHandler))


# ---------------------------------------------------------------------------
# set_level
# ---------------------------------------------------------------------------

def test_set_level_with_int():
    logger.set_level(logging.DEBUG)
    assert logger._logger.level == logging.DEBUG


def test_set_level_with_string():
    logger.set_level('debug')
    assert logger._logger.level == logging.DEBUG


def test_set_level_with_string_case_insensitive():
    logger.set_level('WARNING')
    assert logger._logger.level == logging.WARNING


def test_set_level_with_unknown_string_falls_back_to_info():
    logger.set_level('nonexistent')
    assert logger._logger.level == logging.INFO


# ---------------------------------------------------------------------------
# debug / info / warn / error
# ---------------------------------------------------------------------------

def test_debug_emits_message(capture):
    logger.set_level(logger.DEBUG)
    logger.debug('test debug %s', 'msg')
    assert any('test debug msg' in m for m in capture)


def test_info_emits_message(capture):
    logger.info('test info %s', 'msg')
    assert any('test info msg' in m for m in capture)


def test_warning_emits_message(capture):
    logger.warning('test warn %s', 'msg')
    assert any('test warn msg' in m for m in capture)


def test_error_emits_message(capture):
    logger.error('test error %s', 'msg')
    assert any('test error msg' in m for m in capture)


# ---------------------------------------------------------------------------
# counter
# ---------------------------------------------------------------------------

def test_counter_zero_is_noop(capture):
    logger.counter(0, 10)
    assert capture == []


def test_counter_emits_at_divisible_count(capture):
    logger.counter(4, 10, divide=2)
    assert any('(4/10)' in m for m in capture)


def test_counter_emits_at_max_count(capture):
    logger.counter(10, 10, divide=3)
    assert any('(10/10)' in m for m in capture)


def test_counter_skips_non_divisible_count(capture):
    logger.counter(3, 10, divide=2)
    assert capture == []


def test_counter_with_message(capture):
    logger.counter(5, 10, divide=5, message='loading')
    assert any('(5/10)' in m and 'loading' in m for m in capture)


def test_counter_without_message(capture):
    logger.counter(1, 5)
    assert any('events processed' in m for m in capture)


# ---------------------------------------------------------------------------
# _header_line
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


def test_header_line_empty_message():
    result = logger._header_line('', '=')
    assert '=' in result
    assert ' ' not in result


def test_header_line_odd_length_message_padded():
    """Odd-length messages are padded with a space to make them even."""
    result = logger._header_line('abc', '-')
    assert 'abc' in result


# ---------------------------------------------------------------------------
# header1 / header2 / header3
# ---------------------------------------------------------------------------

def test_header1_default_level(capture):
    logger.header1('title')
    assert any('title' in m for m in capture)
    assert any('=' * 80 == m for m in capture)


def test_header1_custom_level():
    messages = []
    logger.header1('title', level=messages.append)
    assert any('=' * 80 == m for m in messages)
    assert any('title' in m for m in messages)


def test_header2_default_level(capture):
    logger.header2('section')
    assert any('section' in m for m in capture)


def test_header2_custom_level():
    messages = []
    logger.header2('section', level=messages.append)
    assert len(messages) == 1
    assert 'section' in messages[0]
    assert '-' in messages[0]


def test_header3_default_level(capture):
    logger.header3('sub')
    assert any('sub' in m for m in capture)


def test_header3_custom_level():
    messages = []
    logger.header3('sub', level=messages.append)
    assert len(messages) == 1
    assert 'sub' in messages[0]
    assert '=' in messages[0]


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------

def test_table_basic_output(capture):
    logger.table(['col1', 'col2'], [['a', 'b'], ['c', 'd']])
    text = '\n'.join(capture)
    assert 'col1' in text
    assert 'col2' in text


def test_table_with_header(capture):
    logger.table(['x'], [['v']], header='my header')
    text = '\n'.join(capture)
    assert 'my header' in text


def test_table_with_footer(capture):
    logger.table(['x'], [['v']], footer='my footer')
    text = '\n'.join(capture)
    assert 'my footer' in text


def test_table_with_header_and_footer(capture):
    logger.table(['x'], [['v']], header='H', footer='F')
    text = '\n'.join(capture)
    assert 'H' in text
    assert 'F' in text


def test_table_with_separator_row(capture):
    """Rows equal to '-' produce separator lines."""
    logger.table(['x'], [['a'], '-', ['b']])
    text = '\n'.join(capture)
    assert 'a' in text
    assert 'b' in text


def test_table_long_value_truncated(capture):
    long_val = 'x' * 50
    logger.table(['col'], [[long_val]], max_length=10)
    text = '\n'.join(capture)
    assert 'x' * 50 not in text
    assert 'x' * 10 in text


def test_table_column_width_adapts_to_data(capture):
    logger.table(['a'], [['longvalue']])
    text = '\n'.join(capture)
    assert 'longvalue' in text


# ---------------------------------------------------------------------------
# log_call
# ---------------------------------------------------------------------------

def test_log_call_returns_function_result():
    class Dummy:
        @logger.log_call
        def method(self, x):
            return x * 2

    logger.set_level(logger.DEBUG)
    assert Dummy().method(5) == 10


def test_log_call_preserves_function_name():
    class Dummy:
        @logger.log_call
        def my_method(self):
            pass

    assert Dummy.my_method.__name__ == 'my_method'


def test_log_call_emits_start_and_end(capture):
    class Dummy:
        @logger.log_call
        def work(self):
            return 42

    logger.set_level(logger.DEBUG)
    Dummy().work()
    text = '\n'.join(capture)
    assert 'START' in text
    assert 'END' in text
    assert 'Dummy' in text


def test_log_call_logs_args_and_return(capture):
    class Dummy:
        @logger.log_call
        def add(self, a, b=0):
            return a + b

    logger.set_level(logger.DEBUG)
    result = Dummy().add(3, b=7)
    assert result == 10
    text = '\n'.join(capture)
    assert 'args=' in text
    assert 'return=' in text


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
