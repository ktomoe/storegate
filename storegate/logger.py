"""Logger module.

In principle, modules in storegate library use this logger.

Examples:
    >>> from storegate import logger
    >>> logger.set_level(logger.INFO)
    >>> logger.add_file_handler('storegate.log')
    >>> logger.debug("This message is not printed at INFO level.")
"""
from __future__ import annotations

import logging
import functools
from typing import Any, Callable, TypeVar

DEBUG = logging.DEBUG        # 10
INFO = logging.INFO          # 20
WARN = logging.WARNING       # 30
ERROR = logging.ERROR        # 40
DISABLED = logging.CRITICAL + 10  # 60

_logger = logging.getLogger('storegate')
_logger.setLevel(logging.INFO)
_logger.propagate = False

_fmt = logging.Formatter('%(asctime)s [%(levelname).1s] %(message)s',
                          datefmt='%Y-%m-%d %H:%M:%S')

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.DEBUG)
_console_handler.setFormatter(_fmt)
_logger.addHandler(_console_handler)

_F = TypeVar('_F', bound=Callable[..., Any])


def set_level(level: int | str) -> None:
    """Set log level.

    Args:
        level (str or int): ``DEBUG``=10, ``INFO``=20, ``WARN``=30, ``ERROR``=40, ``DISABLED``=60.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    _logger.setLevel(level)


def add_file_handler(filename: str, level: int = DEBUG) -> None:
    """Add a file handler.

    Args:
        filename (str): Path to the log file.
        level (int): Minimum log level written to the file. Defaults to DEBUG.
    """
    fh = logging.FileHandler(filename)
    fh.setLevel(level)
    fh.setFormatter(_fmt)
    _logger.addHandler(fh)


def debug(msg: str, *args: Any) -> None:
    """Show debug [D] message."""
    _logger.debug(msg, *args)


def info(msg: str, *args: Any) -> None:
    """Show information [I] message."""
    _logger.info(msg, *args)


def warn(msg: str, *args: Any) -> None:
    """Show warning [W] message."""
    _logger.warning(msg, *args)


def error(msg: str, *args: Any) -> None:
    """Show error [E] message."""
    _logger.error(msg, *args)


def counter(count: int, max_counts: int, divide: int = 1, message: str | None = None) -> None:
    """Show process counter as information.

    >>> '({count}/{max_counts}) events processed (message)'
    """
    if count == 0:
        return

    if (count % divide == 0) or (count == max_counts):
        if message is None:
            info(f'({count}/{max_counts}) events processed')
        else:
            info(f'({count}/{max_counts}) events processed ({message})')


def header1(message: str, level: Callable[[str], None] | None = None) -> None:
    """Show the following header.

    >>> '================================='
    >>> '============ message ============'
    >>> '================================='
    """
    if level is None:
        level = info
    if len(message) % 2 == 1:
        message += ' '
    len1 = 80
    len2 = (len1 - len(message) - 2) // 2
    level("=" * len1)
    if message == '':
        level(("=" * len2) + '==' + ("=" * len2))
    else:
        level(("=" * len2) + ' ' + message + ' ' + ("=" * len2))
    level("=" * len1)


def header2(message: str, level: Callable[[str], None] | None = None) -> None:
    """Show the following header.

    >>> '------------ message ------------'
    """
    if level is None:
        level = info
    if len(message) % 2 == 1:
        message += ' '
    len1 = 80
    len2 = (len1 - len(message) - 2) // 2
    if message == '':
        level(("-" * len2) + '--' + ("-" * len2))
    else:
        level(("-" * len2) + ' ' + message + ' ' + ("-" * len2))


def header3(message: str, level: Callable[[str], None] | None = None) -> None:
    """Show the following header.

    >>> ============ message ============
    """
    if level is None:
        level = info
    if len(message) % 2 == 1:
        message += ' '
    len1 = 80
    len2 = (len1 - len(message) - 2) // 2
    if message == '':
        level(("=" * len2) + '==' + ("=" * len2))
    else:
        level(("=" * len2) + ' ' + message + ' ' + ("=" * len2))


def table(
    names: list[str],
    data: list[list[str] | str],
    header: str | None = None,
    footer: str | None = None,
    max_length: int = 30,
) -> None:
    """Show table. All data must be str.

    >>> names = ['var0', 'var1']
    >>> data = [['hoge0', 'hoge1'], ['hoge2', 'hoge3']]
    >>> header = 'header message'
    >>> footer = 'footer message'
    >>>
    >>> '=============='
    >>> 'header message'
    >>> '=============='
    >>> 'var0   var1'
    >>> '--------------'
    >>> 'hoge0  hoge1'
    >>> 'hoge2  hoge3'
    >>> '--------------'
    >>> 'footer message'
    >>> '=============='
    """
    lengths = [5] * len(names)

    for index, name in enumerate(names):
        if len(name) > lengths[index]:
            lengths[index] = len(name)
        if len(name) > max_length:
            lengths[index] = max_length

    for idata in data:
        if idata == '-':
            continue
        for index, var in enumerate(idata):  # type: ignore[arg-type]
            if len(var) > lengths[index]:
                lengths[index] = len(var)

            if len(var) > max_length:
                lengths[index] = max_length

    total_length = sum(lengths) + len(names) * 2

    # header
    if header is not None:
        info('=' * total_length)
        info(header)

    # names
    info('=' * total_length)
    message = ''
    for index, name in enumerate(names):
        name = name.ljust(lengths[index])
        message += f'{name[:max_length]}  '
    info(message)
    info('-' * total_length)

    # data
    for idata in data:
        if idata == '-':
            info('-' * total_length)
            continue

        message = ''
        for index, var in enumerate(idata):  # type: ignore[arg-type]
            var = var.ljust(lengths[index])
            message += f'{var[:max_length]}  '
        info(message)
    info('=' * total_length)

    # footer
    if footer is not None:
        info(footer)
        info('=' * total_length)


def log_call(func: _F) -> _F:
    """Show the header and footer indicating start and end algorithm.

    Examples:
        >>> from storegate import logger
        >>>
        >>> @logger.log_call
        >>> def your_func(arg0, arg1):
        >>>     pass
    """
    @functools.wraps(func)
    def wrapper(obj: Any, *args: Any, **kwargs: Any) -> Any:
        header2(f'{obj.__class__.__name__} {func.__qualname__} START', debug)
        debug(f'args={args} kwargs={kwargs}')
        rtn = func(obj, *args, **kwargs)
        debug(f'return={rtn}')
        header2(f'{obj.__class__.__name__} {func.__qualname__} END', debug)
        return rtn

    return wrapper  # type: ignore[return-value]
