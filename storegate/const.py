"""Constant module.

Attributes:
    TRAIN (str): constant str to indicate *train* phase.
    VALID (str): constant str to indicate *valid* phase.
    TEST (str): constant str to indicate *test* phase.
    PHASES (tuple): constant tuple of TRAIN, VALID and TEST.
PBAR_FORMAT (str): format of tqdm progress bar.
    PBAR_ARGS (dict): args of tqdm progress bar.

Examples:
    >>> from storegate import const
    >>>
    >>> phase = 'train'
    >>> if phase == const.TRAIN:
    ...     pass
    >>> if phase in const.PHASES:
    ...     pass
"""
from __future__ import annotations

from typing import Any, Final

TRAIN: Final[str] = 'train'
VALID: Final[str] = 'valid'
TEST: Final[str] = 'test'
PHASES: Final[tuple[str, ...]] = (TRAIN, VALID, TEST)

PBAR_FORMAT: Final[str] = '{desc}: {percentage:3.0f}%| {n_fmt: >4}/{total_fmt: >4} [{rate_fmt: >16}{postfix}]'
PBAR_ARGS: Final[dict[str, Any]] = dict(unit=' batch', ncols=150, bar_format=PBAR_FORMAT)
