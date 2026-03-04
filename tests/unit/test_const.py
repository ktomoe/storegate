"""Unit tests for const module."""
from storegate import const


def test_phase_values():
    assert const.TRAIN == 'train'
    assert const.VALID == 'valid'
    assert const.TEST == 'test'


def test_phases_tuple():
    assert const.PHASES == ('train', 'valid', 'test')
    assert len(const.PHASES) == 3


def test_phases_contains_all_phase_constants():
    assert const.TRAIN in const.PHASES
    assert const.VALID in const.PHASES
    assert const.TEST in const.PHASES
