"""Tests for storegate/__init__.py."""
import storegate


def test_version_is_string():
    assert isinstance(storegate.__version__, str)


def test_all_contains_expected_names():
    expected = {
        "Database",
        "NumpyDatabase",
        "ZarrDatabase",
        "HybridDatabase",
        "StoreGateReadView",
        "StoreGate",
        "__version__",
    }
    assert set(storegate.__all__) == expected


def test_public_classes_importable():
    assert storegate.Database is not None
    assert storegate.NumpyDatabase is not None
    assert storegate.ZarrDatabase is not None
    assert storegate.HybridDatabase is not None
    assert storegate.StoreGateReadView is not None
    assert storegate.StoreGate is not None
