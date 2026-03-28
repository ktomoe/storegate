"""Tests for storegate/agent/__init__.py."""


def test_agent_importable():
    from storegate.agent import Agent
    assert Agent is not None


def test_search_agent_importable():
    from storegate.agent import SearchAgent
    assert SearchAgent is not None


def test_grid_search_agent_importable():
    from storegate.agent import GridSearchAgent
    assert GridSearchAgent is not None


def test_random_search_agent_importable():
    from storegate.agent import RandomSearchAgent
    assert RandomSearchAgent is not None


def test_all_exports():
    import storegate.agent as agent_mod
    assert set(agent_mod.__all__) == {
        "Agent",
        "SearchAgent",
        "GridSearchAgent",
        "RandomSearchAgent",
    }
