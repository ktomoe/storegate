"""GridSearchAgent module."""

from storegate.agent.search_agent import SearchAgent


class GridSearchAgent(SearchAgent):
    """Agent scanning all possible hyperparameter combinations.

    Exhaustively evaluates every combination in the search space (Cartesian
    product).  All behaviour is inherited from :class:`SearchAgent`; this
    class exists as a named entry point for discoverability and as an
    extension point for grid-specific overrides in the future.

    See :class:`RandomSearchAgent` for random sampling instead.
    """
