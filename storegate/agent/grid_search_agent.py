"""Grid search agent."""
from __future__ import annotations

from itertools import product
from typing import Any

from storegate.agent.search_agent import SearchAgent


class GridSearchAgent(SearchAgent):
    """Evaluate every combination of the given hyperparameters."""

    def _generate_hp_list(self) -> list[dict[str, Any]]:
        if not self._hps:
            return [{}]

        for key, values in self._hps.items():
            if len(values) == 0:
                raise ValueError(
                    f"Hyperparameter space for '{key}' must not be empty."
                )

        keys = list(self._hps.keys())
        values = [self._hps[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in product(*values)]
