"""Random search agent."""
from __future__ import annotations

import random
from typing import Any

from storegate import utilities as util
from storegate.agent.search_agent import SearchAgent


class RandomSearchAgent(SearchAgent):
    """Sample random hyperparameter combinations from the given search space.

    ``num_samples`` controls how many combinations are drawn, and ``seed``
    makes the sampling reproducible. Backend persistence semantics follow
    :class:`SearchAgent`; use ``ZarrDatabase`` when sampled job outputs must
    survive worker-process exit.
    """

    def __init__(
        self,
        *,
        num_samples: int = 1,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._num_samples = util.ensure_positive_int(num_samples, "num_samples")
        self._seed = seed

    def _generate_hp_list(self) -> list[dict[str, Any]]:
        if not self._hps:
            return [{}]

        rng = random.Random(self._seed)
        hp_items = list(self._hps.items())

        for key, values in hp_items:
            if not values:
                raise ValueError(
                    f"Hyperparameter space for '{key}' must not be empty."
                )

        return [
            {key: rng.choice(values) for key, values in hp_items}
            for _ in range(self._num_samples)
        ]
