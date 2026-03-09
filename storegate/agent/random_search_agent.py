"""RandomSearchAgent module."""
import random
from math import prod
from typing import Any

from storegate.agent.search_agent import SearchAgent


class RandomSearchAgent(SearchAgent):
    """Agent randomly sampling hyperparameter combinations.

    Unlike GridSearchAgent which exhausts all combinations, RandomSearchAgent
    draws ``num_iter`` random samples from the search space, which is useful when
    the grid is too large to enumerate.

    Args:
        num_iter (int): Number of random hyperparameter combinations to sample.
        seed (int or None): Random seed for reproducibility.
        replace (bool): Whether to sample with replacement. Defaults to
            ``False``, which guarantees unique hyperparameter combinations.
        **kwargs: Forwarded to SearchAgent.

    Examples:
        >>> agent = RandomSearchAgent(
        ...     task=my_task,
        ...     hps={'lr': [1e-3, 1e-4, 1e-5], 'batch_size': [32, 64, 128]},
        ...     num_iter=10,
        ...     seed=42,
        ...     replace=False,
        ...     cuda_ids=[0, 1],
        ... )
        >>> agent.execute()
        >>> agent.finalize()
    """

    def __init__(
        self,
        num_iter: int,
        seed: int | None = None,
        replace: bool = False,
        **kwargs: Any,
    ) -> None:
        if not isinstance(num_iter, int) or isinstance(num_iter, bool):
            raise TypeError(
                f'num_iter must be a positive integer, got: {num_iter!r}.'
            )
        if num_iter <= 0:
            raise ValueError(
                f'num_iter must be a positive integer, got: {num_iter!r}.'
            )
        if not isinstance(replace, bool):
            raise TypeError(f'replace must be a bool, got: {replace!r}.')
        self._num_iter = num_iter
        self._seed = seed
        self._replace = replace
        super().__init__(**kwargs)

    @staticmethod
    def _combination_from_index(
        keys: list[str],
        values: list[list[Any]],
        index: int,
    ) -> dict[str, Any]:
        selected: list[Any] = [None] * len(values)
        for ii in range(len(values) - 1, -1, -1):
            index, offset = divmod(index, len(values[ii]))
            selected[ii] = values[ii][offset]
        return dict(zip(keys, selected))

    def all_combinations(self, hps: dict[str, list[Any]] | None) -> list[dict[str, Any]]:
        """Randomly sample ``num_iter`` combinations from the search space."""
        hps = self._validate_hps(hps)
        if hps is None:
            return [{}]

        keys = list(hps)
        values = [hps[key] for key in keys]
        total_combinations = prod(len(candidates) for candidates in values)
        rng = random.Random(self._seed)
        if self._replace:
            return [
                self._combination_from_index(
                    keys,
                    values,
                    rng.randrange(total_combinations),
                )
                for _ in range(self._num_iter)
            ]
        if self._num_iter > total_combinations:
            raise ValueError(
                'num_iter must be less than or equal to the total number of '
                f'unique combinations ({total_combinations}) when replace=False.'
            )
        return [
            self._combination_from_index(keys, values, index)
            for index in rng.sample(range(total_combinations), k=self._num_iter)
        ]
