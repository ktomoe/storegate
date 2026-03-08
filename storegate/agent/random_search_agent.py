"""RandomSearchAgent module."""
import random
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

    def all_combinations(self, hps: dict[str, list[Any]] | None) -> list[dict[str, Any]]:
        """Randomly sample ``num_iter`` combinations from the search space."""
        if hps is None:
            return [{}]

        combinations = super().all_combinations(hps)
        rng = random.Random(self._seed)
        if self._replace:
            return [dict(rng.choice(combinations)) for _ in range(self._num_iter)]
        if self._num_iter > len(combinations):
            raise ValueError(
                'num_iter must be less than or equal to the total number of '
                f'unique combinations ({len(combinations)}) when replace=False.'
            )
        return [dict(combo) for combo in rng.sample(combinations, k=self._num_iter)]
