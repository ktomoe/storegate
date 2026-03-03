"""GridSearchAgent module."""

import multiprocessing as mp

from storegate import logger
from storegate.agent.search_agent import SearchAgent

class GridSearchAgent(SearchAgent):
    """Agent scanning all possible hyper parameters."""
    def __init__(self, **kwargs):
        """Initialize grid search agent."""
        super().__init__(**kwargs)
        self._context = 'spawn'

    def execute(self):
        ctx = mp.get_context(self._context)
        queue = ctx.Queue()
        args = []

        for job_id, hps in enumerate(self._hps):
            if self._num_trials is None:
                args.append([self._task, hps, job_id, None])
            else:
                for trial_id in range(self._num_trials):
                    args.append([self._task, hps, job_id, trial_id])

        self.execute_pool_jobs(ctx, queue, args)


