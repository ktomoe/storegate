from __future__ import annotations

import concurrent.futures
import copy
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any
from tqdm import tqdm
from itertools import product

from storegate import logger
from storegate.agent import Agent


class SearchAgent(Agent):
    """Search agent class of agent."""
    def __init__(self,
                 task: Any = None,
                 hps: dict[str, list[Any]] | None = None,
                 num_trials: int | None = None,
                 cuda_ids: list[int] | None = None,
                 disable_tqdm: bool = True,
                 json_dump: str | None = None) -> None:
        """Initialize search agent.

        Args:
            task: Task instance to execute.
            hps (dict): Hyperparameter search space. Each key maps to a list of candidate values.
            num_trials (int or None): Number of repeated trials per hyperparameter set.
            cuda_ids (list[int]): List of CUDA device IDs to use as parallel workers.
                Each element corresponds to one worker process, e.g. ``[0, 1, 2, 3]``.
                The number of concurrent processes equals ``len(cuda_ids)``.
            disable_tqdm (bool): If True, suppress the tqdm progress bar.
            json_dump (str or None): File path to dump the result history as JSON.
        """
        if isinstance(cuda_ids, int):
            raise TypeError(
                f'cuda_ids must be a list of device IDs (e.g. [0, 1]), not an int. '
                f'Got: {cuda_ids!r}'
            )

        self._task = task
        self._hps: list[dict[str, Any]] = self.all_combinations(hps)
        self._num_trials = num_trials
        self._cuda_ids = cuda_ids
        self._disable_tqdm = disable_tqdm
        self._json_dump: Path | None = self._validate_json_dump(json_dump)

        self._context = 'spawn'
        self._history: list[dict[str, Any]] = []


    @staticmethod
    def _validate_json_dump(json_dump: str | None) -> Path | None:
        """Validate and resolve the json_dump path.

        Raises:
            ValueError: If the path has a non-.json suffix or parent directory does not exist.
        """
        if json_dump is None:
            return None
        path = Path(json_dump).resolve()
        if path.suffix != '.json':
            raise ValueError(
                f'json_dump must end with ".json", got: {json_dump!r}'
            )
        if not path.parent.exists():
            raise ValueError(
                f'json_dump parent directory does not exist: {path.parent}'
            )
        return path

    def all_combinations(self, hps: dict[str, list[Any]] | None) -> list[dict[str, Any]]:
        if hps is None:
            return [{}]

        keys = hps.keys()
        values = hps.values()

        return [dict(zip(keys, value)) for value in product(*values)]

    def execute(self) -> None:
        self._history = []
        args: list[list[Any]] = []

        for job_id, hps in enumerate(self._hps):
            if self._num_trials is None:
                args.append([self._task, hps, job_id, None])
            else:
                for trial_id in range(self._num_trials):
                    args.append([self._task, hps, job_id, trial_id])

        self.execute_pool_jobs(args)


    def finalize(self) -> None:
        self._history.sort(key=lambda r: (r['job_id'], r.get('trial_id') or 0))
        if self._json_dump is not None:
            self._json_dump.write_text(
                json.dumps(self._history, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )


    def execute_pool_jobs(self, args: list[list[Any]]) -> None:
        """(expert method) Execute multiprocessing pool jobs.

        Uses ``concurrent.futures.ProcessPoolExecutor`` with ``as_completed``
        for event-driven result collection — no busy-wait, no TOCTOU.
        Results are appended to ``self._history`` as each job finishes.
        """
        jobs = copy.deepcopy(args)

        # cuda_ids=None means use the task's own device; run with a single worker.
        cuda_ids = self._cuda_ids if self._cuda_ids is not None else [None]
        num_jobs = len(jobs)
        num_workers = len(cuda_ids)

        pbar_args = dict(ncols=80, total=num_jobs, disable=self._disable_tqdm)

        futures: dict[concurrent.futures.Future[dict[str, Any]], None] = {}
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context(self._context),
        ) as executor, tqdm(**pbar_args) as pbar:
            for ii, job_arg in enumerate(jobs):
                task, hps, job_id, trial_id = job_arg
                cuda_id = cuda_ids[ii % num_workers]
                if cuda_id is not None:
                    hps = dict(hps)
                    hps['cuda_id'] = cuda_id
                futures[executor.submit(self.execute_task, task, hps, job_id, trial_id)] = None

            for future in concurrent.futures.as_completed(futures):
                self._history.append(future.result())
                pbar.update(1)
                if self._disable_tqdm:
                    logger.info(f'completed process ({len(self._history)}/{num_jobs})')


    def execute_task(
        self,
        task: Any,
        hps: dict[str, Any],
        job_id: int,
        trial_id: int | None = None,
    ) -> dict[str, Any]:
        """Execute pipeline."""
        result: dict[str, Any] = {
            'hps': hps,
            'job_id': job_id,
            'trial_id': trial_id,
        }

        try:
            task.set_hps(hps)
            result['result'] = task.execute()
            task.finalize()
        except Exception as e:
            result['error'] = f'{type(e).__name__}: {e}'
            logger.error(f'Job {job_id} (trial {trial_id}) failed: {result["error"]}')

        return result
