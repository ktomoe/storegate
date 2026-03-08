import concurrent.futures
import copy
import json
import multiprocessing as mp
import signal
from collections.abc import Sequence
from contextlib import contextmanager
from itertools import islice, product
from pathlib import Path
from typing import Any

from tqdm import tqdm  # type: ignore[import-untyped]

from storegate import logger
from storegate.agent import Agent


def _terminate_executor_processes(
    executor: concurrent.futures.ProcessPoolExecutor,
    kill_after: float = 1.0,
) -> None:
    """Terminate all worker processes currently owned by the executor."""
    processes = list(getattr(executor, '_processes', {}).values())

    for process in processes:
        if process.is_alive():
            process.terminate()

    for process in processes:
        process.join(timeout=kill_after)
        if process.is_alive():
            process.kill()
            process.join(timeout=kill_after)


class SearchAgent(Agent):
    """Search agent class of agent."""
    def __init__(self,
                 task: Any = None,
                 hps: dict[str, list[Any]] | None = None,
                 num_trials: int | None = None,
                 cuda_ids: list[int] | None = None,
                 disable_tqdm: bool = True,
                 json_dump: str | None = None,
                 job_timeout: float | None = None,
                 suffix_job_id: bool = True) -> None:
        """Initialize search agent.

        Args:
            task: Task instance to execute.
            hps (dict): Hyperparameter search space. Each key maps to a list of candidate values.
            num_trials (int or None): Number of repeated trials per hyperparameter set.
            cuda_ids (list[int] or None): List of CUDA device IDs to inject into
                parallel jobs. ``None`` runs all jobs serially in the current
                process using the task's own device.
                When a list is provided, the number of concurrent processes
                equals ``len(cuda_ids)``.
                For each submitted job, the agent injects one ``cuda_id`` value
                chosen from this list according to the job's submission index.
                This is not an exclusive worker-to-GPU binding.
            disable_tqdm (bool): If True, suppress the tqdm progress bar.
            json_dump (str or None): File path to dump the result history as JSON.
            job_timeout (float or None): Maximum seconds to wait for any pending job to
                complete.  If no job finishes within this window, all remaining pending
                jobs are cancelled and a ``TimeoutError`` is recorded in their result
                entries.  ``None`` (default) means wait indefinitely.
            suffix_job_id (bool): If True, inject ``output_var_names`` into each job's
                hyperparameters with ``_job{job_id}`` appended to every output variable
                name. Supports ``str``, ``list[str]``, and phase dictionaries.
        """
        if isinstance(cuda_ids, int):
            raise TypeError(
                f'cuda_ids must be a list of device IDs (e.g. [0, 1]), not an int. '
                f'Got: {cuda_ids!r}'
            )
        if cuda_ids is not None:
            if not isinstance(cuda_ids, list):
                raise TypeError(
                    f'cuda_ids must be a list of non-negative integers, got: {type(cuda_ids).__name__}.'
                )
            if len(cuda_ids) == 0:
                raise ValueError('cuda_ids must not be an empty list. Use None for single-worker execution.')
            for cuda_id in cuda_ids:
                if not isinstance(cuda_id, int) or isinstance(cuda_id, bool):
                    raise TypeError(
                        f'cuda_ids must contain only non-negative integers, got: {cuda_id!r}.'
                    )
                if cuda_id < 0:
                    raise ValueError(
                        f'cuda_ids must contain only non-negative integers, got: {cuda_id!r}.'
                    )

        self._task = task
        self._hps: list[dict[str, Any]] = self.all_combinations(hps)
        self._num_trials = num_trials
        self._cuda_ids = cuda_ids
        self._disable_tqdm = disable_tqdm
        self._json_dump: Path | None = self._validate_json_dump(json_dump)
        self._job_timeout = job_timeout
        self._suffix_job_id = suffix_job_id

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

    @staticmethod
    def _suffix_output_var_names(
        output_var_names: Any,
        job_id: int,
    ) -> Any:
        suffix = f'_job{job_id}'

        if output_var_names is None:
            return None
        if isinstance(output_var_names, str):
            return output_var_names + suffix
        if isinstance(output_var_names, list):
            return [var_name + suffix for var_name in output_var_names]
        if isinstance(output_var_names, dict):
            return {
                phase: SearchAgent._suffix_output_var_names(phase_var_names, job_id)
                for phase, phase_var_names in output_var_names.items()
            }
        raise TypeError(
            'output_var_names must be str, list[str], dict[str, ...], or None '
            f'when suffix_job_id=True, got {type(output_var_names).__name__}.'
        )

    def execute(self) -> None:
        """Run all hyperparameter jobs and collect results into ``_history``.

        Note:
            Each call to ``execute()`` **resets** the result history.
            Previous results are discarded.  Call ``finalize()`` before
            calling ``execute()`` again if you need to preserve them.
        """
        self._history = []
        args: list[list[Any]] = []

        for job_id, hps in enumerate(self._hps):
            if self._num_trials is None:
                args.append([self._task, hps, job_id, None])
            else:
                for trial_id in range(self._num_trials):
                    args.append([self._task, hps, job_id, trial_id])

        if self._cuda_ids is None:
            self.execute_serial_jobs(args)
            return
        self.execute_pool_jobs(args)


    def finalize(self) -> None:
        self._history.sort(key=lambda r: (r['job_id'], r.get('trial_id') or 0))
        if self._json_dump is not None:
            self._json_dump.write_text(
                json.dumps(self._history, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )

    @contextmanager
    def _serial_job_timeout(self, enabled: bool) -> Any:
        """Enforce ``job_timeout`` for serial execution when supported."""
        if (not enabled) or self._job_timeout is None or not hasattr(signal, 'SIGALRM'):
            yield
            return

        timeout_message = f'job did not complete within {self._job_timeout}s'

        def _raise_timeout(signum: int, frame: Any) -> None:
            raise TimeoutError(timeout_message)

        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, self._job_timeout)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer != (0.0, 0.0):
                signal.setitimer(signal.ITIMER_REAL, *previous_timer)

    def execute_serial_jobs(self, args: list[list[Any]]) -> None:
        """Execute jobs serially in the current process."""
        num_jobs = len(args)
        pbar_args = dict(ncols=80, total=num_jobs, disable=self._disable_tqdm)

        with tqdm(**pbar_args) as pbar:
            for task, hps, job_id, trial_id in args:
                task_for_job = copy.deepcopy(task)
                result = self.execute_task(
                    task_for_job,
                    hps,
                    job_id,
                    trial_id,
                    enforce_timeout=True,
                )
                self._history.append(result)
                pbar.update(1)
                if self._disable_tqdm:
                    logger.info(f'completed process ({len(self._history)}/{num_jobs})')


    def execute_pool_jobs(self, args: list[list[Any]]) -> None:
        """(expert method) Execute multiprocessing pool jobs.

        Uses a sliding-window submission strategy: at most ``num_workers``
        futures are in-flight at any time.  A new job is submitted only after
        a running job completes, so memory usage stays proportional to the
        worker count rather than the total number of jobs.

        If ``job_timeout`` was set at construction, each call to
        ``concurrent.futures.wait`` is bounded by that duration.  When the
        timeout expires with no completed jobs, all remaining pending jobs are
        cancelled and a ``TimeoutError`` entry is appended to ``_history`` for
        each cancelled job.
        """
        if self._cuda_ids is None:
            raise RuntimeError('execute_pool_jobs() requires cuda_ids to be set.')
        cuda_ids: Sequence[int] = self._cuda_ids
        num_jobs = len(args)
        num_workers = len(cuda_ids)

        pbar_args = dict(ncols=80, total=num_jobs, disable=self._disable_tqdm)

        # Maps each future back to its original job_arg so we can build a
        # meaningful error record if the future is cancelled due to timeout.
        future_to_arg: dict[concurrent.futures.Future[dict[str, Any]], list[Any]] = {}

        def _submit(
            executor: concurrent.futures.ProcessPoolExecutor,
            ii: int,
            job_arg: list[Any],
        ) -> concurrent.futures.Future[dict[str, Any]]:
            task, hps, job_id, trial_id = job_arg
            cuda_id = cuda_ids[ii % num_workers]
            if cuda_id is not None:
                hps = dict(hps)
                hps['cuda_id'] = cuda_id
            future = executor.submit(self.execute_task, task, hps, job_id, trial_id)
            future_to_arg[future] = [task, hps, job_id, trial_id]
            return future

        args_iter = iter(enumerate(args))

        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context(self._context),
            max_tasks_per_child=1,
        )

        try:
            with tqdm(**pbar_args) as pbar:
                # Fill the initial window (at most num_workers jobs in-flight).
                pending: set[concurrent.futures.Future[dict[str, Any]]] = {
                    _submit(executor, ii, job_arg)
                    for ii, job_arg in islice(args_iter, num_workers)
                }

                while pending:
                    done, pending = concurrent.futures.wait(
                        pending,
                        timeout=self._job_timeout,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )

                    if not done:
                        # Timeout: no job finished within job_timeout seconds.
                        # Cancel and record all futures currently in-flight.
                        for future in list(pending):
                            future.cancel()
                            _, hps, job_id, trial_id = future_to_arg.pop(future)
                            error_msg = (
                                f'TimeoutError: job did not complete within {self._job_timeout}s'
                            )
                            self._history.append({
                                'hps': hps,
                                'job_id': job_id,
                                'trial_id': trial_id,
                                'error': error_msg,
                            })
                            logger.error(f'Job {job_id} (trial {trial_id}) timed out after {self._job_timeout}s')
                            pbar.update(1)
                        # Also record timeout for jobs that were never submitted
                        # (still waiting in the sliding-window queue).
                        for _, job_arg in args_iter:
                            _, hps, job_id, trial_id = job_arg
                            error_msg = (
                                f'TimeoutError: job did not complete within {self._job_timeout}s'
                            )
                            self._history.append({
                                'hps': hps,
                                'job_id': job_id,
                                'trial_id': trial_id,
                                'error': error_msg,
                            })
                            logger.error(f'Job {job_id} (trial {trial_id}) timed out after {self._job_timeout}s')
                            pbar.update(1)
                        _terminate_executor_processes(executor)
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    for future in done:
                        completed_job_arg: list[Any] | None = future_to_arg.pop(future) if future in future_to_arg else None
                        try:
                            self._history.append(future.result())
                        except Exception as e:
                            if completed_job_arg is None:
                                hps = {}
                                job_id = -1
                                trial_id = None
                            else:
                                _, hps, job_id, trial_id = completed_job_arg
                            error_msg = f'{type(e).__name__}: {e}'
                            self._history.append({'hps': hps, 'job_id': job_id, 'trial_id': trial_id, 'error': error_msg})
                            logger.error(f'Job {job_id} (trial {trial_id}) raised in worker: {error_msg}')
                        pbar.update(1)
                        if self._disable_tqdm:
                            logger.info(f'completed process ({len(self._history)}/{num_jobs})')
                        # A slot is free — submit the next job immediately.
                        try:
                            ii, job_arg = next(args_iter)
                            pending.add(_submit(executor, ii, job_arg))
                        except StopIteration:
                            pass
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


    def execute_task(
        self,
        task: Any,
        hps: dict[str, Any],
        job_id: int,
        trial_id: int | None = None,
        enforce_timeout: bool = False,
    ) -> dict[str, Any]:
        """Execute pipeline."""
        result: dict[str, Any] = {
            'hps': hps,
            'job_id': job_id,
            'trial_id': trial_id,
        }

        try:
            task_hps = dict(hps)
            if self._suffix_job_id and hasattr(task, '_output_var_names'):
                base_output_var_names = task_hps.get(
                    'output_var_names',
                    getattr(task, '_output_var_names'),
                )
                task_hps['output_var_names'] = self._suffix_output_var_names(
                    base_output_var_names,
                    job_id,
                )
            with self._serial_job_timeout(enforce_timeout):
                task.set_hps(task_hps)
                result['result'] = task.execute()
        except Exception as e:
            result['error'] = f'{type(e).__name__}: {e}'
            logger.error(f'Job {job_id} (trial {trial_id}) failed: {result["error"]}')
        finally:
            try:
                task.finalize()
            except Exception as e:
                logger.error(f'Job {job_id} (trial {trial_id}) finalize failed: {type(e).__name__}: {e}')

        return result
