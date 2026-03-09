import copy
import json
import multiprocessing as mp
import signal
from dataclasses import dataclass
from collections.abc import Sequence
from contextlib import contextmanager
from itertools import product
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any, cast

from tqdm import tqdm  # type: ignore[import-untyped]

from storegate import logger
from storegate.agent import Agent


@dataclass
class _RunningJob:
    process: mp.Process
    result_pipe: Connection
    hps: dict[str, Any]
    job_id: int
    trial_id: int | None


def _shutdown_running_jobs(
    running_jobs: dict[int, _RunningJob],
    kill_after: float = 1.0,
) -> None:
    """Terminate any still-running worker processes and close their resources."""
    jobs = list(running_jobs.values())
    for job in jobs:
        job.result_pipe.close()
        if job.process.is_alive():
            job.process.terminate()

    for job in jobs:
        job.process.join(timeout=kill_after)
        if job.process.is_alive():
            job.process.kill()
            job.process.join(timeout=kill_after)
        job.process.close()


def _close_finished_job(job: _RunningJob) -> None:
    """Close resources for a worker that has already exited."""
    job.result_pipe.close()
    job.process.join()
    job.process.close()


def _execute_task_in_subprocess(
    task: Any,
    hps: dict[str, Any],
    job_id: int,
    trial_id: int | None,
    suffix_job_id: bool,
    result_pipe: Connection,
) -> None:
    task_hps = dict(hps)
    result: dict[str, Any] = {
        'hps': task_hps,
        'job_id': job_id,
        'trial_id': trial_id,
    }

    try:
        if suffix_job_id and hasattr(task, '_output_var_names'):
            base_output_var_names = task_hps.get(
                'output_var_names',
                getattr(task, '_output_var_names'),
            )
            task_hps['output_var_names'] = SearchAgent._suffix_output_var_names(
                base_output_var_names,
                job_id,
                trial_id,
            )
        task.set_hps(task_hps)
        result['result'] = task.execute()
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'
        logger.error(f'Job {job_id} (trial {trial_id}) failed: {result["error"]}')
    finally:
        try:
            task.finalize()
        except Exception as e:
            logger.error(
                f'Job {job_id} (trial {trial_id}) finalize failed: '
                f'{type(e).__name__}: {e}'
            )
        try:
            result_pipe.send(result)
        except (BrokenPipeError, EOFError, OSError):
            pass
        result_pipe.close()


class SearchAgent(Agent):
    """Search agent class of agent."""

    @staticmethod
    def _validate_cuda_id(cuda_id: object) -> None:
        if not isinstance(cuda_id, int) or isinstance(cuda_id, bool):
            raise TypeError(
                f'cuda_ids must contain only non-negative integers, got: {cuda_id!r}.'
            )
        if cuda_id < 0:
            raise ValueError(
                f'cuda_ids must contain only non-negative integers, got: {cuda_id!r}.'
            )

    @classmethod
    def _validate_cuda_ids(cls, cuda_ids: list[int] | None) -> list[int] | None:
        if isinstance(cuda_ids, int):
            raise TypeError(
                f'cuda_ids must be a list of device IDs (e.g. [0, 1]), not an int. '
                f'Got: {cuda_ids!r}'
            )
        if cuda_ids is None:
            return None
        if not isinstance(cuda_ids, list):
            raise TypeError(
                f'cuda_ids must be a list of non-negative integers, got: {type(cuda_ids).__name__}.'
            )
        if len(cuda_ids) == 0:
            raise ValueError(
                'cuda_ids must not be an empty list. Use None for single-worker execution.'
            )
        for cuda_id in cuda_ids:
            cls._validate_cuda_id(cuda_id)
        return cuda_ids

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
                hyperparameters with ``_job{job_id}_trial{trial_id}`` appended to
                every output variable name. When ``trial_id`` is ``None``, the
                implicit single trial is treated as ``trial0``. Supports ``str``,
                ``list[str]``, and phase dictionaries.
        """
        self._task = task
        self._hps: list[dict[str, Any]] = self.all_combinations(hps)
        self._num_trials = num_trials
        self._cuda_ids = self._validate_cuda_ids(cuda_ids)
        self._disable_tqdm = disable_tqdm
        self._json_dump: Path | None = self._validate_json_dump(json_dump)
        self._job_timeout = job_timeout
        self._suffix_job_id = suffix_job_id

        self._context = 'spawn'
        self._history: list[dict[str, Any]] = []


    @staticmethod
    def _validate_hps(hps: dict[str, list[Any]] | None) -> dict[str, list[Any]] | None:
        """Validate the hyperparameter search space.

        Each entry must map a hyperparameter name to a non-empty ``list`` of
        candidate values. This rejects common misconfigurations that would
        otherwise silently produce zero jobs or iterate over a string one
        character at a time.
        """
        if hps is None:
            return None
        if not isinstance(hps, dict):
            raise TypeError(
                f'hps must be a dict[str, list[Any]] or None, got: {type(hps).__name__}.'
            )

        validated: dict[str, list[Any]] = {}
        for key, candidates in hps.items():
            if not isinstance(candidates, list):
                raise TypeError(
                    f"hps['{key}'] must be a non-empty list of candidate values, "
                    f'got: {type(candidates).__name__}.'
                )
            if len(candidates) == 0:
                raise ValueError(
                    f"hps['{key}'] must be a non-empty list of candidate values."
                )
            validated[key] = candidates

        return validated

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
        hps = self._validate_hps(hps)
        if hps is None:
            return [{}]

        keys = hps.keys()
        values = hps.values()
        combinations = [dict(zip(keys, value)) for value in product(*values)]
        if not combinations:
            raise ValueError('hps must generate at least one job.')
        return combinations

    @staticmethod
    def _suffix_output_var_names(
        output_var_names: Any,
        job_id: int,
        trial_id: int | None = None,
    ) -> Any:
        suffix = f'_job{job_id}_trial{0 if trial_id is None else trial_id}'

        if output_var_names is None:
            return None
        if isinstance(output_var_names, str):
            return output_var_names + suffix
        if isinstance(output_var_names, list):
            return [var_name + suffix for var_name in output_var_names]
        if isinstance(output_var_names, dict):
            return {
                phase: SearchAgent._suffix_output_var_names(
                    phase_var_names,
                    job_id,
                    trial_id,
                )
                for phase, phase_var_names in output_var_names.items()
            }
        raise TypeError(
            'output_var_names must be str, list[str], dict[str, ...], or None '
            f'when suffix_job_id=True, got {type(output_var_names).__name__}.'
        )

    def _build_job_args(self) -> list[list[Any]]:
        trial_ids: list[int | None] = (
            [None] if self._num_trials is None else list(range(self._num_trials))
        )
        return [
            [self._task, hps, job_id, trial_id]
            for job_id, hps in enumerate(self._hps)
            for trial_id in trial_ids
        ]

    def execute(self) -> None:
        """Run all hyperparameter jobs and collect results into ``_history``.

        Note:
            Each call to ``execute()`` **resets** the result history.
            Previous results are discarded.  Call ``finalize()`` before
            calling ``execute()`` again if you need to preserve them.
        """
        self._history = []
        args = self._build_job_args()

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
        child processes are in-flight at any time. A new job is submitted only
        after a running job exits, so memory usage stays proportional to the
        worker count rather than the total number of jobs.

        If ``job_timeout`` was set at construction, the parent waits at most
        that many seconds for any worker to exit. When the timeout expires with
        no completed jobs, all running and queued jobs are recorded as timed
        out, and active workers are terminated in the unified shutdown path.
        """
        if self._cuda_ids is None:
            raise RuntimeError('execute_pool_jobs() requires cuda_ids to be set.')
        cuda_ids: Sequence[int] = self._cuda_ids
        num_jobs = len(args)
        num_workers = len(cuda_ids)
        context = cast(Any, mp.get_context(self._context))

        pbar_args = dict(ncols=80, total=num_jobs, disable=self._disable_tqdm)

        def _submit(
            ii: int,
            job_arg: list[Any],
            running_jobs: dict[int, _RunningJob],
        ) -> None:
            task, hps, job_id, trial_id = job_arg
            task_hps = dict(hps)
            cuda_id = cuda_ids[ii % num_workers]
            task_hps['cuda_id'] = cuda_id

            parent_pipe, child_pipe = context.Pipe(duplex=False)
            process = context.Process(
                target=_execute_task_in_subprocess,
                args=(
                    task,
                    task_hps,
                    job_id,
                    trial_id,
                    self._suffix_job_id,
                    child_pipe,
                ),
            )
            process.start()
            child_pipe.close()
            running_jobs[process.sentinel] = _RunningJob(
                process=process,
                result_pipe=parent_pipe,
                hps=task_hps,
                job_id=job_id,
                trial_id=trial_id,
            )

        args_iter = iter(enumerate(args))
        running_jobs: dict[int, _RunningJob] = {}

        try:
            with tqdm(**pbar_args) as pbar:
                while len(running_jobs) < num_workers:
                    try:
                        ii, job_arg = next(args_iter)
                    except StopIteration:
                        break
                    _submit(ii, job_arg, running_jobs)

                while running_jobs:
                    ready_sentinels = wait(
                        list(running_jobs),
                        timeout=self._job_timeout,
                    )

                    if not ready_sentinels:
                        error_msg = (
                            f'TimeoutError: job did not complete within {self._job_timeout}s'
                        )
                        for job in running_jobs.values():
                            self._history.append({
                                'hps': job.hps,
                                'job_id': job.job_id,
                                'trial_id': job.trial_id,
                                'error': error_msg,
                            })
                            logger.error(
                                f'Job {job.job_id} (trial {job.trial_id}) timed out after '
                                f'{self._job_timeout}s'
                            )
                            pbar.update(1)
                        for _, job_arg in args_iter:
                            _, hps, job_id, trial_id = job_arg
                            self._history.append({
                                'hps': hps,
                                'job_id': job_id,
                                'trial_id': trial_id,
                                'error': error_msg,
                            })
                            logger.error(
                                f'Job {job_id} (trial {trial_id}) timed out after '
                                f'{self._job_timeout}s'
                            )
                            pbar.update(1)
                        break

                    for sentinel in ready_sentinels:
                        job = running_jobs.pop(cast(int, sentinel))
                        if job.result_pipe.poll():
                            try:
                                self._history.append(job.result_pipe.recv())
                            except EOFError:
                                error_msg = (
                                    'ChildProcessError: worker exited before returning '
                                    f'a result (exit code {job.process.exitcode}).'
                                )
                                self._history.append({
                                    'hps': job.hps,
                                    'job_id': job.job_id,
                                    'trial_id': job.trial_id,
                                    'error': error_msg,
                                })
                                logger.error(
                                    f'Job {job.job_id} (trial {job.trial_id}) raised in '
                                    f'worker: {error_msg}'
                                )
                        else:
                            error_msg = (
                                'ChildProcessError: worker exited before returning '
                                f'a result (exit code {job.process.exitcode}).'
                            )
                            self._history.append({
                                'hps': job.hps,
                                'job_id': job.job_id,
                                'trial_id': job.trial_id,
                                'error': error_msg,
                            })
                            logger.error(
                                f'Job {job.job_id} (trial {job.trial_id}) raised in '
                                f'worker: {error_msg}'
                            )
                        _close_finished_job(job)
                        pbar.update(1)
                        if self._disable_tqdm:
                            logger.info(f'completed process ({len(self._history)}/{num_jobs})')

                    while len(running_jobs) < num_workers:
                        try:
                            ii, job_arg = next(args_iter)
                        except StopIteration:
                            break
                        _submit(ii, job_arg, running_jobs)
        finally:
            _shutdown_running_jobs(running_jobs)


    def execute_task(
        self,
        task: Any,
        hps: dict[str, Any],
        job_id: int,
        trial_id: int | None = None,
        enforce_timeout: bool = False,
    ) -> dict[str, Any]:
        """Execute pipeline."""
        task_hps = dict(hps)
        result: dict[str, Any] = {
            'hps': task_hps,
            'job_id': job_id,
            'trial_id': trial_id,
        }

        try:
            if self._suffix_job_id and hasattr(task, '_output_var_names'):
                base_output_var_names = task_hps.get(
                    'output_var_names',
                    getattr(task, '_output_var_names'),
                )
                task_hps['output_var_names'] = self._suffix_output_var_names(
                    base_output_var_names,
                    job_id,
                    trial_id,
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
