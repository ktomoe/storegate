import copy
import time
import json
import multiprocessing as mp
from tqdm import tqdm
from itertools import product

from storegate import logger
from storegate.agent import Agent

class SearchAgent(Agent):
    """Search agent class of agent."""
    def __init__(self, task=None,
                       hps=None,
                       num_trials=None,
                       cuda_ids=None,
                       disable_tqdm=True,
                       json_dump=None):
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
        self._hps = self.all_combinations(hps)
        self._num_trials = num_trials
        self._cuda_ids = cuda_ids
        self._disable_tqdm = disable_tqdm
        self._json_dump = json_dump

        self._context = 'spawn'
        self._history = []


    def all_combinations(self, hps):
        if hps is None:
            return [{}]

        keys = hps.keys()
        values = hps.values()

        return [dict(zip(keys, value)) for value in product(*values)]

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


    def finalize(self):
        self._history.sort(key=lambda r: (r['job_id'], r.get('trial_id') or 0))
        if self._json_dump:
            with open(self._json_dump, 'w', encoding="utf-8") as f:
                json.dump(self._history, f, ensure_ascii=False, indent=2)


    def execute_pool_jobs(self, ctx, queue, args):
        """(expert method) Execute multiprocessing pool jobs."""
        jobs = copy.deepcopy(args)

        pool = [0] * len(self._cuda_ids)
        num_jobs = len(jobs)
        all_done = False

        pbar_args = dict(ncols=80, total=num_jobs, disable=self._disable_tqdm)
        with tqdm(**pbar_args) as pbar:
            while not all_done:
                time.sleep(0.05)

                if len(jobs) == 0:
                    done = True
                    for ii, process in enumerate(pool):
                        if (process != 0) and (process.is_alive()):
                            done = False
                    all_done = done

                else:
                    for ii, process in enumerate(pool):
                        if len(jobs) == 0:
                            continue

                        if (process == 0) or (not process.is_alive()):
                            time.sleep(0.05)
                            job_arg = jobs.pop(0)
                            pool[ii] = ctx.Process(target=self.execute_wrapper,
                                                   args=(queue, *job_arg, self._cuda_ids[ii]),
                                                   daemon=False)
                            pool[ii].start()
                            pbar.update(1)

                            if self._disable_tqdm:
                                logger.info(f'launch process ({num_jobs - len(jobs)}/{num_jobs})')

                while not queue.empty():
                    self._history.append(queue.get())

        while not queue.empty():
            self._history.append(queue.get())


    def execute_wrapper(self, queue, task, hps, job_id, trial_id, cuda_id):
        """(expert method) Wrapper method to execute multiprocessing pipeline."""
        hps['cuda_id'] = cuda_id
        result = self.execute_task(task, hps, job_id, trial_id)
        queue.put(result)


    def execute_task(self, task, hps, job_id, trial_id=None):
        """Execute pipeline."""
        result = {
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
