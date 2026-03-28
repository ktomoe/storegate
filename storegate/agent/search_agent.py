"""Hyperparameter search agents."""
from __future__ import annotations

import copy
import json
import multiprocessing as mp
from multiprocessing.process import BaseProcess
import queue
import threading
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from storegate import utilities as util
from storegate.agent.agent import Agent
from storegate.formatters import ProgressFormatter

if TYPE_CHECKING:
    from storegate.storegate import StoreGate


# -- worker -------------------------------------------------------------------

def _worker_loop(
    cuda_id: int,
    slot_label: str,
    job: dict[str, Any],
    result_queue: mp.Queue,
    progress_queue: mp.Queue | None,
    task_cls: type,
    task_args: dict[str, Any],
    storegate: StoreGate | None,
) -> None:
    """Run exactly one job in a fresh child process."""
    try:
        job_id: str = job["job_id"]
        hp_set: dict[str, Any] = job["hps"]

        if progress_queue is not None:
            progress_queue.put({
                "type": "job_start",
                "job_id": job_id,
                "slot_label": slot_label,
                "hp_idx": job["hp_idx"],
                "trial": job["trial"],
                "hps": hp_set,
                "cuda_id": cuda_id,
            })

        try:
            args = _apply_job_id(task_args, job_id)
            task = task_cls(**args)
            task.set_cuda_id(cuda_id)

            if storegate is not None:
                task.storegate = storegate

            if progress_queue is not None:
                def _cb(info: dict[str, Any], _jid: str = job_id) -> None:
                    progress_queue.put({
                        "type": "batch",
                        "job_id": _jid,
                        "slot_label": slot_label,
                        "info": info,
                    })
                task._progress_callback = _cb

            task.set_hps(hp_set)
            result = task.execute()

            result_queue.put({
                "job_id": job_id,
                "hp_idx": job["hp_idx"],
                "trial": job["trial"],
                "hps": hp_set,
                "cuda_id": cuda_id,
                "status": "success",
                "result": result,
            })
            if progress_queue is not None:
                progress_queue.put({
                    "type": "job_end",
                    "job_id": job_id,
                    "slot_label": slot_label,
                    "status": "success",
                })
        except Exception as exc:
            result_queue.put({
                "job_id": job_id,
                "hp_idx": job["hp_idx"],
                "trial": job["trial"],
                "hps": hp_set,
                "cuda_id": cuda_id,
                "status": "error",
                "error": str(exc),
            })
            if progress_queue is not None:
                progress_queue.put({
                    "type": "job_end",
                    "job_id": job_id,
                    "slot_label": slot_label,
                    "status": "error",
                })
    finally:
        if storegate is not None:
            try:
                storegate.close()
            except Exception:
                pass


def _apply_job_id(
    task_args: dict[str, Any], job_id: str,
) -> dict[str, Any]:
    """Return a deep copy of *task_args* with *job_id* appended to
    every output variable name so that concurrent predict writes do
    not collide."""
    args = copy.deepcopy(task_args)
    var_names = args.get("var_names")
    if var_names is not None and "outputs" in var_names:
        outputs = var_names["outputs"]
        if isinstance(outputs, str):
            var_names["outputs"] = f"{outputs}_{job_id}"
        elif isinstance(outputs, list):
            var_names["outputs"] = [f"{n}_{job_id}" for n in outputs]
    return args


class SearchAgent(Agent):
    """Base class for hyperparameter search agents.

    Subclasses implement :meth:`_generate_hp_list` to define which
    hyperparameter combinations to evaluate.  Each combination is
    executed *num_trials* times.  When ``cuda_ids`` is provided,
    jobs run in child processes via ``multiprocessing`` with at most
    ``len(cuda_ids)`` jobs active at once. Each job gets its own
    fresh child process; completed jobs free a slot for the next job.

    Persisted outputs that must remain visible after the
    worker exits should therefore be written to a durable backend such as
    ``ZarrDatabase``. ``NumpyDatabase`` is in-memory only and is suitable for
    ephemeral per-process data, not for persisted cross-process search results.
    """

    def __init__(
        self,
        storegate: StoreGate | None = None,
        task: type | None = None,
        task_args: dict[str, Any] | None = None,
        hps: dict[str, list[Any]] | None = None,
        num_trials: int = 1,
        cuda_ids: list[int] | None = None,
        progress: bool = True,
    ) -> None:
        self._storegate = storegate
        self._task = task
        self._task_args = task_args or {}
        self._hps = hps or {}
        self._num_trials = util.ensure_positive_int(num_trials, "num_trials")
        self._cuda_ids = cuda_ids
        self._progress = progress
        self._formatter = ProgressFormatter()

    # -- storegate property ---------------------------------------------------
    @property
    def storegate(self) -> StoreGate | None:
        """Return the storegate shared with every spawned task."""
        return self._storegate

    @storegate.setter
    def storegate(self, sg: StoreGate | None) -> None:
        self._storegate = sg

    # -- abstract -------------------------------------------------------------
    @abstractmethod
    def _generate_hp_list(self) -> list[dict[str, Any]]:
        """Return the list of hyperparameter dicts to evaluate."""

    # -- execute --------------------------------------------------------------
    def execute(self) -> list[dict[str, Any]]:
        """Run all HP combinations x trials and return a result list.

        Each element is a dict with keys ``job_id``, ``hp_idx``,
        ``trial``, ``hps``, ``cuda_id``, ``status``, and either
        ``result`` (on success) or ``error`` (on failure).
        """
        hp_list = self._generate_hp_list()

        jobs: list[dict[str, Any]] = []
        for hp_idx, hp_set in enumerate(hp_list):
            for trial in range(self._num_trials):
                jobs.append({
                    "job_id": f"hp{hp_idx}_trial{trial}",
                    "hp_idx": hp_idx,
                    "trial": trial,
                    "hps": hp_set,
                })

        if not jobs:
            return []

        total = len(jobs)
        fmt = self._formatter

        if self._progress:
            fmt.print_header(
                total,
                len(hp_list),
                self._num_trials,
                slot_labels=self._get_slot_labels(),
            )

        if self._cuda_ids:
            results = self._run_parallel(jobs, total)
        else:
            results = self._run_sequential(jobs, total)

        if self._progress:
            n_ok = sum(1 for r in results if r["status"] == "success")
            fmt.print_footer(n_ok, total - n_ok)

        results.sort(key=lambda r: (r["hp_idx"], r["trial"]))
        return results

    # -- sequential execution (CPU / cuda_ids=None) ---------------------------
    def _run_sequential(
        self, jobs: list[dict[str, Any]], total: int,
    ) -> list[dict[str, Any]]:
        fmt = self._formatter
        task_cls = self._require_task_cls()
        results: list[dict[str, Any]] = []
        slot_label = self._get_sequential_slot_label()

        for job_num, job in enumerate(jobs, 1):
            job_id = job["job_id"]
            hp_set = job["hps"]
            task: Any = None

            if self._progress:
                fmt.print_job_start(slot_label, job_id, hp_set, job_num, total)

            try:
                args = _apply_job_id(self._task_args, job_id)
                task = task_cls(**args)

                if self._storegate is not None:
                    task.storegate = self._storegate

                if self._progress:
                    def _cb(info: dict[str, Any], _jid: str = job_id) -> None:
                        fmt.print_batch(slot_label, _jid, info)
                    task._progress_callback = _cb

                task.set_hps(hp_set)
                result = task.execute()

                status = "success"
                results.append({
                    "job_id": job_id,
                    "hp_idx": job["hp_idx"],
                    "trial": job["trial"],
                    "hps": hp_set,
                    "cuda_id": None,
                    "status": status,
                    "result": result,
                })
            except Exception as exc:
                status = "error"
                results.append({
                    "job_id": job_id,
                    "hp_idx": job["hp_idx"],
                    "trial": job["trial"],
                    "hps": hp_set,
                    "cuda_id": None,
                    "status": status,
                    "error": str(exc),
                })
            finally:
                if task is not None:
                    reset = getattr(task, "reset", None)
                    if callable(reset):
                        reset()

            if self._progress:
                fmt.print_job_end(slot_label, job_id, status, job_num, total)

        return results

    # -- parallel execution (GPU) ---------------------------------------------
    def _run_parallel(
        self, jobs: list[dict[str, Any]], total: int,
    ) -> list[dict[str, Any]]:
        fmt = self._formatter
        ctx = mp.get_context("spawn")
        cuda_ids = self._cuda_ids
        assert cuda_ids is not None
        task_cls = self._require_task_cls()
        num_workers = len(cuda_ids)
        slot_labels = self._get_parallel_slot_labels()

        result_queue = ctx.Queue()
        progress_queue: mp.Queue | None = ctx.Queue() if self._progress else None

        # Background thread to drain progress messages
        stop_event = threading.Event()
        job_counter: dict[str, int] = {}

        def _drain_progress() -> None:
            assert progress_queue is not None
            seq = 0
            while not stop_event.is_set():
                try:
                    msg = progress_queue.get(timeout=0.5)
                except Exception:
                    continue
                msg_type = msg.get("type")
                if msg_type == "job_start":
                    seq += 1
                    job_counter[msg["job_id"]] = seq
                    fmt.print_job_start(
                        msg["slot_label"],
                        msg["job_id"],
                        msg["hps"],
                        seq,
                        total,
                    )
                elif msg_type == "batch":
                    fmt.print_batch(
                        msg["slot_label"],
                        msg["job_id"],
                        msg["info"],
                    )
                elif msg_type == "job_end":
                    jid = msg["job_id"]
                    fmt.print_job_end(
                        msg["slot_label"],
                        jid,
                        msg["status"],
                        job_counter.get(jid, 0),
                        total,
                    )

        if self._progress:
            drain_thread = threading.Thread(target=_drain_progress, daemon=True)
            drain_thread.start()

        results: list[dict[str, Any]] = []
        next_job_idx = 0
        active_jobs: dict[str, tuple[BaseProcess, int]] = {}
        active_slots: dict[int, str] = {}

        def _launch_job(slot_idx: int, job: dict[str, Any]) -> None:
            process = ctx.Process(
                target=_worker_loop,
                args=(
                    cuda_ids[slot_idx],
                    slot_labels[slot_idx],
                    job,
                    result_queue,
                    progress_queue,
                    task_cls,
                    self._task_args,
                    self._storegate,
                ),
            )
            process.start()
            active_jobs[job["job_id"]] = (process, slot_idx)
            active_slots[slot_idx] = job["job_id"]

        def _fill_open_slots() -> None:
            nonlocal next_job_idx
            for slot_idx in range(num_workers):
                if slot_idx in active_slots or next_job_idx >= len(jobs):
                    continue
                _launch_job(slot_idx, jobs[next_job_idx])
                next_job_idx += 1

        _fill_open_slots()

        try:
            while len(results) < len(jobs):
                try:
                    result = result_queue.get(timeout=0.5)
                except queue.Empty:
                    crashed = [
                        (job_id, process)
                        for job_id, (process, _slot_idx) in active_jobs.items()
                        if process.exitcode not in (None, 0)
                    ]
                    if crashed:
                        self._terminate_alive_workers(
                            [process for process, _slot_idx in active_jobs.values()]
                        )
                        raise RuntimeError(
                            "SearchAgent worker crashed before returning a result: "
                            + ", ".join(
                                f"job_id={job_id} pid={process.pid} exitcode={process.exitcode}"
                                for job_id, process in crashed
                            )
                        )

                    if (
                        next_job_idx >= len(jobs)
                        and active_jobs
                        and all(
                            process.exitcode is not None
                            for process, _slot_idx in active_jobs.values()
                        )
                    ):
                        raise RuntimeError(
                            "SearchAgent workers exited before producing all results: "
                            f"expected {len(jobs)}, got {len(results)}."
                        )

                    continue

                results.append(result)
                job_id = result["job_id"]
                process_info = active_jobs.pop(job_id, None)
                if process_info is None:
                    continue

                process, slot_idx = process_info
                process.join()
                active_slots.pop(slot_idx, None)
                _fill_open_slots()

            return results
        finally:
            self._terminate_alive_workers(
                [process for process, _slot_idx in active_jobs.values()]
            )
            for process, _slot_idx in active_jobs.values():
                process.join()

            if self._progress:
                stop_event.set()
                drain_thread.join(timeout=2.0)

    def _get_slot_labels(self) -> list[str]:
        if self._cuda_ids:
            return self._get_parallel_slot_labels()
        return [self._get_sequential_slot_label()]

    def _get_sequential_slot_label(self) -> str:
        return "cpu"

    def _get_parallel_slot_labels(self) -> list[str]:
        cuda_ids = self._cuda_ids
        assert cuda_ids is not None
        counts: dict[int, int] = {}
        labels: list[str] = []

        for cuda_id in cuda_ids:
            counts[cuda_id] = counts.get(cuda_id, 0) + 1
            ordinal = counts[cuda_id]
            if ordinal == 1:
                labels.append(f"cuda:{cuda_id}")
            else:
                labels.append(f"cuda:{cuda_id}({ordinal})")

        return labels

    def _terminate_alive_workers(self, workers: list[BaseProcess]) -> None:
        for worker in workers:
            if worker.exitcode is None:
                worker.terminate()

    def _require_task_cls(self) -> type:
        task_cls = self._task
        if task_cls is None:
            raise ValueError("task is required.")
        return task_cls

    # -- I/O ------------------------------------------------------------------
    def save_results(
        self,
        results: list[dict[str, Any]],
        output_path: str | Path,
    ) -> None:
        """Serialize *results* to a JSON file."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    """Fallback encoder for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )
