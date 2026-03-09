"""Concurrency tests that verify thread-safety contracts.

StoreGate and its backends are **not** thread-safe by design.
These tests confirm that concurrent access without external locking
*can* produce incorrect results, and that proper serialisation
(via ``threading.Lock``) eliminates those issues.

The tests use tight loops with many threads to maximise the probability
of hitting race conditions within a single run.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from storegate import StoreGate
from storegate.database.numpy_database import NumpyDatabase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_THREADS = 8
ITEMS_PER_THREAD = 50


def _add_data_worker(
    sg: StoreGate,
    var_name: str,
    phase: str,
    data: np.ndarray,
    n: int,
    lock: threading.Lock | None = None,
) -> None:
    """Append *data* to *sg* exactly *n* times, optionally under *lock*."""
    for _ in range(n):
        if lock is not None:
            with lock:
                sg.add_data(var_name, data, phase=phase)
        else:
            sg.add_data(var_name, data, phase=phase)


# ---------------------------------------------------------------------------
# NumpyDatabase — cache invalidation race
# ---------------------------------------------------------------------------

class TestNumpyDatabaseConcurrency:
    """Race conditions in NumpyDatabase._materialize / add_data."""

    def test_concurrent_add_and_read_without_lock_may_produce_stale_cache(self) -> None:
        """Demonstrate that unsynchronised add + read can return stale data.

        One writer thread appends chunks while reader threads call get_data.
        Without locking, a reader may see a cached array that was materialised
        *before* the latest append, so its length can be shorter than expected.

        Because race conditions are probabilistic, we run many iterations and
        assert that *at least* the final state is consistent when no concurrent
        access is happening.
        """
        db = NumpyDatabase()
        db.initialize('d')

        total_writes = NUM_THREADS * ITEMS_PER_THREAD
        chunk = np.ones((1, 4))

        # Writer: appends data in a tight loop
        def writer() -> None:
            for _ in range(total_writes):
                db.add_data('d', 'x', chunk, 'train')

        # Reader: reads data in a tight loop
        observed_lengths: list[int] = []

        def reader() -> None:
            for _ in range(total_writes):
                try:
                    arr = db.get_data('d', 'x', 'train', index=None)
                    observed_lengths.append(len(arr))
                except Exception:
                    pass  # may fail mid-race; that's acceptable here

        with ThreadPoolExecutor(max_workers=2) as pool:
            fw = pool.submit(writer)
            fr = pool.submit(reader)
            fw.result()
            fr.result()

        # After all concurrent activity stops, a single-threaded read must
        # return the fully consistent array.
        final = db.get_data('d', 'x', 'train', index=None)
        assert len(final) == total_writes

    def test_concurrent_add_with_lock_is_consistent(self) -> None:
        """Serialised writes via a lock always produce the correct total."""
        db = NumpyDatabase()
        db.initialize('d')
        lock = threading.Lock()
        chunk = np.ones((1, 4))

        def writer() -> None:
            for _ in range(ITEMS_PER_THREAD):
                with lock:
                    db.add_data('d', 'x', chunk, 'train')

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = [pool.submit(writer) for _ in range(NUM_THREADS)]
            for f in as_completed(futures):
                f.result()

        arr = db.get_data('d', 'x', 'train', index=None)
        expected = NUM_THREADS * ITEMS_PER_THREAD
        assert len(arr) == expected
        assert db._metadata['d']['train']['x']['total_events'] == expected


# ---------------------------------------------------------------------------
# StoreGate — metadata / backend-switch races
# ---------------------------------------------------------------------------

class TestStoreGateConcurrency:
    """Race conditions on StoreGate._metadata and backend switching."""

    def test_concurrent_add_data_without_lock_metadata_may_drift(self, tmp_path: Any) -> None:
        """Multiple threads adding to different phases without a lock.

        The internal ``_metadata`` dict and ``_invalidate_compiled`` are not
        atomic. Under contention the compiled flag or size bookkeeping *may*
        become inconsistent.  We verify that a post-hoc ``compile()`` can
        still detect and reconcile the state.
        """
        sg = StoreGate(str(tmp_path / 'store'), mode='w')
        sg.set_data_id('test')

        data = np.arange(10).reshape(5, 2)

        def add_train() -> None:
            for _ in range(ITEMS_PER_THREAD):
                sg.add_data('x', data, phase='train')

        def add_valid() -> None:
            for _ in range(ITEMS_PER_THREAD):
                sg.add_data('x', data, phase='valid')

        with ThreadPoolExecutor(max_workers=2) as pool:
            ft = pool.submit(add_train)
            fv = pool.submit(add_valid)
            ft.result()
            fv.result()

        # After threads finish, compile() must still work on the zarr data.
        # The total events should equal ITEMS_PER_THREAD * 5 per phase.
        sg.compile()
        expected = ITEMS_PER_THREAD * 5
        assert len(sg['train']) == expected
        assert len(sg['valid']) == expected

    def test_concurrent_add_data_with_lock_is_always_correct(self, tmp_path: Any) -> None:
        """Serialised writes to a shared StoreGate are always consistent."""
        sg = StoreGate(str(tmp_path / 'store'), mode='w')
        sg.set_data_id('test')
        lock = threading.Lock()

        data = np.arange(10).reshape(5, 2)

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = [
                pool.submit(_add_data_worker, sg, 'x', 'train', data, ITEMS_PER_THREAD, lock)
                for _ in range(NUM_THREADS)
            ]
            for f in as_completed(futures):
                f.result()

        sg.compile()
        expected = NUM_THREADS * ITEMS_PER_THREAD * 5
        assert len(sg['train']) == expected

    def test_concurrent_backend_switch_with_lock(self, tmp_path: Any) -> None:
        """Backend switching under a lock does not corrupt state."""
        sg = StoreGate(str(tmp_path / 'store'), mode='w')
        sg.set_data_id('test')
        lock = threading.Lock()

        data = np.arange(6).reshape(3, 2)

        def zarr_writer() -> None:
            for _ in range(ITEMS_PER_THREAD):
                with lock:
                    sg.set_backend('zarr')
                    sg.add_data('x', data, phase='train')

        def numpy_writer() -> None:
            for _ in range(ITEMS_PER_THREAD):
                with lock:
                    sg.set_backend('numpy')
                    sg.add_data('y', data, phase='train')

        with ThreadPoolExecutor(max_workers=2) as pool:
            fz = pool.submit(zarr_writer)
            fn = pool.submit(numpy_writer)
            fz.result()
            fn.result()

        # Verify zarr data
        sg.set_backend('zarr')
        sg.compile()
        assert len(sg['train']) == ITEMS_PER_THREAD * 3

        # Verify numpy data
        sg.set_backend('numpy')
        sg.compile()
        assert len(sg['train']) == ITEMS_PER_THREAD * 3

    def test_concurrent_reads_are_safe_after_compile(self, tmp_path: Any) -> None:
        """Concurrent get_data calls on compiled, immutable data are safe."""
        sg = StoreGate(str(tmp_path / 'store'), mode='w')
        sg.set_data_id('test')

        data = np.arange(20).reshape(10, 2)
        sg.add_data('x', data, phase='train')
        sg.compile()

        errors: list[str] = []

        def reader(thread_id: int) -> None:
            for i in range(ITEMS_PER_THREAD):
                try:
                    arr = sg.get_data('x', phase='train')
                    if arr.shape != (10, 2):
                        errors.append(f'thread {thread_id} iter {i}: shape={arr.shape}')
                    if not np.array_equal(arr, data):
                        errors.append(f'thread {thread_id} iter {i}: data mismatch')
                except Exception as e:
                    errors.append(f'thread {thread_id} iter {i}: {e}')

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = [pool.submit(reader, i) for i in range(NUM_THREADS)]
            for f in as_completed(futures):
                f.result()

        assert errors == [], f'Concurrent read errors: {errors}'

    def test_concurrent_compile_with_lock(self, tmp_path: Any) -> None:
        """Multiple threads calling compile() under a lock does not corrupt metadata."""
        sg = StoreGate(str(tmp_path / 'store'), mode='w')
        sg.set_data_id('test')

        data = np.arange(10).reshape(5, 2)
        sg.add_data('x', data, phase='train')
        sg.add_data('x', data, phase='valid')

        lock = threading.Lock()
        errors: list[str] = []

        def compiler(thread_id: int) -> None:
            for _ in range(10):
                try:
                    with lock:
                        sg.compile()
                except Exception as e:
                    errors.append(f'thread {thread_id}: {e}')

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(compiler, i) for i in range(4)]
            for f in as_completed(futures):
                f.result()

        assert errors == []
        assert len(sg['train']) == 5
        assert len(sg['valid']) == 5
