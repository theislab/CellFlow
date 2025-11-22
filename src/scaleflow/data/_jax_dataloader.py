import queue
import threading
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import numpy as np

from scaleflow.data._data import (
    TrainingData,
)
from scaleflow.data._dataloader import TrainSampler


def _prefetch_to_device(
    sampler: TrainSampler,
    seed: int,
    num_iterations: int,
    prefetch_factor: int = 2,
    num_workers: int = 4,
) -> Generator[dict[str, Any], None, None]:
    import jax

    seq = np.random.SeedSequence(seed)
    random_generators = [np.random.default_rng(s) for s in seq.spawn(num_workers)]

    q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=prefetch_factor * num_workers)
    sem = threading.Semaphore(num_iterations)
    stop_event = threading.Event()

    def worker(rng: np.random.Generator):
        while not stop_event.is_set() and sem.acquire(blocking=False):
            batch = sampler.sample(rng)
            batch = jax.device_put(batch, jax.devices()[0], donate=True)
            jax.block_until_ready(batch)
            while not stop_event.is_set():
                try:
                    q.put(batch, timeout=1.0)
                    break  # Batch successfully put into the queue; break out of retry loop
                except queue.Full:
                    continue

        return

    # Start multiple worker threads
    ts = []
    for i in range(num_workers):
        t = threading.Thread(target=worker, daemon=True, name=f"worker-{i}", args=(random_generators[i],))
        t.start()
        ts.append(t)

    try:
        for _ in range(num_iterations):
            # Yield batches from the queue; blocks waiting for available batch
            yield q.get()
    finally:
        # When the generator is closed or garbage collected, clean up the worker threads
        stop_event.set()  # Signal all workers to exit
        for t in ts:
            t.join()  # Wait for all worker threads to finish


@dataclass
class JaxOutOfCoreTrainSampler:
    """
    A sampler that prefetches batches to the GPU for out-of-core training.

    Here out-of-core means that data can be more than the GPU memory.

    Parameters
    ----------
    data
        The training data.
    seed
        The seed for the random number generator.
    batch_size
        The batch size.
    num_workers
        The number of workers to use for prefetching.
    prefetch_factor
        The prefetch factor similar to PyTorch's DataLoader.

    """

    data: TrainingData
    seed: int
    batch_size: int = 1024
    num_workers: int = 4
    prefetch_factor: int = 2

    def __post_init__(self):
        self.inner = TrainSampler(data=self.data, batch_size=self.batch_size)
        self._iterator = None

    def set_sampler(self, num_iterations: int) -> None:
        self._iterator = _prefetch_to_device(
            sampler=self.inner,
            seed=self.seed,
            num_iterations=num_iterations,
            prefetch_factor=self.prefetch_factor,
            num_workers=self.num_workers,
        )

    def sample(self, rng=None) -> dict[str, Any]:
        if self._iterator is None:
            raise ValueError(
                "Sampler not set. Use `set_sampler` to set the sampler with"
                "the number of iterations. Without the number of iterations,"
                " the sampler will not be able to sample the data."
            )
        if rng is not None:
            del rng
        return next(self._iterator)
