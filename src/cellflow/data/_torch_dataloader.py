from dataclasses import dataclass
from functools import partial

import numpy as np

from cellflow.compat import TorchIterableDataset
from cellflow.data._dataloader import (
    CombinedTrainSampler,
    CombinedTrainSamplerWithPool,
)


def _worker_init_fn_helper(worker_id, random_generators):
    import torch

    del worker_id
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id  # type: ignore[union-attr]
    rng = random_generators[worker_id]
    worker_info.dataset.set_rng(rng)  # type: ignore[union-attr]
    return rng


@dataclass
class TorchCombinedTrainSampler(TorchIterableDataset):
    """
    Combined training sampler that iterates over multiple samplers.

    Need to call set_rng(rng) before using the sampler.

    Args:
        inner_sampler: The inner sampler.
    """

    inner_sampler: CombinedTrainSampler | CombinedTrainSamplerWithPool

    def __iter__(self):
        return self

    def __next__(self):
        return self.inner_sampler.sample()

    @classmethod
    def from_zarr_paths(
        cls,
        data_paths: list[str],
        use_pool: bool = False,
        batch_size: int = 1024,
        seed: int = 42,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        weights: np.ndarray | None = None,
        dataset_names: list[str] | None = None,
        pool_size: int | None = 100,
        replacement_prob: float | None = 0.01,
    ):
        if use_pool and (pool_size is None or replacement_prob is None):
            raise ValueError("pool_size and replacement_prob must be provided when use_pool is True")
        elif not use_pool and (pool_size is not None or replacement_prob is not None):
            raise ValueError("pool_size and replacement_prob must not be provided when use_pool is False")
        import torch

        seq = np.random.SeedSequence(seed)
        random_generators = [np.random.default_rng(s) for s in seq.spawn(num_workers)]
        worker_init_fn = partial(_worker_init_fn_helper, random_generators=random_generators)
        if use_pool:
            inner_sampler = CombinedTrainSamplerWithPool.from_zarr_paths(
                data_paths,
                weights=weights,
                dataset_names=dataset_names,
                batch_size=batch_size,
                pool_size=pool_size,
                replacement_prob=replacement_prob,
            )
        else:
            inner_sampler = CombinedTrainSampler.from_zarr_paths(
                data_paths,
                weights=weights,
                batch_size=batch_size,
                dataset_names=dataset_names,
            )
        combined_sampler = cls(inner_sampler)
        return torch.utils.data.DataLoader(
            combined_sampler,
            batch_size=None,
            worker_init_fn=worker_init_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
