from dataclasses import dataclass
from functools import partial

import numpy as np

from scaleflow.compat import TorchIterableDataset
from scaleflow.data._data import MappedCellData
from scaleflow.data._dataloader import TrainSampler, ReservoirSampler


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
        samplers: List of training samplers.
        rng: Random number generator.
    """

    samplers: list[TrainSampler]
    weights: np.ndarray | None = None
    rng: np.random.Generator | None = None
    dataset_names: list[str] | None = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = np.ones(len(self.samplers))
        self.weights = np.asarray(self.weights)
        assert len(self.weights) == len(self.samplers)
        self.weights = self.weights / self.weights.sum()

    def set_rng(self, rng: np.random.Generator):
        self.rng = rng

    def __iter__(self):
        return self

    def __next__(self):
        if self.rng is None:
            raise ValueError("Please call set_rng() before using the sampler.")
        dataset_idx = self.rng.choice(len(self.samplers), p=self.weights)
        res = self.samplers[dataset_idx].sample(self.rng)
        if self.dataset_names is not None:
            res["dataset_name"] = self.dataset_names[dataset_idx]
        return res

    @classmethod
    def combine_zarr_training_samplers(
        cls,
        data_paths: list[str],
        batch_size: int = 1024,
        seed: int = 42,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        weights: np.ndarray | None = None,
        dataset_names: list[str] | None = None,
    ):
        import torch

        seq = np.random.SeedSequence(seed)
        random_generators = [np.random.default_rng(s) for s in seq.spawn(num_workers)]
        worker_init_fn = partial(_worker_init_fn_helper, random_generators=random_generators)
        data = [MappedCellData.read_zarr(path) for path in data_paths]
        samplers = [ReservoirSampler(data[i], batch_size) for i in range(len(data))]
        combined_sampler = cls(samplers, weights=weights, dataset_names=dataset_names)
        return torch.utils.data.DataLoader(
            combined_sampler,
            batch_size=None,
            worker_init_fn=worker_init_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
