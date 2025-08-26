from dataclasses import dataclass
from functools import partial

import numpy as np

from cellflow.compat import TorchIterableDataset
from cellflow.data._data import ZarrTrainingData
from cellflow.data._dataloader import CombinedTrainSampler, TrainSampler


def _worker_init_fn_helper(worker_id, random_generators):
    import torch

    del worker_id
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id  # type: ignore[union-attr]
    rng = random_generators[worker_id]
    worker_info.dataset.set_rng(rng)  # type: ignore[union-attr]
    return rng


@dataclass
class TorchCombinedTrainSampler(CombinedTrainSampler, TorchIterableDataset):
    """
    Combined training sampler that iterates over multiple samplers.

    Need to call set_rng(rng) before using the sampler.

    Args:
        samplers: List of training samplers.
        rng: Random number generator.
    """

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample()

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
        random_generators = [np.random.default_rng(s) for s in seq.spawn(len(data_paths))]
        worker_init_fn = partial(_worker_init_fn_helper, random_generators=random_generators)
        data = [ZarrTrainingData.read_zarr(path) for path in data_paths]
        samplers = [TrainSampler(data[i], batch_size) for i in range(len(data))]
        combined_sampler = cls(samplers, weights=weights, dataset_names=dataset_names)
        return torch.utils.data.DataLoader(
            combined_sampler,
            batch_size=None,
            worker_init_fn=worker_init_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
