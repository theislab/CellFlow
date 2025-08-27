import abc
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from cellflow.data._data import (
    PredictionData,
    TrainingData,
    ValidationData,
    ZarrTrainingData,
)

__all__ = [
    "TrainSampler",
    "ValidationSampler",
    "PredictionSampler",
]


class TrainSampler:
    """Data sampler for :class:`~cellflow.data.TrainingData`.

    Parameters
    ----------
    data
        The training data.
    batch_size
        The batch size.

    """

    def __init__(self, data: TrainingData | ZarrTrainingData, batch_size: int = 1024):
        self._data = data
        self._data_idcs = np.arange(data.cell_data.shape[0])
        self.batch_size = batch_size
        self.n_source_dists = data.n_controls
        self.n_target_dists = data.n_perturbations

        self._control_to_perturbation_keys = sorted(data.control_to_perturbation.keys())
        self._has_condition_data = data.condition_data is not None

    def _sample_source_dist_idx(self, rng) -> int:
        """Sample a source distribution index."""
        return rng.choice(self.n_source_dists)

    def _sample_target_dist_idx(self, source_dist_idx, rng):
        """Sample a target distribution index given the source distribution index."""
        return rng.choice(self._data.control_to_perturbation[source_dist_idx])

    def _get_embeddings(self, idx, condition_data) -> dict[str, np.ndarray]:
        """Get embeddings for a given index."""
        result = {}
        for key, arr in condition_data.items():
            result[key] = np.expand_dims(arr[idx], 0)
        return result

    def _sample_from_mask(self, rng, mask) -> np.ndarray:
        """Sample indices according to a mask."""
        # Convert mask to probability distribution
        valid_indices = np.where(mask)[0]

        # Handle case with no valid indices (should not happen in practice)
        if len(valid_indices) == 0:
            raise ValueError("No valid indices found in the mask")

        # Sample from valid indices with equal probability
        batch_idcs = rng.choice(valid_indices, self.batch_size, replace=True)
        return batch_idcs

    def sample(self, rng) -> dict[str, Any]:
        """Sample a batch of data.

        Parameters
        ----------
        seed : int, optional
            Random seed

        Returns
        -------
        Dictionary with source and target data
        """
        # Sample source distribution index
        source_dist_idx = self._sample_source_dist_idx(rng)

        # Get source cells
        source_cells_mask = self._data.split_covariates_mask == source_dist_idx
        source_batch_idcs = self._sample_from_mask(rng, source_cells_mask)
        source_batch = self._data.cell_data[source_batch_idcs]

        target_dist_idx = self._sample_target_dist_idx(source_dist_idx, rng)
        target_cells_mask = self._data.perturbation_covariates_mask == target_dist_idx
        target_batch_idcs = self._sample_from_mask(rng, target_cells_mask)
        target_batch = self._data.cell_data[target_batch_idcs]

        if not self._has_condition_data:
            return {"src_cell_data": source_batch, "tgt_cell_data": target_batch}
        else:
            condition_batch = self._get_embeddings(target_dist_idx, self._data.condition_data)
            return {
                "src_cell_data": source_batch,
                "tgt_cell_data": target_batch,
                "condition": condition_batch,
            }

    @property
    def data(self) -> TrainingData | ZarrTrainingData:
        """The training data."""
        return self._data


@dataclass
class TrainSamplerWithPool(TrainSampler):
    """Data sampler with gradual pool replacement using reservoir sampling.

    This approach replaces pool elements one by one rather than refreshing
    the entire pool, providing better cache locality while maintaining
    reasonable randomness.

    Parameters
    ----------
    data
        The training data.
    batch_size
        The batch size.
    pool_size
        The size of the pool of source distribution indices.
    replacement_prob
        Probability of replacing a pool element after each sample.
        Lower values = longer cache retention, less randomness.
        Higher values = faster cache turnover, more randomness.
    replace_in_pool
        Whether to allow replacement when sampling from the pool.
    """

    def __init__(
        self,
        data: TrainingData | ZarrTrainingData,
        batch_size: int = 1024,
        pool_size: int = 100,
        replacement_prob: float = 0.01,
    ):
        super().__init__(data, batch_size)
        self._pool_size = pool_size
        self._replacement_prob = replacement_prob
        self._src_idx_pool = np.empty(self._pool_size, dtype=int)
        self._pool_usage_count = np.zeros(self._pool_size, dtype=int)
        self._all_src_idx_pool = set(range(self.n_source_dists))
        self._initialized = False

    def _init_pool(self, rng):
        """Initialize the pool with random source distribution indices."""
        self._src_idx_pool = rng.choice(self.n_source_dists, size=self._pool_size, replace=False)
        self._initialized = True

    def _sample_source_dist_idx(self, rng) -> int:
        """Sample a source distribution index with gradual pool replacement."""
        if not self._initialized:
            self._init_pool(rng)

        # Sample from current pool
        pool_idx = rng.choice(self._pool_size)
        source_idx = self._src_idx_pool[pool_idx]

        # Increment usage count for monitoring
        self._pool_usage_count[pool_idx] += 1

        # Gradually replace elements based on replacement probability
        if rng.random() < self._replacement_prob:
            self._replace_pool_element(rng, pool_idx)

        return source_idx

    def _replace_pool_element(self, rng, pool_idx):
        """Replace a single pool element with a new one."""
        # Get all indices not currently in the pool
        available_indices = list(self._all_src_idx_pool - set(self._src_idx_pool))

        if available_indices:
            # Choose new element (could be weighted by usage count)
            new_idx = rng.choice(available_indices)
            self._src_idx_pool[pool_idx] = new_idx
            self._pool_usage_count[pool_idx] = 0

    def get_pool_stats(self) -> dict:
        """Get statistics about the current pool state."""
        if self._src_idx_pool is None:
            return {"pool_size": 0, "avg_usage": 0, "unique_sources": 0}
        return {
            "pool_size": self._pool_size,
            "avg_usage": float(np.mean(self._pool_usage_count)),
            "unique_sources": len(set(self._src_idx_pool)),
            "pool_elements": self._src_idx_pool.copy(),
            "usage_counts": self._pool_usage_count.copy(),
        }


class BaseValidSampler(abc.ABC):
    @abc.abstractmethod
    def sample(*args, **kwargs):
        pass

    def _get_key(self, cond_idx: int) -> tuple[str, ...]:
        if len(self._data.perturbation_idx_to_id):  # type: ignore[attr-defined]
            return self._data.perturbation_idx_to_id[cond_idx]  # type: ignore[attr-defined]
        cov_combination = self._data.perturbation_idx_to_covariates[cond_idx]  # type: ignore[attr-defined]
        return tuple(cov_combination[i] for i in range(len(cov_combination)))

    def _get_perturbation_to_control(self, data: ValidationData | PredictionData) -> dict[int, np.ndarray]:
        d = {}
        for k, v in data.control_to_perturbation.items():
            for el in v:
                d[el] = k
        return d

    def _get_condition_data(self, cond_idx: int) -> dict[str, np.ndarray]:
        return {k: v[[cond_idx], ...] for k, v in self._data.condition_data.items()}  # type: ignore[attr-defined]


class ValidationSampler(BaseValidSampler):
    """Data sampler for :class:`~cellflow.data.ValidationData`.

    Parameters
    ----------
    val_data
        The validation data.
    seed
        Random seed.
    """

    def __init__(self, val_data: ValidationData, seed: int = 0) -> None:
        self._data = val_data
        self.perturbation_to_control = self._get_perturbation_to_control(val_data)
        self.n_conditions_on_log_iteration = (
            val_data.n_conditions_on_log_iteration
            if val_data.n_conditions_on_log_iteration is not None
            else val_data.n_perturbations
        )
        self.n_conditions_on_train_end = (
            val_data.n_conditions_on_train_end
            if val_data.n_conditions_on_train_end is not None
            else val_data.n_perturbations
        )
        self.rng = np.random.default_rng(seed)
        if self._data.condition_data is None:
            raise NotImplementedError("Validation data must have condition data.")

    def sample(self, mode: Literal["on_log_iteration", "on_train_end"]) -> Any:
        """Sample data for validation.

        Parameters
        ----------
        mode
            Sampling mode. Either ``"on_log_iteration"`` or ``"on_train_end"``.

        Returns
        -------
        Dictionary with source, condition, and target data from the validation data.
        """
        size = self.n_conditions_on_log_iteration if mode == "on_log_iteration" else self.n_conditions_on_train_end
        condition_idcs = self.rng.choice(self._data.n_perturbations, size=(size,), replace=False)

        source_idcs = [self.perturbation_to_control[cond_idx] for cond_idx in condition_idcs]
        source_cells_mask = [self._data.split_covariates_mask == source_idx for source_idx in source_idcs]
        source_cells = [self._data.cell_data[mask] for mask in source_cells_mask]
        target_cells_mask = [cond_idx == self._data.perturbation_covariates_mask for cond_idx in condition_idcs]
        target_cells = [self._data.cell_data[mask] for mask in target_cells_mask]
        conditions = [self._get_condition_data(cond_idx) for cond_idx in condition_idcs]
        cell_rep_dict = {}
        cond_dict = {}
        true_dict = {}
        for i in range(len(condition_idcs)):
            k = self._get_key(condition_idcs[i])
            cell_rep_dict[k] = source_cells[i]
            cond_dict[k] = conditions[i]
            true_dict[k] = target_cells[i]

        return {"source": cell_rep_dict, "condition": cond_dict, "target": true_dict}

    @property
    def data(self) -> ValidationData:
        """The validation data."""
        return self._data


class PredictionSampler(BaseValidSampler):
    """Data sampler for :class:`~cellflow.data.PredictionData`.

    Parameters
    ----------
    pred_data
        The prediction data.

    """

    def __init__(self, pred_data: PredictionData) -> None:
        self._data = pred_data
        self.perturbation_to_control = self._get_perturbation_to_control(pred_data)
        if self._data.condition_data is None:
            raise NotImplementedError("Validation data must have condition data.")

    def sample(self) -> Any:
        """Sample data for prediction.

        Returns
        -------
        Dictionary with source and condition data from the prediction data.
        """
        condition_idcs = range(self._data.n_perturbations)

        source_idcs = [self.perturbation_to_control[cond_idx] for cond_idx in condition_idcs]
        source_cells_mask = [self._data.split_covariates_mask == source_idx for source_idx in source_idcs]
        source_cells = [self._data.cell_data[mask] for mask in source_cells_mask]
        conditions = [self._get_condition_data(cond_idx) for cond_idx in condition_idcs]
        cell_rep_dict = {}
        cond_dict = {}
        for i in range(len(condition_idcs)):
            k = self._get_key(condition_idcs[i])
            cell_rep_dict[k] = source_cells[i]
            cond_dict[k] = conditions[i]

        return {
            "source": cell_rep_dict,
            "condition": cond_dict,
        }

    @property
    def data(self) -> PredictionData:
        """The training data."""
        return self._data


@dataclass
class CombinedTrainSampler:
    """
    Combined training sampler that iterates over multiple samplers.

    Need to call set_rng(rng) before using the sampler.

    Args:
        samplers: List of training samplers.
        weights: Weights for the samplers.
        dataset_names: Names for the samplers.
        rng: Random number generator.
    """

    samplers: list[TrainSampler]
    weights: np.ndarray | None = None
    dataset_names: list[str] | None = None
    rng: np.random.Generator | None = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = np.ones(len(self.samplers))
        self.weights = np.asarray(self.weights)
        assert len(self.weights) == len(self.samplers)
        self.weights = self.weights / self.weights.sum()

    def set_rng(self, rng: np.random.Generator):
        self.rng = rng

    def sample(self, *args, **kwargs) -> dict[str, Any]:
        del args, kwargs
        if self.rng is None:
            raise ValueError("Please call set_rng() before using the sampler.")
        dataset_idx = self.rng.choice(len(self.samplers), p=self.weights)
        res = self.samplers[dataset_idx].sample(self.rng)
        if self.dataset_names is not None:
            res["dataset_name"] = self.dataset_names[dataset_idx]
        return res

    @staticmethod
    def train_sampler_class() -> type[TrainSampler]:
        return TrainSampler

    @classmethod
    def _from_zarr_paths(
        cls,
        data_paths: list[str],
        weights: np.ndarray | None = None,
        dataset_names: list[str] | None = None,
        sampler_kwargs: dict[str, Any] | None = None,
    ) -> "CombinedTrainSampler":
        """Create a combined sampler for multiple datasets.

        Args:
            data_paths: List of paths to the datasets.
            weights: Weights for the datasets.
            dataset_names: Names for the datasets.
            sampler_kwargs: Batch size, pool size, replacement probability, whenever applicable.
        """
        samplers = []
        sampler_kwargs = sampler_kwargs or {}
        for data_path in data_paths:
            data = ZarrTrainingData.read_zarr(data_path)
            sampler = cls.train_sampler_class()(data=data, **sampler_kwargs)
            samplers.append(sampler)

        combined_sampler = cls(
            samplers=samplers,
            weights=weights,
            dataset_names=dataset_names,
        )

        return combined_sampler

    @classmethod
    def from_zarr_paths(
        cls,
        data_paths: list[str],
        weights: np.ndarray | None = None,
        dataset_names: list[str] | None = None,
        batch_size: int = 1024,
    ):
        """Create a combined sampler for multiple datasets.

        Args:
            data_paths: List of paths to the datasets.
            weights: Weights for the datasets.
            dataset_names: Names for the datasets.
            batch_size: Batch size.
        """
        return cls._from_zarr_paths(
            data_paths,
            weights=weights,
            dataset_names=dataset_names,
            sampler_kwargs={"batch_size": batch_size},
        )


@dataclass
class CombinedTrainSamplerWithPool(CombinedTrainSampler):
    """Combined training sampler that iterates over multiple samplers with a pool of source distribution indices.

    Args:
        samplers: List of training samplers.
        pool_size: Size of the pool of source distribution indices.
        pool_refresh_freq: Frequency of refreshing the pool of source distribution indices.
        replace_in_refresh: Whether to replace the source distribution indices in the pool when refreshing.
        replace_in_pool: Whether to replace the source distribution indices in the pool when sampling.
    """

    samplers: list[TrainSamplerWithPool]
    weights: np.ndarray | None = None
    dataset_names: list[str] | None = None
    rng: np.random.Generator | None = None

    def get_pool_stats(self) -> dict[str, dict]:
        """Get pool statistics for all samplers that support it."""
        stats = {}
        for i, sampler in enumerate(self.samplers):
            if hasattr(sampler, "get_pool_stats"):
                name = self.dataset_names[i] if self.dataset_names else f"sampler_{i}"
                stats[name] = sampler.get_pool_stats()
        return stats

    @staticmethod
    def train_sampler_class() -> type[TrainSamplerWithPool]:
        return TrainSamplerWithPool

    @classmethod
    def from_zarr_paths(
        cls,
        data_paths: list[str],
        weights: np.ndarray | None = None,
        dataset_names: list[str] | None = None,
        batch_size: int = 1024,
        pool_size: int = 100,
        replacement_prob: float = 0.01,
    ):
        """Create a combined sampler for multiple datasets.

        Args:
            data_paths: List of paths to the datasets.
            weights: Weights for the datasets.
            dataset_names: Names for the datasets.
            batch_size: Batch size.
            pool_size: Size of the pool of source distribution indices.
            replacement_prob: Probability of replacing a pool element after each sample.
        """
        return cls._from_zarr_paths(
            data_paths,
            weights=weights,
            dataset_names=dataset_names,
            sampler_kwargs={
                "batch_size": batch_size,
                "pool_size": pool_size,
                "replacement_prob": replacement_prob,
            },
        )
