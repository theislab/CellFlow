import abc
from typing import Any, Literal, Tuple, Dict, Callable

import jax
import jax.numpy as jnp
import numpy as np

from cfp.data._data import PredictionData, TrainingData, ValidationData

__all__ = ["TrainSampler", "ValidationSampler", "PredictionSampler"]


class TrainSampler:
    """Data sampler for :class:`~cfp.data.TrainingData`.

    Parameters
    ----------
    data
        The training data.
    batch_size
        The batch size.

    """

    def __init__(self, data: TrainingData, batch_size: int = 1024):
        self._data = data
        self._data_idcs = jnp.arange(data.cell_data.shape[0])
        self.batch_size = batch_size
        self.n_source_dists = data.n_controls
        self.n_target_dists = data.n_perturbations
        self.get_embeddings = lambda idx: {
            pert_cov: jnp.expand_dims(arr[idx], 0)
            for pert_cov, arr in self._data.condition_data.items()
        }

        # Store masks for each source distribution for efficient lookup
        self._source_dist_masks = []
        for i in range(self.n_source_dists):
            mask = self._data.split_covariates_mask == i
            self._source_dist_masks.append(mask)

        # Store masks for each target distribution for efficient lookup
        self._target_dist_masks = []
        for i in range(self.n_target_dists):
            mask = self._data.perturbation_covariates_mask == i
            self._target_dist_masks.append(mask)

        # Pre-compile the smaller function that only handles random sampling
        self._sample_indices = jax.jit(self._create_sample_indices_fn())

        # Separate conditional sampling functions to avoid tracing large arrays
        self._conditional_sampling_fns = []
        for i in range(self.n_source_dists):
            targets = self._data.control_to_perturbation.get(i, jnp.array([]))
            if len(targets) > 0:
                self._conditional_sampling_fns.append(
                    lambda key, targets=targets: targets[jax.random.randint(key, (), 0, len(targets))]
                )
            else:
                self._conditional_sampling_fns.append(lambda key: 0)  # Fallback

    def _create_sample_indices_fn(self) -> Callable:
        """Create a JAX function for sampling indices without tracing large arrays."""

        def sample_indices(
            rng: jax.Array, source_dist_idx: int, target_dist_idx: int, source_mask_sum: int, target_mask_sum: int
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Sample source and target indices using provided distribution indices."""
            rng_1, rng_2 = jax.random.split(rng)

            # Create probability distribution for the source
            source_batch_idcs = jax.random.randint(rng_1, (self.batch_size,), 0, source_mask_sum)

            # Create probability distribution for the target
            target_batch_idcs = jax.random.randint(rng_2, (self.batch_size,), 0, target_mask_sum)

            return source_batch_idcs, target_batch_idcs

        return sample_indices

    def sample(self, rng: jax.Array) -> dict:
        """Sample a batch of data for training.

        Parameters
        ----------
        rng
            JAX random key.

        Returns
        -------
        Dictionary with source and target data.
        """
        # Split the random keys
        rng_1, rng_2 = jax.random.split(rng)

        # Select a random source distribution
        source_dist_idx = jax.random.randint(rng_1, (), 0, self.n_source_dists).item()
        source_mask = self._source_dist_masks[source_dist_idx]
        source_indices = jnp.where(source_mask)[0]
        source_mask_sum = len(source_indices)

        # Get the corresponding target distribution
        # Using Python function call instead of jax.lax.switch to avoid tracing
        target_dist_idx = self._conditional_sampling_fns[source_dist_idx](rng_2).item()
        target_mask = self._target_dist_masks[target_dist_idx]
        target_indices = jnp.where(target_mask)[0]
        target_mask_sum = len(target_indices)

        # Sample indices using the JAX-compiled function
        source_relative_idcs, target_relative_idcs = self._sample_indices(
            rng, source_dist_idx, target_dist_idx, source_mask_sum, target_mask_sum
        )

        # Convert to actual indices in the dataset
        source_batch_idcs = source_indices[source_relative_idcs]
        target_batch_idcs = target_indices[target_relative_idcs]

        # Get the actual data
        source_batch = self._data.cell_data[source_batch_idcs]
        target_batch = self._data.cell_data[target_batch_idcs]

        if self._data.condition_data is None:
            return {"src_cell_data": source_batch, "tgt_cell_data": target_batch}

        # Get embeddings for the target condition
        condition_batch = self.get_embeddings(target_dist_idx)

        return {
            "src_cell_data": source_batch,
            "tgt_cell_data": target_batch,
            "condition": condition_batch,
        }

    @property
    def data(self) -> TrainingData:
        """The training data."""
        return self._data


class BaseValidSampler(abc.ABC):

    @abc.abstractmethod
    def sample(*args, **kwargs):
        pass

    def _get_key(self, cond_idx: int) -> tuple[str, ...]:
        if len(self._data.perturbation_idx_to_id):  # type: ignore[attr-defined]
            return self._data.perturbation_idx_to_id[cond_idx]  # type: ignore[attr-defined]
        cov_combination = self._data.perturbation_idx_to_covariates[cond_idx]  # type: ignore[attr-defined]
        return tuple(cov_combination[i] for i in range(len(cov_combination)))

    def _get_perturbation_to_control(
        self, data: ValidationData | PredictionData
    ) -> dict[int, int]:
        d = {}
        for k, v in data.control_to_perturbation.items():
            for el in v:
                d[el] = k
        return d

    def _get_condition_data(self, cond_idx: int) -> jnp.ndarray:
        return {k: v[[cond_idx], ...] for k, v in self._data.condition_data.items()}  # type: ignore[attr-defined]


class ValidationSampler(BaseValidSampler):
    """Data sampler for :class:`~cfp.data.ValidationData`.

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
        size = (
            self.n_conditions_on_log_iteration
            if mode == "on_log_iteration"
            else self.n_conditions_on_train_end
        )
        condition_idcs = self.rng.choice(
            self._data.n_perturbations, size=(size,), replace=False
        )

        source_idcs = [
            self.perturbation_to_control[cond_idx] for cond_idx in condition_idcs
        ]
        source_cells_mask = [
            self._data.split_covariates_mask == source_idx for source_idx in source_idcs
        ]
        source_cells = [self._data.cell_data[mask] for mask in source_cells_mask]
        target_cells_mask = [
            cond_idx == self._data.perturbation_covariates_mask
            for cond_idx in condition_idcs
        ]
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
    """Data sampler for :class:`~cfp.data.PredictionData`.

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

        source_idcs = [
            self.perturbation_to_control[cond_idx] for cond_idx in condition_idcs
        ]
        source_cells_mask = [
            self._data.split_covariates_mask == source_idx for source_idx in source_idcs
        ]
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
