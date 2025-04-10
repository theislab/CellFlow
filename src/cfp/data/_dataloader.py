import abc
from typing import Any, Literal, Dict, List, Tuple, Callable

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
        # Pre-compile the control_to_perturbation mapping as JAX arrays
        self._control_to_perturbation_lens = jnp.array(
            [len(data.control_to_perturbation[i]) for i in range(self.n_source_dists)]
        )
        max_len = jnp.max(self._control_to_perturbation_lens)
        # Pad the control_to_perturbation arrays to the maximum length
        # Create padded arrays for each source distribution
        padded_arrays = []
        for i in range(self.n_source_dists):
            arr = jnp.array(data.control_to_perturbation[i], dtype=jnp.int32)
            # Pad with a safe value (first element) if needed
            if len(arr) < max_len:
                padding = jnp.full(max_len - len(arr), arr[0], dtype=jnp.int32)
                arr = jnp.concatenate([arr, padding])
            padded_arrays.append(arr)
        self._control_to_perturbation_matrix = jnp.stack(padded_arrays)
        self._control_to_perturbation_keys = list(data.control_to_perturbation.keys())
        self._control_to_perturbation_idxs = jnp.arange(len(self._control_to_perturbation_keys), dtype=jnp.int32)

        # Cache condition keys for efficient lookup
        self._has_condition_data = data.condition_data is not None

        # Define helper functions with explicit parameters
        @jax.jit
        def _sample_target_dist_idx(
            source_dist_idx: jnp.ndarray,
            rng: jax.Array,
            control_to_perturbation_lens: jnp.ndarray,
            control_to_perturbation_matrix: jnp.ndarray,
        ) -> jnp.ndarray:
            """Sample a target distribution index given the source distribution index."""
            target_dist_idx = jax.random.randint(
                rng, shape=(), minval=0, maxval=control_to_perturbation_lens[source_dist_idx]
            )
            return control_to_perturbation_matrix[source_dist_idx, target_dist_idx]

        @jax.jit
        def _get_embeddings(idx: jnp.ndarray, condition_data: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
            """Get embeddings for a given index."""
            # Using explicit return dictionary avoids capturing condition_data in closure
            result = {}
            for key, arr in condition_data.items():
                result[key] = jnp.expand_dims(arr[idx], 0)
            return result

        @jax.jit
        def _sample_from_mask(
            rng: jax.Array, mask: jnp.ndarray, data_idcs: jnp.ndarray
        ) -> jnp.ndarray:
            """Sample indices according to a mask."""
            cond_p = mask / jnp.count_nonzero(mask)
            batch_idcs = jax.random.choice(rng, data_idcs, (self.batch_size,), replace=True, p=cond_p)
            return batch_idcs

        @jax.jit
        def _sample_batch(
            rng: jax.Array,
            cell_data: jnp.ndarray,
            split_covariates_mask: jnp.ndarray,
            perturbation_covariates_mask: jnp.ndarray,
            data_idcs: jnp.ndarray,
            n_source_dists: int,
            condition_data: Dict[str, jnp.ndarray] = None,
            control_to_perturbation_lens: jnp.ndarray = None,
            control_to_perturbation_matrix: jnp.ndarray = None,
        ) -> Dict[str, Any]:
            """Sample a batch of data."""
            # Split the random key
            rng_1, rng_2, rng_3, rng_4 = jax.random.split(rng, 4)

            # Sample source distribution index
            source_dist_idx = jax.random.randint(rng_1, shape=(), minval=0, maxval=n_source_dists)

            # Get source cells
            source_cells_mask = split_covariates_mask == source_dist_idx
            source_batch_idcs = _sample_from_mask(rng_2, source_cells_mask, data_idcs)
            source_batch = cell_data[source_batch_idcs]

            # Get target distribution index using the helper function
            target_dist_idx = _sample_target_dist_idx(source_dist_idx, rng_3, control_to_perturbation_lens, control_to_perturbation_matrix)

            # Get target cells
            target_cells_mask = perturbation_covariates_mask == target_dist_idx
            target_batch_idcs = _sample_from_mask(rng_4, target_cells_mask, data_idcs)
            target_batch = cell_data[target_batch_idcs]

            # Return with or without condition
            if condition_data is None:
                return {"src_cell_data": source_batch, "tgt_cell_data": target_batch}
            else:
                condition_batch = _get_embeddings(target_dist_idx, condition_data)
                return {
                    "src_cell_data": source_batch,
                    "tgt_cell_data": target_batch,
                    "condition": condition_batch,
                }

        # Store the helper functions and main sampling function
        self._sample_target_dist_idx = _sample_target_dist_idx
        self._get_embeddings = _get_embeddings
        self._sample_from_mask = _sample_from_mask
        self._sample_batch = _sample_batch

    def sample(self, rng: jax.Array) -> Any:
        """Sample data for training.

        Parameters
        ----------
        rng
            Random key.

        Returns
        -------
        Dictionary with source and target data from the training data.
        """
        # Pass all data explicitly to the sampling function
        return self._sample_batch(
            rng=rng,
            cell_data=self._data.cell_data,
            split_covariates_mask=self._data.split_covariates_mask,
            perturbation_covariates_mask=self._data.perturbation_covariates_mask,
            data_idcs=self._data_idcs,
            n_source_dists=self.n_source_dists,
            condition_data=self._data.condition_data,
            control_to_perturbation_matrix=self._control_to_perturbation_matrix,
            control_to_perturbation_lens=self._control_to_perturbation_lens,
        )

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


class CpuTrainSampler:
    """NumPy-based data sampler for TrainingData.

    Parameters
    ----------
    data
        The training data.
    batch_size
        The batch size.
    """

    def __init__(self, data, batch_size: int = 1024):
        self._data = data
        self._data_idcs = np.arange(data.cell_data.shape[0])
        self.batch_size = batch_size
        self.n_source_dists = data.n_controls
        self.n_target_dists = data.n_perturbations
        
        # Pre-compute mappings for efficiency
        self._control_to_perturbation_lens = np.array(
            [len(data.control_to_perturbation[i]) for i in range(self.n_source_dists)]
        )
        max_len = np.max(self._control_to_perturbation_lens)
        
        # Create padded arrays for the control_to_perturbation mapping
        padded_arrays = []
        for i in range(self.n_source_dists):
            arr = np.array(data.control_to_perturbation[i], dtype=np.int32)
            # Pad with the first element if needed
            if len(arr) < max_len:
                padding = np.full(max_len - len(arr), arr[0], dtype=np.int32)
                arr = np.concatenate([arr, padding])
            padded_arrays.append(arr)
            
        self._control_to_perturbation_matrix = np.stack(padded_arrays)
        self._control_to_perturbation_keys = list(data.control_to_perturbation.keys())
        self._control_to_perturbation_idxs = np.arange(len(self._control_to_perturbation_keys), dtype=np.int32)
        
        # Cache condition data flag
        self._has_condition_data = data.condition_data is not None

    def _sample_target_dist_idx(self, source_dist_idx, rng):
        """Sample a target distribution index given the source distribution index."""
        max_val = self._control_to_perturbation_lens[source_dist_idx]
        target_dist_idx = rng.integers(0, max_val)
        return self._control_to_perturbation_matrix[source_dist_idx, target_dist_idx]

    def _get_embeddings(self, idx, condition_data):
        """Get embeddings for a given index."""
        result = {}
        for key, arr in condition_data.items():
            result[key] = np.expand_dims(arr[idx], 0)
        return result

    def _sample_from_mask(self, rng, mask):
        """Sample indices according to a mask."""
        # Convert mask to probability distribution
        valid_indices = np.where(mask)[0]
        
        # Handle case with no valid indices (should not happen in practice)
        if len(valid_indices) == 0:
            return rng.choice(self._data_idcs, self.batch_size, replace=True)
            
        # Sample from valid indices with equal probability
        batch_idcs = rng.choice(valid_indices, self.batch_size, replace=True)
        return batch_idcs

    def sample(self, rng) -> Dict[str, Any]:
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
        source_dist_idx = rng.integers(0, self.n_source_dists)
        
        # Get source cells
        source_cells_mask = self._data.split_covariates_mask == source_dist_idx
        source_batch_idcs = self._sample_from_mask(rng, source_cells_mask)
        source_batch = self._data.cell_data[source_batch_idcs]
        
        # Get target distribution index
        target_dist_idx = self._sample_target_dist_idx(source_dist_idx, rng)
        
        # Get target cells
        target_cells_mask = self._data.perturbation_covariates_mask == target_dist_idx
        target_batch_idcs = self._sample_from_mask(rng, target_cells_mask)
        target_batch = self._data.cell_data[target_batch_idcs]
        
        # Return with or without condition
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
    def data(self):
        """The training data."""
        return self._data