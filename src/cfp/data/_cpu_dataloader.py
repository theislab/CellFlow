import abc
from typing import Any, Literal, Dict, List, Tuple, Callable

import numpy as np

from cfp.data._data import PredictionData, TrainingData, ValidationData
from cfp.data._dataloader import BaseValidSampler

__all__ = ["CpuTrainSampler", ]


class CpuTrainSampler:
    """Data sampler for :class:`~cfp.data.TrainingData`.

    Parameters
    ----------
    data
        The training data.
    batch_size
        The batch size.
    """

    def __init__(self, data: TrainingData, batch_size: int = 1024, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self._data = data
        self._data_idcs = np.arange(data.cell_data.shape[0])
        self.batch_size = batch_size
        self.n_source_dists = data.n_controls
        self.n_target_dists = data.n_perturbations
        # Pre-compile the control_to_perturbation mapping as NumPy arrays
        self._control_to_perturbation_lens = np.array(
            [len(data.control_to_perturbation[i]) for i in range(self.n_source_dists)]
        )
        max_len = np.max(self._control_to_perturbation_lens)
        # Pad the control_to_perturbation arrays to the maximum length
        # Create padded arrays for each source distribution
        padded_arrays = []
        for i in range(self.n_source_dists):
            arr = np.array(data.control_to_perturbation[i], dtype=np.int32)
            # Pad with a safe value (first element) if needed
            if len(arr) < max_len:
                padding = np.full(max_len - len(arr), arr[0], dtype=np.int32)
                arr = np.concatenate([arr, padding])
            padded_arrays.append(arr)
        self._control_to_perturbation_matrix = np.stack(padded_arrays)
        self._control_to_perturbation_keys = list(data.control_to_perturbation.keys())
        self._control_to_perturbation_idxs = np.arange(len(self._control_to_perturbation_keys), dtype=np.int32)

        # Cache condition keys for efficient lookup
        self._has_condition_data = data.condition_data is not None

        # Define helper functions with explicit parameters
        
        def _sample_target_dist_idx(
            source_dist_idx: int,
            rng: np.random.Generator,
            control_to_perturbation_lens: np.ndarray,
            control_to_perturbation_matrix: np.ndarray,
        ) -> int:
            """Sample a target distribution index given the source distribution index."""
            target_dist_idx = rng.integers(
                low=0, high=control_to_perturbation_lens[source_dist_idx], size=1
            )[0]
            return control_to_perturbation_matrix[source_dist_idx, target_dist_idx]

        def _get_embeddings(idx: int, condition_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            """Get embeddings for a given index."""
            # Using explicit return dictionary
            result = {}
            for key, arr in condition_data.items():
                result[key] = np.expand_dims(arr[idx], 0)
            return result

        def _sample_from_mask(
            rng: np.random.Generator, mask: np.ndarray, data_idcs: np.ndarray
        ) -> np.ndarray:
            """Sample indices according to a mask."""
            cond_p = mask / np.count_nonzero(mask)
            batch_idcs = rng.choice(data_idcs, size=self.batch_size, replace=True, p=cond_p)
            return batch_idcs

        def _sample_batch(
            rng: np.random.Generator,
            cell_data: np.ndarray,
            split_covariates_mask: np.ndarray,
            perturbation_covariates_mask: np.ndarray,
            data_idcs: np.ndarray,
            n_source_dists: int,
            condition_data: Dict[str, np.ndarray] = None,
            control_to_perturbation_lens: np.ndarray = None,
            control_to_perturbation_matrix: np.ndarray = None,
        ) -> Dict[str, Any]:
            """Sample a batch of data."""
            # Sample source distribution index
            source_dist_idx = rng.integers(low=0, high=n_source_dists, size=1)[0]

            # Get source cells
            source_cells_mask = split_covariates_mask == source_dist_idx
            source_batch_idcs = _sample_from_mask(rng, source_cells_mask, data_idcs)
            source_batch = cell_data[source_batch_idcs]

            # Get target distribution index using the helper function
            target_dist_idx = _sample_target_dist_idx(
                source_dist_idx, rng, control_to_perturbation_lens, control_to_perturbation_matrix
            )

            # Get target cells
            target_cells_mask = perturbation_covariates_mask == target_dist_idx
            target_batch_idcs = _sample_from_mask(rng, target_cells_mask, data_idcs)
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
        
        # Initialize the random number generator
        self.rng = np.random.default_rng()

    def sample(self) -> Any:
        """Sample data for training.

        Parameters
        ----------
        seed
            Optional random seed to use for this sample.

        Returns
        -------
        Dictionary with source and target data from the training data.
        """

        rng = self.rng
            
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

