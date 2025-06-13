import abc
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from cellflow.data._data import PredictionData, TrainingData, ValidationData

__all__ = ["TrainSampler", "ValidationSampler", "PredictionSampler"]


class TrainSampler:
    """Data sampler for :class:`~cellflow.data.TrainingData`.

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
        self.conditional_samplings = [
            lambda key, i=i: jax.random.choice(key, self._data.control_to_perturbation[i])
            for i in range(self.n_source_dists)
        ]
        self.get_embeddings = lambda idx: {
            pert_cov: jnp.expand_dims(arr[idx], 0) for pert_cov, arr in self._data.condition_data.items()
        }

        @jax.jit
        def _sample(rng: jax.Array) -> Any:
            rng_1, rng_2, rng_3, rng_4 = jax.random.split(rng, 4)
            source_dist_idx = jax.random.choice(rng_1, self.n_source_dists)
            source_cells_mask = self._data.split_covariates_mask == source_dist_idx
            src_cond_p = source_cells_mask / jnp.count_nonzero(source_cells_mask)
            source_batch_idcs = jax.random.choice(rng_2, self._data_idcs, [self.batch_size], replace=True, p=src_cond_p)

            source_batch = self._data.cell_data[source_batch_idcs]

            target_dist_idx = jax.lax.switch(source_dist_idx, self.conditional_samplings, rng_3)
            target_cells_mask = self._data.perturbation_covariates_mask == target_dist_idx
            tgt_cond_p = target_cells_mask / jnp.count_nonzero(target_cells_mask)
            target_batch_idcs = jax.random.choice(
                rng_4,
                self._data_idcs,
                [self.batch_size],
                replace=True,
                p=tgt_cond_p,
            )
            target_batch = self._data.cell_data[target_batch_idcs]
            if self._data.condition_data is None:
                return {"src_cell_data": source_batch, "tgt_cell_data": target_batch}

            condition_batch = self.get_embeddings(target_dist_idx)
            return {
                "src_cell_data": source_batch,
                "tgt_cell_data": target_batch,
                "condition": condition_batch,
            }

        self.sample = _sample

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

    def _get_perturbation_to_control(self, data: ValidationData | PredictionData) -> dict[int, int]:
        d = {}
        for k, v in data.control_to_perturbation.items():
            for el in v:
                d[el] = k
        return d

    def _get_condition_data(self, cond_idx: int) -> jnp.ndarray:
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
