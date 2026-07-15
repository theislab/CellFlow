from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from cellflow._types import ArrayLike

__all__ = [
    "BaseDataMixin",
    "ConditionData",
    "PredictionData",
]


@dataclass
class ReturnData:  # TODO: this should rather be a NamedTuple
    split_covariates_mask: np.ndarray | None
    split_idx_to_covariates: dict[int, tuple[Any, ...]]
    perturbation_covariates_mask: np.ndarray | None
    perturbation_idx_to_covariates: dict[int, tuple[Any, ...]]
    perturbation_idx_to_id: dict[int, Any]
    condition_data: dict[str, np.ndarray]
    control_to_perturbation: dict[int, np.ndarray]
    max_combination_length: int


class BaseDataMixin:
    """Base class for data containers."""

    @property
    def n_controls(self) -> int:
        """Returns the number of control covariate values."""
        return len(self.split_idx_to_covariates)  # type: ignore[attr-defined]

    @property
    def n_perturbations(self) -> int:
        """Returns the number of perturbation covariate combinations."""
        return len(self.perturbation_idx_to_covariates)  # type: ignore[attr-defined]

    @property
    def n_perturbation_covariates(self) -> int:
        """Returns the number of perturbation covariates."""
        return len(self.condition_data)  # type: ignore[attr-defined]

    def _format_params(self, fmt: Callable[[Any], str]) -> str:
        params = {
            "n_controls": self.n_controls,
            "n_perturbations": self.n_perturbations,
        }
        return ", ".join(f"{name}={fmt(val)}" for name, val in params.items())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self._format_params(repr)}]"


@dataclass
class ConditionData(BaseDataMixin):
    """Data container containing condition embeddings.

    Parameters
    ----------
    condition_data
        Dictionary with embeddings for conditions.
    max_combination_length
        Maximum number of covariates in a combination.
    null_value
        Token to use for masking `null_value`.
    data_manager
        Data manager used to generate the data.
    """

    condition_data: dict[str, np.ndarray]
    max_combination_length: int
    perturbation_idx_to_covariates: dict[int, tuple[str, ...]]
    perturbation_idx_to_id: dict[int, Any]
    null_value: Any
    data_manager: Any


@dataclass
class PredictionData(BaseDataMixin):
    """Data container to perform prediction.

    Parameters
    ----------
    src_data
        Dictionary with data for source cells.
    condition_data
        Dictionary with embeddings for conditions.
    control_to_perturbation
        Mapping from control index to target distribution indices.
    covariate_encoder
        Encoder for the primary covariate.
    max_combination_length
        Maximum number of covariates in a combination.
    null_value
        Token to use for masking ``null_value``.
    """

    cell_data: jax.Array  # (n_cells, n_features)
    split_covariates_mask: jax.Array  # (n_cells,), which cell assigned to which source distribution
    split_idx_to_covariates: dict[int, tuple[Any, ...]]  # (n_sources,) dictionary explaining split_covariates_mask
    perturbation_idx_to_covariates: dict[
        int, tuple[str, ...]
    ]  # (n_targets,), dictionary explaining perturbation_covariates_mask
    perturbation_idx_to_id: dict[int, Any]
    condition_data: dict[str, ArrayLike]  # (n_targets,) all embeddings for conditions
    control_to_perturbation: dict[int, ArrayLike]
    max_combination_length: int
    null_value: Any
    data_manager: Any
