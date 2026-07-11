from typing import Any

import numpy as np
from flax.training import train_state

from cellflow._types import ArrayLike

__all__ = ["BaseSolver"]


class BaseSolver:
    """Shared functionality for the conditional flow-based solvers.

    Holds the parts that are identical across :class:`cellflow.solvers.OTFlowMatching`
    and :class:`cellflow.solvers.GENOT` — the trained flag, the condition-embedding
    read-out, and which train state supplies inference parameters — while each solver
    implements its own training step (``step_fn``) and prediction (``predict``).

    Subclasses call :meth:`__init__` (via ``super().__init__``) for the common
    constructor scaffolding, then set their solver-specific attributes and create
    ``self.vf_state``. A subclass that keeps a separate EMA inference copy overrides
    :attr:`_inference_state`.
    """

    def __init__(
        self,
        vf: Any,
        probability_path: Any,
        time_sampler: Any,
    ) -> None:
        self._is_trained: bool = False
        self.vf = vf
        self.condition_encoder_mode = vf.condition_mode
        self.condition_encoder_regularization = vf.regularization
        self.probability_path = probability_path
        self.time_sampler = time_sampler
        self._predict_fn_cache: dict[Any, Any] = {}

    @property
    def _inference_state(self) -> train_state.TrainState:
        """Train state whose params are used at inference / for condition embeddings.

        Defaults to :attr:`vf_state`; solvers with an EMA inference copy (e.g.
        :class:`cellflow.solvers.OTFlowMatching`) override this to return it.
        """
        return self.vf_state

    def get_condition_embedding(self, condition: dict[str, ArrayLike], return_as_numpy=True) -> ArrayLike:
        """Get learnt embeddings of the conditions.

        Parameters
        ----------
        condition
            Conditions to encode.
        return_as_numpy
            Whether to return the embeddings as numpy arrays.

        Returns
        -------
        Mean and log-variance of encoded conditions.
        """
        cond_mean, cond_logvar = self.vf.apply(
            {"params": self._inference_state.params},
            condition,
            method="get_condition_embedding",
        )
        if return_as_numpy:
            return np.asarray(cond_mean), np.asarray(cond_logvar)
        return cond_mean, cond_logvar

    @property
    def is_trained(self) -> bool:
        """Whether the model is trained."""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        self._is_trained = value
