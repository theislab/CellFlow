from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from cellflow._types import ArrayLike

if TYPE_CHECKING:
    from dagloader import DAGEvalLoader, DAGLoader

__all__ = [
    "DAGEvalAdapter",
    "DAGTrainAdapter",
]


def _densify(x: ArrayLike) -> ArrayLike:
    """Densify a possibly-sparse streamed cell batch, staying on the array's native backend.

    ``dagloader`` yields native jax arrays (``to="jax"``): dense reps pass straight through, and a
    sparse rep arrives as a ``jax.experimental.sparse`` CSR — densified here via ``.todense()`` (still
    on-device). No numpy cast and no host round-trip, so a GPU-resident batch reaches the solver as-is.
    """
    todense = getattr(x, "todense", None)  # (jax/scipy) sparse → dense; dense arrays lack this
    return todense() if todense is not None else x


class DAGTrainAdapter:
    """Adapt a ``dagloader.DAGLoader`` stream to the trainer's ``sample(rng)`` batch contract.

    Renames the loader's ``{"target", "source", "condition"}`` batch to the model's
    ``{"src_cell_data", "tgt_cell_data", "condition"}`` and densifies any sparse cell rep, keeping the
    arrays on their native (jax) backend, so the in-memory and streaming paths reach the solver
    identically. This is the *training* adapter; validation uses :class:`DAGEvalAdapter`.
    """

    def __init__(self, loader: "DAGLoader"):
        self._loader = loader
        self._iter = iter(loader)

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, ArrayLike | dict[str, ArrayLike]]:
        """Return the next streamed batch as a trainer batch dict.

        ``rng`` is unused — the ``DAGLoader`` owns its own reproducible RNG.
        """
        batch = next(self._iter)
        out: dict[str, ArrayLike | dict[str, ArrayLike]] = {
            "src_cell_data": _densify(batch["source"]),
            "tgt_cell_data": _densify(batch["target"]),
        }
        if "condition" in batch:
            out["condition"] = batch["condition"]
        return out


class DAGEvalAdapter:
    """Adapt a ``dagloader.DAGEvalLoader`` to the trainer's validation ``sample(mode)`` contract.

    Yields per-condition ``{"source", "condition", "target"}`` dicts (keyed by the perturbed leaf), as the
    trainer's validation step expects. The control-rooted ``DAGEvalLoader`` reads each control population's
    cells in full for the source and samples a matched perturbed batch for the target — via annbatch, no
    boolean masking. ``n_conditions_*`` sets how many (control-population, drug) batches to draw per mode.
    This mirrors the in-memory :class:`~cellflow.data._legacy.ValidationSampler` contract for the streaming
    path, so the trainer's validation step is identical either way.

    Parameters
    ----------
    eval_loader
        A :class:`dagloader.DAGEvalLoader` built over the validation source.
    n_conditions_on_log_iteration, n_conditions_on_train_end
        How many conditions to draw per mode; :obj:`None` visits each control population once.
    """

    def __init__(
        self,
        eval_loader: "DAGEvalLoader",
        *,
        n_conditions_on_log_iteration: int | None = None,
        n_conditions_on_train_end: int | None = None,
    ) -> None:
        self._eval = eval_loader
        self.n_conditions_on_log_iteration = n_conditions_on_log_iteration
        self.n_conditions_on_train_end = n_conditions_on_train_end

    def sample(self, mode: Literal["on_log_iteration", "on_train_end"]) -> dict[str, dict[Any, Any]]:
        """Sample a validation batch: per-condition source/condition/target dicts (keyed by perturbed leaf)."""
        # Densify exactly as the training adapter does: with the GPU read path (cupy) annbatch yields a
        # jax *sparse* CSR, which the solver's `predict` can't index — so eval source/target must be dense.
        n = self.n_conditions_on_log_iteration if mode == "on_log_iteration" else self.n_conditions_on_train_end
        source: dict[Any, Any] = {}
        condition: dict[Any, Any] = {}
        target: dict[Any, Any] = {}
        for out in self._eval.iter_conditions(n_conditions=n):
            key = tuple(out["leaf"])
            source[key] = _densify(out["source"])
            condition[key] = out["condition"]
            target[key] = _densify(out["target"])
        return {"source": source, "condition": condition, "target": target}
