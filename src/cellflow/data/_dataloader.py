from typing import TYPE_CHECKING

import numpy as np

from cellflow._types import ArrayLike

if TYPE_CHECKING:
    from dagloader import DAGLoader

__all__ = [
    "DAGLoaderAdapter",
]


def _densify(x: ArrayLike) -> ArrayLike:
    """Densify a possibly-sparse streamed cell batch, staying on the array's native backend.

    ``dagloader`` yields native jax arrays (``to="jax"``): dense reps pass straight through, and a
    sparse rep arrives as a ``jax.experimental.sparse`` CSR — densified here via ``.todense()`` (still
    on-device). No numpy cast and no host round-trip, so a GPU-resident batch reaches the solver as-is.
    """
    todense = getattr(x, "todense", None)  # (jax/scipy) sparse → dense; dense arrays lack this
    return todense() if todense is not None else x


class DAGLoaderAdapter:
    """Adapt a ``dagloader.DAGLoader`` stream to the sampler ``sample()`` batch contract.

    Renames the loader's ``{"target", "source", "condition"}`` batch to the model's
    ``{"src_cell_data", "tgt_cell_data", "condition"}`` and densifies any sparse cell rep, keeping the
    arrays on their native (jax) backend, so the in-memory and streaming paths reach the solver
    identically. Used for training, and (re)used for the streamed validation and prediction paths.
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
