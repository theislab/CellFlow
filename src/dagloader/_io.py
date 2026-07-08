"""Container-agnostic data access for the loader: obs columns, leaf codes, and rep backings.

Every helper works uniformly over an in-memory ``AnnData`` and an out-of-core annbatch
``DatasetCollection``, and over ``X`` / ``obsm`` / ``layers`` representations — so the loader never
branches on the source kind. No cell matrices are touched for grouping (obs only); cells are read
only when a batch is materialized (by annbatch's own loader over the returned backings).
"""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd

from dagloader._schema import Container

__all__ = ["densify", "key_backings", "leaf_codes", "obs_columns"]


def densify(x) -> np.ndarray:
    """A streamed batch → a dense float32 array (annbatch may yield sparse for a sparse source)."""
    return np.asarray(x.todense() if hasattr(x, "todense") else x, dtype=np.float32)


def key_backings(source: Container, loc: str) -> list:
    """The array(s) backing rep ``loc`` for a source, ready to feed one annbatch ``add_datasets``.

    annbatch's ``add_datasets`` concatenates on the obs axis and needs equal feature dims, so each rep
    gets its own loader over its own array(s). For a ``DatasetCollection`` the per-dataset arrays are
    gathered in order (matching the global row layout).
    """
    if loc == "X":
        return [source.X] if isinstance(source, ad.AnnData) else [g["X"] for g in source]
    field, sub = loc.split("/", 1)  # "obsm/X_pca" | "layers/log1p"
    if isinstance(source, ad.AnnData):
        return [getattr(source, field)[sub]]
    return [g[field][sub] for g in source]  # DatasetCollection: one zarr array per dataset


def leaf_codes(obs: pd.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, list[tuple]]:
    """Per-cell leaf code + the ordered leaf combinations (the grouping over ``cols``)."""
    tuples = [tuple(row) for row in obs[list(cols)].to_numpy()]
    leaves = sorted(set(tuples), key=lambda t: tuple(map(str, t)))
    code_of = {lf: i for i, lf in enumerate(leaves)}
    return np.array([code_of[t] for t in tuples], dtype=np.int64), leaves


def obs_columns(source: Container, cols: Sequence[str]) -> pd.DataFrame:
    """obs columns from either container (AnnData attr vs DatasetCollection reader) — no cell matrices."""
    if isinstance(source, ad.AnnData):
        return source.obs[list(cols)]
    return source.obs(columns=list(cols))  # DatasetCollection
