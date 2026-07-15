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

__all__ = ["key_backings", "leaf_codes", "obs_columns"]


def _readable(x):
    """A rep backing annbatch can read (dense passes through; a sparse zarr group gets wrapped).

    A **sparse** rep in a ``DatasetCollection`` is a zarr *group* (CSR: data/indices/indptr) with no
    ``.shape``, so wrap it as an anndata ``CSRDataset`` (which exposes ``.shape`` + row indexing).
    In-memory ``AnnData`` reps (scipy/numpy) and dense zarr arrays already qualify and pass through.
    """
    if hasattr(x, "shape"):
        return x
    from anndata.io import sparse_dataset

    return sparse_dataset(x)


def key_backings(source: Container, loc: str) -> list:
    """The array(s) backing rep ``loc`` for a source, ready to feed one annbatch ``add_datasets``.

    annbatch's ``add_datasets`` concatenates on the obs axis and needs equal feature dims, so each rep
    gets its own loader over its own array(s). For a ``DatasetCollection`` the per-dataset arrays are
    gathered in order (matching the global row layout); a sparse rep's zarr group is wrapped so it is
    readable (see :func:`_readable`).
    """
    if loc == "X":
        return [source.X] if isinstance(source, ad.AnnData) else [_readable(g["X"]) for g in source]
    field, sub = loc.split("/", 1)  # "obsm/X_pca" | "layers/log1p"
    if isinstance(source, ad.AnnData):
        return [getattr(source, field)[sub]]
    return [_readable(g[field][sub]) for g in source]  # DatasetCollection: one backing per dataset


def leaf_codes(obs: pd.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, list[tuple]]:
    """Per-cell leaf code + the ordered leaf combinations (the grouping over ``cols``)."""
    tuples = [tuple(row) for row in obs[list(cols)].to_numpy()]
    leaves = sorted(set(tuples), key=lambda t: tuple(map(str, t)))
    code_of = {lf: i for i, lf in enumerate(leaves)}
    return np.array([code_of[t] for t in tuples], dtype=np.int64), leaves


def obs_columns(source: Container, cols: Sequence[str]) -> pd.DataFrame:
    """Obs columns from either container (AnnData attr vs DatasetCollection reader) — no cell matrices."""
    if isinstance(source, ad.AnnData):
        return source.obs[list(cols)]
    return source.obs(columns=list(cols))  # DatasetCollection
