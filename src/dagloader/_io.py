"""Container-agnostic data access for the loader: obs columns, leaf codes, and rep backings.

Every helper works uniformly over an in-memory ``AnnData`` and an out-of-core annbatch
``DatasetCollection``, and over ``X`` / ``obsm`` / ``layers`` representations — so the loader never
branches on the source kind. No cell matrices are touched for grouping (obs only); cells are read
only when a batch is materialized (by annbatch's own loader over the returned backings).
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd

from dagloader._schema import Container

__all__ = ["key_backings", "leaf_codes", "load_backed_adata", "materialize_node", "obs_columns", "open_source"]


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
    gets its own loader over its own array(s). For a ``DatasetCollection`` (or a ``list`` of AnnData —
    one backing per adata) the per-dataset arrays are gathered in order (matching the global row
    layout); a sparse rep's zarr group is wrapped so it is readable (see :func:`_readable`).
    """
    if isinstance(source, list):  # list of (backed) AnnData: one backing per adata, in list order
        return [b for a in source for b in key_backings(a, loc)]
    if loc == "X":
        return [source.X] if isinstance(source, ad.AnnData) else [_readable(g["X"]) for g in source]
    field, sub = loc.split("/", 1)  # "obsm/X_pca" | "layers/log1p"
    if isinstance(source, ad.AnnData):
        return [getattr(source, field)[sub]]
    return [_readable(g[field][sub]) for g in source]  # DatasetCollection: one backing per dataset


def _read_rows(source: Container, loc: str, row_idx: np.ndarray) -> np.ndarray:
    """Densely read the global rows ``row_idx`` (ascending) of rep ``loc`` from a source's backings."""
    backings = key_backings(source, loc)  # each has ``.shape`` (sparse groups already wrapped by _readable)
    offs = np.concatenate([[0], np.cumsum([b.shape[0] for b in backings])]).astype(np.int64)
    parts = []
    for d, b in enumerate(backings):
        lb = row_idx[(row_idx >= offs[d]) & (row_idx < offs[d + 1])] - offs[d]  # ascending within this dataset
        if lb.size:
            sub = b.oindex[lb] if hasattr(b, "oindex") else b[lb]  # zarr array: orthogonal; CSRDataset/scipy: []
            dense = getattr(sub, "todense", None)
            parts.append(np.asarray(dense()) if dense is not None else np.asarray(sub))
    return np.concatenate(parts, axis=0) if parts else np.empty((0, int(backings[0].shape[1])), dtype=np.float32)


def materialize_node(source: Container, node) -> ad.AnnData:
    """Materialize a node's selected (positive-weight) cells into an in-memory (dense) ``AnnData``.

    The general "read this node's cells into RAM" op behind :attr:`~dagloader.Node.in_memory`: reads only
    the rows whose leaf has positive weight — for each of the node's reps (``keys``) — and sorts them by
    ``cols`` so ``chunk_size > 1`` still reads contiguous runs. Handles dense and sparse (CSR-group)
    backings alike (via :func:`key_backings`). Cells must fit host RAM (the intended use is a small,
    frequently re-drawn population such as matched controls).
    """
    from dagloader._schema import _weight_vector

    obs = obs_columns(source, node.cols)
    codes, leaves = leaf_codes(obs, node.cols)
    selected = np.flatnonzero(_weight_vector(node.weights, leaves) > 0)  # leaf codes with positive weight
    row_idx = np.flatnonzero(np.isin(codes, selected))  # ascending global rows of the selected leaves
    reps = {key: _read_rows(source, key, row_idx) for key in node.keys}
    sub_obs = obs.iloc[row_idx].reset_index(drop=True)
    order = sub_obs.sort_values(list(node.cols), kind="stable").index.to_numpy()  # contiguous runs for chunk>1
    sub_obs, reps = sub_obs.iloc[order].reset_index(drop=True), {k: v[order] for k, v in reps.items()}
    adata = ad.AnnData(X=reps["X"], obs=sub_obs) if "X" in reps else ad.AnnData(obs=sub_obs)  # X-> n_var inferred
    for key, v in reps.items():
        if key != "X":
            adata.obsm[key.split("/", 1)[1]] = v
    return adata


def leaf_codes(obs: pd.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, list[tuple]]:
    """Per-cell leaf code + the ordered leaf combinations (the grouping over ``cols``)."""
    tuples = [tuple(row) for row in obs[list(cols)].to_numpy()]
    leaves = sorted(set(tuples), key=lambda t: tuple(map(str, t)))
    code_of = {lf: i for i, lf in enumerate(leaves)}
    return np.array([code_of[t] for t in tuples], dtype=np.int64), leaves


def obs_columns(source: Container, cols: Sequence[str]) -> pd.DataFrame:
    """Obs columns from any container (AnnData attr vs DatasetCollection reader vs list) — no cell matrices."""
    if isinstance(source, list):  # list of AnnData: concatenate obs in list order (= key_backings order)
        return pd.concat([obs_columns(a, cols) for a in source], ignore_index=True)
    if isinstance(source, ad.AnnData):
        return source.obs[list(cols)]
    return source.obs(columns=list(cols))  # DatasetCollection


def _backed(x):
    """A zarr rep as a readable backing: dense zarr array passes through; a sparse group is wrapped."""
    import zarr

    return x if isinstance(x, zarr.Array) else ad.io.sparse_dataset(x)


def load_backed_adata(g, *, keys: Sequence[str], cols: Sequence[str] = ()) -> ad.AnnData:
    """Open a zarr adata group as a (backed) AnnData, reading only the reps in ``keys`` and obs ``cols``.

    ``X`` is read as a backed sparse dataset or lazy dense zarr array; ``obsm/<k>`` / ``layers/<k>`` reps
    likewise; ``var`` is reduced to its index and ``obs`` to ``cols`` (all of obs if ``cols`` is empty).
    Only the reps a node actually streams are materialized, so unused representations are never touched.
    """
    var = g["var"]
    obs = ad.io.read_elem(g["obs"])
    kw: dict = {
        "obs": obs[list(cols)] if cols else obs,
        "var": pd.DataFrame(index=pd.Index(ad.io.read_elem(var[var.attrs.get("_index")]))),
    }
    if "X" in keys:
        kw["X"] = _backed(g["X"])
    adata = ad.AnnData(**kw)
    for key in keys:
        if key == "X":
            continue
        field, sub = key.split("/", 1)  # "obsm/X_pca" | "layers/log1p"
        getattr(adata, field)[sub] = _backed(g[field][sub])
    return adata


def open_source(src, *, keys: Sequence[str], cols: Sequence[str] = ()) -> Container:
    """Resolve one :class:`~dagloader.Scheme` source value to a :data:`~dagloader.Container`.

    An in-memory ``AnnData`` or ``DatasetCollection`` passes through unchanged. A single path is opened as
    zarr and auto-detected: an annbatch collection root (``encoding-type`` ``"annbatch-preshuffled"``)
    becomes a ``DatasetCollection``; otherwise it is a single adata read backed via :func:`load_backed_adata`.
    A list of paths becomes a list of backed AnnData (the loader feeds one backing per adata to
    ``add_datasets``). Only the reps in ``keys`` and obs ``cols`` are read (see :func:`load_backed_adata`).
    """
    import zarr
    from annbatch import DatasetCollection

    if isinstance(src, (ad.AnnData, DatasetCollection)):
        return src
    if isinstance(src, (str, os.PathLike)):
        g = zarr.open_group(src, mode="r")
        if g.attrs.get("encoding-type") == "annbatch-preshuffled":
            return DatasetCollection(src, mode="r")
        return load_backed_adata(g, keys=keys, cols=cols)
    # list/sequence of adata zarr paths (or already-open AnnData) → list of backed AnnData
    return [a if isinstance(a, ad.AnnData) else load_backed_adata(zarr.open_group(a, mode="r"), keys=keys, cols=cols) for a in src]
