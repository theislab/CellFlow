"""Build the annbatch/dagloader streaming training path from a CellFlow covariate spec.

Turns the ``prepare_data`` covariate arguments into a :class:`dagloader.Scheme` (perturbed root,
matched-control child) and a ``condition_fn`` mapping each sampled leaf to its condition embedding. The
embeddings reuse the in-memory machinery — a cell-free ``AnnData`` shell (``obs`` + ``uns``) drives a
:class:`~cellflow.data._datamanager.DataManager` and
:func:`~cellflow.data._condition.build_condition_data` — so they match the in-memory path exactly. Only
``obs`` (and the embedding tables) are read here; cells are streamed later by ``dagloader``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np

from cellflow._logging import logger
from cellflow.data._condition import _key_layout, build_condition_data, enumerate_perturbations
from cellflow.data._datamanager import DataManager

if TYPE_CHECKING:
    from cellflow._types import ArrayLike
    from dagloader import Container, Scheme

Leaf = tuple[object, ...]  # a scheme leaf: one value per grouping column

__all__ = [
    "AnnbatchTraining",
    "build_annbatch_training",
    "sample_rep_to_key",
]


def sample_rep_to_key(sample_rep: str) -> str:
    """CellFlow ``sample_rep`` → dagloader representation key (``"X"`` or ``"obsm/<key>"``)."""
    return "X" if sample_rep == "X" else f"obsm/{sample_rep}"


@dataclass(frozen=True)
class AnnbatchTraining:
    """Everything the model needs to stream-train, assembled from a covariate spec (cells untouched)."""

    scheme: Scheme
    condition_fn: Callable[[Leaf], dict[str, np.ndarray]]
    condition_data: dict[str, np.ndarray]
    data_manager: DataManager
    data_dim: int
    max_combination_length: int


def build_annbatch_training(
    data: Container | str | os.PathLike | Sequence[str | os.PathLike | ad.AnnData],
    *,
    sample_rep: str,
    control_key: str,
    perturbation_covariates: Mapping[str, Sequence[str]],
    perturbation_covariate_reps: Mapping[str, str] | None = None,
    sample_covariates: Sequence[str] | None = None,
    sample_covariate_reps: Mapping[str, str] | None = None,
    split_covariates: Sequence[str] | None = None,
    max_combination_length: int | None = None,
    null_value: float = 0.0,
    rep_dict: Mapping[str, Mapping[str, ArrayLike]] | None = None,
    seed: int = 0,
    control_in_memory: bool = True,
    min_cells_per_condition: int = 0,
    chunk_size: int = 1,
) -> AnnbatchTraining:
    """Assemble the :class:`dagloader.Scheme` + ``condition_fn`` for the streaming path (obs only).

    ``data`` is an out-of-core :class:`annbatch.DatasetCollection`, an in-memory ``AnnData``, an adata
    zarr path, or a list of adata zarr paths (paths are resolved via :func:`~dagloader._io.open_source`).
    ``rep_dict`` holds the covariate embedding tables (as ``adata.uns`` would); pass :obj:`None` when the
    primary covariate is categorical (one-hot).

    ``control_in_memory`` tells dagloader to materialize the control (child) node into RAM (sets
    :attr:`~dagloader.Node.in_memory`; dagloader owns the read via :func:`~dagloader._io.materialize_node`),
    so the matched control is served from memory while the perturbed target keeps streaming out of core — a
    large dataloader speedup, since controls are re-drawn every batch. Only enable it when the controls fit
    in host RAM (the small population by design).

    Two perturbed-only weight filters shape the root (target) node — controls are never filtered (with
    ``control_in_memory`` the control node is materialized+sorted in RAM; otherwise its run-length is
    annbatch's own concern, not ours):

    ``min_cells_per_condition`` zero-weights any perturbed condition with fewer than this many *total*
    cells — a scientific filter on untrainable tiny conditions. Default ``0`` drops nothing.

    ``chunk_size`` (the streamed ``SamplerConfig.chunk_size``) drives the run-length filter. With
    ``chunk_size > 1`` annbatch reads contiguous ``chunk_size``-long slices, so every run of a positive-weight
    class must be ``>= chunk_size``. Any perturbed condition whose *smallest* contiguous run is shorter is
    zero-weighted here (and thereby excluded from every split), so the rest stream chunked without annbatch
    raising. This is the per-run guard a *total* filter can't provide — a big condition with a rare
    sub-``chunk_size`` sliver in one plate is dropped. Dropped counts are logged. Default ``1`` filters
    nothing; with both filters inactive (``min_cells_per_condition=0`` and ``chunk_size=1``) the root weights
    are ``uniform`` — byte-identical to before.
    """
    from dagloader import Bind, Node, Scheme, uniform
    from dagloader._io import key_backings, obs_columns, open_source

    context = tuple(split_covariates or ())
    pert_cols = tuple(c for grp in perturbation_covariates.values() for c in grp)
    samp_cols = tuple(sample_covariates or ())
    cols = tuple(dict.fromkeys((*context, *pert_cols, *samp_cols)))  # grouping cols (deduped, ordered)
    key = sample_rep_to_key(sample_rep)

    # Accept a zarr path / list of adata zarr paths too: resolve to a Container (reads only `key` + the
    # grouping obs). Path-backed data is out-of-core, so — like a DatasetCollection — it is never
    # reordered here; only a user-supplied in-memory AnnData is stable-sorted below.
    from_path = isinstance(data, str | os.PathLike | list | tuple)
    if from_path:
        data = open_source(data, keys=[key], cols=[*cols, control_key])

    obs = obs_columns(data, [*cols, control_key])

    # In-memory data: stable-sort by the grouping columns so `chunk_size > 1` reads contiguous slices
    # (cheap, and cell order is irrelevant). Out-of-core data isn't reordered (expensive zarr re-sort) —
    # it must be built grouped; the run-length filter below drops short-run perturbed conditions and
    # annbatch validates the rest (and the controls) when it builds its samplers.
    if isinstance(data, ad.AnnData) and not from_path:
        order = obs[list(cols)].reset_index(drop=True).sort_values(list(cols), kind="stable").index.to_numpy()
        data = data[order].copy()
        obs = obs_columns(data, [*cols, control_key])

    # The encoder and the scheme leaves depend only on the UNIQUE (grouping-cols, control) combinations —
    # a few ×10^4 rows — not on the ~10^8 cells. So deduplicate ONCE here and drive the whole encoder
    # (shell + DataManager + build_condition_data + enumerate_perturbations) and the pert/ctrl leaf lists
    # off that tiny frame; feeding the full obs made every step O(n_cells) (a ~10-min prepare on Tahoe).
    # Cast string grouping cols to `category` first so this single full-obs dedup hashes small integer
    # codes, not raw strings. Parity-safe: `enumerate_perturbations` casts string cols the same way, so
    # the leaf order is unchanged; only *object* (string) cols are cast, leaving numeric/bool covariates
    # numeric (casting them would flip DataManager's numeric-vs-categorical detection) and preserving
    # already-categorical cols' category order.
    to_categorical = {c: "category" for c in cols if obs[c].dtype == object}
    if to_categorical:
        obs = obs.astype(to_categorical)
    uniq = obs[[*cols, control_key]].drop_duplicates().reset_index(drop=True)

    # DataManager as a covariate-encoder factory: reads only obs + uns, so a cell-free, deduplicated shell
    # suffices — it reads the unique category values (not per-cell counts), so the encoder is identical.
    # `sample_rep` is stored for validation's `_get_cell_data` (verification is type-only, so it's safe here).
    shell = ad.AnnData(obs=uniq.copy())
    shell.uns = dict(rep_dict or {})
    dm = DataManager(
        shell,
        sample_rep=sample_rep,
        control_key=control_key,
        perturbation_covariates=dict(perturbation_covariates),
        perturbation_covariate_reps=dict(perturbation_covariate_reps) if perturbation_covariate_reps else None,
        sample_covariates=list(samp_cols),
        sample_covariate_reps=dict(sample_covariate_reps) if sample_covariate_reps else None,
        split_covariates=list(context),
        max_combination_length=max_combination_length,
        null_value=null_value,
    )

    # Per-condition embeddings — the shared helper → identical to the in-memory path (parity-tested).
    condition_data = build_condition_data(
        uniq,
        shell.uns,
        control_key=control_key,
        perturb_covar_keys=dm._perturb_covar_keys,
        split_covariates=list(context),
        sample_covariates=list(samp_cols),
        perturbation_covariates=dict(perturbation_covariates),
        covariate_reps=dm._covariate_reps,
        covar_to_idx=dm.covar_to_idx,
        is_categorical=dm.is_categorical,
        primary_one_hot_encoder=dm.primary_one_hot_encoder,
        primary_group=dm.primary_group,
        linked_perturb_covars=dm.linked_perturb_covars,
        max_combination_length=dm.max_combination_length,
        null_value=null_value,
    )

    # leaf → perturbation index → embedding. `enumerate_perturbations` lays tuples out in `tuple_keys`
    # order (differs from `cols`), so re-project the leaf; string-normalize both sides so dtype quirks
    # (categorical / numpy scalars) don't break the match.
    idx_to_cov = enumerate_perturbations(
        uniq,
        control_key=control_key,
        perturb_covar_keys=dm._perturb_covar_keys,
        split_covariates=list(context),
        sample_covariates=list(samp_cols),
    )
    _, tuple_keys = _key_layout(dm._perturb_covar_keys, list(context), list(samp_cols))
    cov_to_idx = {tuple(map(str, cov)): i for i, cov in idx_to_cov.items()}
    reorder = [cols.index(c) for c in tuple_keys]

    def condition_fn(leaf: Leaf) -> dict[str, np.ndarray]:
        idx = cov_to_idx[tuple(str(leaf[i]) for i in reorder)]
        return {group: condition_data[group][[idx]] for group in condition_data}

    # The Scheme: root = perturbed combos, child = matched-control combos (bound on the context columns).
    # Built off the deduplicated frame — identical set of leaves to the full obs (order is irrelevant:
    # `uniform` builds a dict and the loader resolves weights per string-sorted leaf).
    ctrl_flag = uniq[control_key].to_numpy().astype(bool)
    pert = [tuple(r) for r in uniq.loc[~ctrl_flag, list(cols)].drop_duplicates().to_numpy()]
    ctrl = [tuple(r) for r in uniq.loc[ctrl_flag, list(cols)].drop_duplicates().to_numpy()]

    # Root (perturbed) leaf weights: `uniform(pert)` minus two perturbed-only filters (controls keep
    # `uniform(ctrl)` — see the docstring). Both derive from ONE pass over the full `obs` (physical order)
    # via `leaf_codes`: per-leaf total cells (bincount) and per-leaf smallest contiguous run (run-length
    # min). Keyed by string-tuple so lookups survive `.to_numpy()` dtype quirks. When both filters are
    # inactive (min_cells_per_condition=0 and chunk_size<=1) we skip the pass and use `uniform(pert)` —
    # byte-identical to before.
    if min_cells_per_condition > 0 or chunk_size > 1:
        from dagloader._io import leaf_codes

        codes, leaves = leaf_codes(obs, list(cols))
        total = np.bincount(codes, minlength=len(leaves))
        run_starts = np.concatenate([[0], np.flatnonzero(np.diff(codes) != 0) + 1])
        run_len = np.diff(np.concatenate([run_starts, [len(codes)]]))
        min_run = np.full(len(leaves), len(codes) + 1, dtype=np.int64)
        np.minimum.at(min_run, codes[run_starts], run_len)  # smallest run per leaf
        stat = {tuple(map(str, lf)): (int(total[i]), int(min_run[i])) for i, lf in enumerate(leaves)}

        def _keep(leaf: Leaf) -> bool:  # perturbed-only: total-cells filter AND per-run (chunk) filter
            n_total, shortest = stat.get(tuple(map(str, leaf)), (0, 0))
            return n_total >= min_cells_per_condition and (chunk_size <= 1 or shortest >= chunk_size)

        pert_weights = {leaf: (1.0 if _keep(leaf) else 0.0) for leaf in pert}
        n_kept = sum(w > 0 for w in pert_weights.values())
        if pert and n_kept == 0:
            largest = max((stat.get(tuple(map(str, lf)), (0, 0))[0] for lf in pert), default=0)
            longest = max((stat.get(tuple(map(str, lf)), (0, 0))[1] for lf in pert), default=0)
            raise ValueError(
                f"dropped every perturbed condition: none has >= min_cells_per_condition="
                f"{min_cells_per_condition} total cells (largest {largest}) and a contiguous run >= "
                f"chunk_size={chunk_size} (longest run {longest}). Lower the thresholds, use chunk_size=1, or "
                f"group the data so each condition forms runs >= chunk_size (e.g. `add_adatas(groupby=...)`)."
            )
        if n_kept < len(pert):
            dropped = sum(stat.get(tuple(map(str, lf)), (0, 0))[0] for lf, w in pert_weights.items() if w == 0)
            allc = sum(stat.get(tuple(map(str, lf)), (0, 0))[0] for lf in pert)
            logger.info(
                f"annbatch streaming: dropped {len(pert) - n_kept}/{len(pert)} perturbed conditions "
                f"({dropped:,}/{allc:,} cells = {dropped / max(allc, 1) * 100:.1f}%) — below "
                f"min_cells_per_condition={min_cells_per_condition} or with a contiguous run < "
                f"chunk_size={chunk_size}. Controls are unaffected."
            )
    else:
        pert_weights = uniform(pert)

    # `control_in_memory` just tells dagloader to materialize the control (child) node into RAM — the
    # perturbed target keeps streaming out of core. dagloader owns the materialization (Node.in_memory);
    # the bind still matches control↔target by the context columns.
    scheme = Scheme(
        sources={"data": data},
        nodes={
            "pert": Node("data", cols, key, pert_weights),
            "ctrl": Node("data", cols, key, uniform(ctrl), in_memory=control_in_memory),
        },
        root="pert",
        binds=(Bind("pert", "ctrl", common=context),),
        seed=seed,
    )

    data_dim = int(key_backings(data, key)[0].shape[1])  # key_backings wraps sparse groups → has .shape
    return AnnbatchTraining(
        scheme=scheme,
        condition_fn=condition_fn,
        condition_data=condition_data,
        data_manager=dm,
        data_dim=data_dim,
        max_combination_length=dm.max_combination_length,
    )
