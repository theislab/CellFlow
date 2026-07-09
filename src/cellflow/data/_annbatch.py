"""Build the annbatch/dagloader streaming training path from a CellFlow covariate spec.

Turns the ``prepare_data`` covariate arguments into a :class:`dagloader.Scheme` (perturbed root,
matched-control child) and a ``condition_fn`` mapping each sampled leaf to its condition embedding. The
embeddings reuse the in-memory machinery — a cell-free ``AnnData`` shell (``obs`` + ``uns``) drives a
:class:`~cellflow.data._datamanager.DataManager` and
:func:`~cellflow.data._condition.build_condition_data` — so they match the in-memory path exactly. Only
``obs`` (and the embedding tables) are read here; cells are streamed later by ``dagloader``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np

from cellflow.data._condition import _key_layout, build_condition_data, enumerate_perturbations
from cellflow.data._datamanager import DataManager

if TYPE_CHECKING:
    from dagloader import Scheme

__all__ = ["AnnbatchTraining", "assert_source_chunkable", "build_annbatch_training", "sample_rep_to_key"]


def assert_source_chunkable(source: Any, cols: Sequence[str], chunk_size: int) -> None:
    """Raise unless ``source`` satisfies annbatch's run-length rule for ``chunk_size`` (reads ``obs`` only).

    With ``chunk_size > 1`` annbatch reads contiguous slices, so every contiguous run of each category
    (a ``cols`` combination) must be at least ``chunk_size`` cells; a category may span several runs.
    On a shorter run, raises pointing at ``DatasetCollection.add_adatas(groupby=...)``. In-memory sources
    are grouped automatically by :func:`build_annbatch_training`, so this only bites out-of-core inputs.
    """
    import numpy as np

    from dagloader._io import leaf_codes, obs_columns

    codes, _ = leaf_codes(obs_columns(source, list(cols)), list(cols))
    if len(codes) == 0:
        return
    run_starts = np.concatenate([[0], np.flatnonzero(np.diff(codes) != 0) + 1])
    run_lengths = np.diff(np.concatenate([run_starts, [len(codes)]]))
    shortest = int(run_lengths.min())
    if shortest < chunk_size:
        raise ValueError(
            f"chunk_size={chunk_size} requires every contiguous run of each category (a {list(cols)} "
            f"combination) to be at least chunk_size cells, but the source has a run of only {shortest}. "
            f"Group the source so each category forms long runs — e.g. create the DatasetCollection with "
            f"`add_adatas(..., groupby={list(cols)})` — or use chunk_size=1. (A category may span several "
            f"runs; only each run's length matters.)"
        )


def sample_rep_to_key(sample_rep: str) -> str:
    """CellFlow ``sample_rep`` → dagloader representation key (``"X"`` or ``"obsm/<key>"``)."""
    return "X" if sample_rep == "X" else f"obsm/{sample_rep}"


@dataclass(frozen=True)
class AnnbatchTraining:
    """Everything the model needs to stream-train, assembled from a covariate spec (cells untouched)."""

    scheme: Scheme
    condition_fn: Callable[[tuple], dict[str, np.ndarray]]
    condition_data: dict[str, np.ndarray]
    data_manager: DataManager
    data_dim: int
    max_combination_length: int


def build_annbatch_training(
    source: Any,
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
    rep_dict: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> AnnbatchTraining:
    """Assemble the :class:`dagloader.Scheme` + ``condition_fn`` for the streaming path (obs only).

    ``source`` is an out-of-core :class:`annbatch.DatasetCollection` or an in-memory ``AnnData``.
    ``rep_dict`` holds the covariate embedding tables (as ``adata.uns`` would); pass :obj:`None` when the
    primary covariate is categorical (one-hot).
    """
    from dagloader import Bind, Node, Scheme, uniform
    from dagloader._io import key_backings, obs_columns

    context = tuple(split_covariates or ())
    pert_cols = tuple(c for grp in perturbation_covariates.values() for c in grp)
    samp_cols = tuple(sample_covariates or ())
    cols = tuple(dict.fromkeys((*context, *pert_cols, *samp_cols)))  # grouping cols (deduped, ordered)
    key = sample_rep_to_key(sample_rep)

    obs = obs_columns(source, [*cols, control_key])

    # In-memory source: stable-sort by the grouping columns so `chunk_size > 1` reads contiguous slices
    # (cheap, and cell order is irrelevant). Out-of-core sources aren't reordered (expensive zarr re-sort)
    # — they must be built grouped; `assert_source_chunkable` enforces that.
    if isinstance(source, ad.AnnData):
        order = obs[list(cols)].reset_index(drop=True).sort_values(list(cols), kind="stable").index.to_numpy()
        source = source[order].copy()
        obs = obs_columns(source, [*cols, control_key])

    # DataManager as a covariate-encoder factory: reads only obs + uns, so a cell-free shell suffices.
    # `sample_rep` is stored for validation's `_get_cell_data` (verification is type-only, so it's safe here).
    shell = ad.AnnData(obs=obs.copy())
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
        obs,
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
        obs,
        control_key=control_key,
        perturb_covar_keys=dm._perturb_covar_keys,
        split_covariates=list(context),
        sample_covariates=list(samp_cols),
    )
    _, tuple_keys = _key_layout(dm._perturb_covar_keys, list(context), list(samp_cols))
    cov_to_idx = {tuple(map(str, cov)): i for i, cov in idx_to_cov.items()}
    reorder = [cols.index(c) for c in tuple_keys]

    def condition_fn(leaf: tuple) -> dict[str, np.ndarray]:
        idx = cov_to_idx[tuple(str(leaf[i]) for i in reorder)]
        return {group: condition_data[group][[idx]] for group in condition_data}

    # The Scheme: root = perturbed combos, child = matched-control combos (bound on the context columns).
    ctrl_flag = obs[control_key].to_numpy().astype(bool)
    pert = [tuple(r) for r in obs.loc[~ctrl_flag, list(cols)].drop_duplicates().to_numpy()]
    ctrl = [tuple(r) for r in obs.loc[ctrl_flag, list(cols)].drop_duplicates().to_numpy()]
    scheme = Scheme(
        sources={"data": source},
        nodes={
            "pert": Node("data", cols, key, uniform(pert)),
            "ctrl": Node("data", cols, key, uniform(ctrl)),
        },
        root="pert",
        binds=(Bind("pert", "ctrl", common=context),),
        seed=seed,
    )

    data_dim = int(key_backings(source, key)[0].shape[1])
    return AnnbatchTraining(
        scheme=scheme,
        condition_fn=condition_fn,
        condition_data=condition_data,
        data_manager=dm,
        data_dim=data_dim,
        max_combination_length=dm.max_combination_length,
    )
