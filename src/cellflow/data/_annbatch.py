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
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np

from cellflow.data._condition import _key_layout, build_condition_data, enumerate_perturbations
from cellflow.data._datamanager import DataManager

if TYPE_CHECKING:
    from cellflow._types import ArrayLike
    from dagloader import Container, Scheme, Weights

Leaf = tuple[object, ...]  # a scheme leaf: one value per grouping column

__all__ = [
    "AnnbatchTraining",
    "assert_source_chunkable",
    "build_annbatch_training",
    "sample_rep_to_key",
]


def assert_source_chunkable(
    source: Container, cols: Sequence[str], chunk_size: int, weights: Weights | None = None
) -> None:
    """Raise unless ``source`` satisfies annbatch's run-length rule for ``chunk_size`` (reads ``obs`` only).

    With ``chunk_size > 1`` annbatch reads contiguous slices, so every contiguous run of each category
    (a ``cols`` combination) must be at least ``chunk_size`` cells; a category may span several runs.
    On a shorter run, raises pointing at ``DatasetCollection.add_adatas(groupby=...)``. In-memory sources
    are grouped automatically by :func:`build_annbatch_training`, so this only bites out-of-core inputs.

    ``weights`` (a node's ``{combo: weight}``) restricts the check to positive-weight leaves: annbatch's
    ``ClassSampler`` never reads a zero-weight leaf, so it exempts them from the run-length rule. Passing
    the root node's weights is what lets ``min_cells_per_condition`` unblock ``chunk_size > 1`` — a
    sub-threshold condition is zero-weighted, so its short run no longer blocks chunked reads. With
    ``weights=None`` every run is checked (the strict, weight-agnostic behavior).
    """
    from dagloader._io import leaf_codes, obs_columns

    codes, leaves = leaf_codes(obs_columns(source, list(cols)), list(cols))
    if len(codes) == 0:
        return
    run_starts = np.concatenate([[0], np.flatnonzero(np.diff(codes) != 0) + 1])
    run_lengths = np.diff(np.concatenate([run_starts, [len(codes)]]))
    if weights is not None:  # exempt zero-weight leaves (mirrors ClassSampler): check only their runs
        positive = [i for i, lf in enumerate(leaves) if weights.get(tuple(lf), 0.0) > 0]
        run_lengths = run_lengths[np.isin(codes[run_starts], positive)]
        if len(run_lengths) == 0:
            return
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
    condition_fn: Callable[[Leaf], dict[str, np.ndarray]]
    condition_data: dict[str, np.ndarray]
    data_manager: DataManager
    data_dim: int
    max_combination_length: int


def build_annbatch_training(
    source: Container,
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
    control_in_memory: bool = False,
    min_cells_per_condition: int = 0,
) -> AnnbatchTraining:
    """Assemble the :class:`dagloader.Scheme` + ``condition_fn`` for the streaming path (obs only).

    ``source`` is an out-of-core :class:`annbatch.DatasetCollection` or an in-memory ``AnnData``.
    ``rep_dict`` holds the covariate embedding tables (as ``adata.uns`` would); pass :obj:`None` when the
    primary covariate is categorical (one-hot).

    ``control_in_memory`` tells dagloader to materialize the control (child) node into RAM (sets
    :attr:`~dagloader.Node.in_memory`; dagloader owns the read via :func:`~dagloader._io.materialize_node`),
    so the matched control is served from memory while the perturbed target keeps streaming out of core — a
    large dataloader speedup, since controls are re-drawn every batch. Only enable it when the controls fit
    in host RAM (the small population by design).

    ``min_cells_per_condition`` zero-weights (drops from sampling) any perturbed condition with fewer than
    this many *total* cells — a scientific filter on untrainable tiny conditions, and the lever that unblocks
    ``chunk_size > 1``: a zero-weight leaf is exempt from annbatch's run-length rule, so a condition with a
    sub-``chunk_size`` run no longer blocks chunked reads. The default ``0`` drops nothing (weights identical
    to ``uniform``). Caveat: this counts *total* cells per condition, so it unblocks ``chunk_size > 1`` only
    when the kept conditions' per-plate *runs* are also ``>= chunk_size``; a big condition with a rare
    sub-``chunk_size`` run in one plate would still need a per-run guard (a generic ``min_rows_per_leaf`` in
    dagloader — out of scope here).
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

    # Root (perturbed) leaf weights: uniform over the perturbed combos, except conditions with fewer than
    # `min_cells_per_condition` total cells are zero-weighted (dropped from sampling). Counts come from the
    # full `obs` (all cells) via a groupby — cheap on the categorical grouping cols; keyed by string-tuple so
    # the lookup is robust to `.to_numpy()` dtype quirks. The default 0 keeps every leaf, so `pert_weights`
    # equals `uniform(pert)` and the whole scheme is byte-identical to before.
    if min_cells_per_condition > 0:
        cnt = obs.groupby(list(cols), observed=True).size().reset_index(name="_n").to_numpy()
        count_of = {tuple(map(str, row[:-1])): int(row[-1]) for row in cnt}  # {str(cols-combo): n_cells}
        pert_weights = {
            leaf: (1.0 if count_of.get(tuple(map(str, leaf)), 0) >= min_cells_per_condition else 0.0) for leaf in pert
        }
        if pert and not any(pert_weights.values()):
            largest = max(count_of.get(tuple(map(str, lf)), 0) for lf in pert)
            raise ValueError(
                f"min_cells_per_condition={min_cells_per_condition} dropped every perturbed condition "
                f"(the largest has {largest} cells); lower the threshold."
            )
    else:
        pert_weights = uniform(pert)

    # `control_in_memory` just tells dagloader to materialize the control (child) node into RAM — the
    # perturbed target keeps streaming out of core. dagloader owns the materialization (Node.in_memory);
    # the bind still matches control↔target by the context columns.
    scheme = Scheme(
        sources={"data": source},
        nodes={
            "pert": Node("data", cols, key, pert_weights),
            "ctrl": Node("data", cols, key, uniform(ctrl), in_memory=control_in_memory),
        },
        root="pert",
        binds=(Bind("pert", "ctrl", common=context),),
        seed=seed,
    )

    data_dim = int(key_backings(source, key)[0].shape[1])  # key_backings wraps sparse groups → has .shape
    return AnnbatchTraining(
        scheme=scheme,
        condition_fn=condition_fn,
        condition_data=condition_data,
        data_manager=dm,
        data_dim=data_dim,
        max_combination_length=dm.max_combination_length,
    )
