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
from typing import TYPE_CHECKING, Any, Literal

import anndata as ad
import numpy as np

from cellflow.data._condition import _key_layout, build_condition_data, enumerate_perturbations
from cellflow.data._datamanager import DataManager

if TYPE_CHECKING:
    from cellflow._types import ArrayLike
    from dagloader import Container, DAGEvalLoader, Scheme

Leaf = tuple[object, ...]  # a scheme leaf: one value per grouping column

__all__ = [
    "AnnbatchTraining",
    "AnnbatchValidationSampler",
    "assert_source_chunkable",
    "build_annbatch_training",
    "sample_rep_to_key",
]


class AnnbatchValidationSampler:
    """Validation sampler over a :class:`dagloader.DAGEvalLoader` — the trainer's ``sample(mode)`` contract.

    Yields per-condition ``{"source", "condition", "target"}`` dicts (keyed by the perturbed leaf), as the
    trainer expects. The control-rooted ``DAGEvalLoader`` reads each control population's cells in full for
    the source and samples a matched perturbed batch for the target — via annbatch, no boolean masking.
    ``n_conditions_*`` sets how many (control-population, drug) batches to draw per mode.

    Parameters
    ----------
    eval_loader
        A :class:`dagloader.DAGEvalLoader` built over the validation source.
    n_conditions_on_log_iteration, n_conditions_on_train_end
        How many conditions to draw per mode; :obj:`None` visits each control population once.
    """

    def __init__(
        self,
        eval_loader: DAGEvalLoader,
        *,
        n_conditions_on_log_iteration: int | None = None,
        n_conditions_on_train_end: int | None = None,
    ) -> None:
        self._eval = eval_loader
        self.n_conditions_on_log_iteration = n_conditions_on_log_iteration
        self.n_conditions_on_train_end = n_conditions_on_train_end

    def sample(self, mode: Literal["on_log_iteration", "on_train_end"]) -> dict[str, dict[Any, Any]]:
        """Sample a validation batch: per-condition source/condition/target dicts (keyed by perturbed leaf)."""
        # Densify the streamed cell reps exactly as the training adapter does: with the GPU read path
        # (cupy) annbatch yields a jax *sparse* CSR, which the solver's `predict` can't index — so the
        # eval source/target must be densified here too, or validation crashes on `jnp.asarray(sparse)`.
        from cellflow.data._dataloader import _densify

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


def assert_source_chunkable(source: Container, cols: Sequence[str], chunk_size: int) -> None:
    """Raise unless ``source`` satisfies annbatch's run-length rule for ``chunk_size`` (reads ``obs`` only).

    With ``chunk_size > 1`` annbatch reads contiguous slices, so every contiguous run of each category
    (a ``cols`` combination) must be at least ``chunk_size`` cells; a category may span several runs.
    On a shorter run, raises pointing at ``DatasetCollection.add_adatas(groupby=...)``. In-memory sources
    are grouped automatically by :func:`build_annbatch_training`, so this only bites out-of-core inputs.
    """
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
    condition_fn: Callable[[Leaf], dict[str, np.ndarray]]
    condition_data: dict[str, np.ndarray]
    data_manager: DataManager
    data_dim: int
    max_combination_length: int


def _backing_nrows(b) -> int:
    """Obs count of a rep backing — a dense array or an anndata sparse zarr group (shape lives in attrs)."""
    return int(b.shape[0]) if hasattr(b, "shape") else int(tuple(b.attrs["shape"])[0])


def _backing_ncols(b) -> int:
    """Feature dim of a rep backing (dense array or anndata sparse zarr group)."""
    return int(b.shape[1]) if hasattr(b, "shape") else int(tuple(b.attrs["shape"])[1])


def _materialize_controls(source: Container, obs, cols: Sequence[str], control_key: str, key: str) -> ad.AnnData:
    """Read the control cells' rep into an in-memory (dense) ``AnnData`` — a second Scheme source in RAM.

    Controls are the small population re-drawn for *every* target batch, so re-streaming them from disk
    each step is the dataloader bottleneck; pinning them in memory removes it. Reads only control rows
    (per-dataset gather over the rep backings), densifies, and sorts by ``cols`` so ``chunk_size > 1``
    still reads contiguous runs. Requires the controls to fit in host RAM (they are small by design).
    """
    from dagloader._io import key_backings

    # a rep backing is a dense array (ndarray / zarr array / scipy sparse) OR an anndata sparse zarr
    # *group* (CSR: data/indices/indptr sub-arrays) — the latter has no `.shape` and needs a sparse reader.
    def _rows(b, loc: np.ndarray) -> np.ndarray:
        if hasattr(b, "shape"):
            sub = b.oindex[loc] if hasattr(b, "oindex") else b[loc]
        else:  # sparse zarr group → anndata sparse dataset supports efficient row fancy-indexing
            try:
                from anndata.io import sparse_dataset
            except ImportError:  # older anndata
                from anndata.experimental import sparse_dataset
            sub = sparse_dataset(b)[loc]
        dense = getattr(sub, "todense", None)
        return np.asarray(dense()) if dense is not None else np.asarray(sub)

    ctrl_mask = obs[control_key].to_numpy().astype(bool)
    ctrl_idx = np.flatnonzero(ctrl_mask)
    backings = key_backings(source, key)  # global-obs-order backings: 1 (AnnData) or per-dataset (collection)
    offs = np.concatenate([[0], np.cumsum([_backing_nrows(b) for b in backings])]).astype(np.int64)
    parts = []
    for d, b in enumerate(backings):
        loc = np.sort(ctrl_idx[(ctrl_idx >= offs[d]) & (ctrl_idx < offs[d + 1])] - offs[d])
        if loc.size:
            parts.append(_rows(b, loc))
    rep = np.concatenate(parts, axis=0) if parts else np.empty((0, _backing_ncols(backings[0])), dtype=np.float32)
    ctrl_obs = obs.loc[ctrl_mask, [*cols, control_key]].reset_index(drop=True)
    order = ctrl_obs.sort_values(list(cols), kind="stable").index.to_numpy()  # contiguous runs for chunk_size>1
    ctrl_obs, rep = ctrl_obs.iloc[order].reset_index(drop=True), rep[order]
    if key == "X":
        return ad.AnnData(X=rep, obs=ctrl_obs)  # construct WITH X so n_var is inferred (not 0)
    ctrl_ad = ad.AnnData(obs=ctrl_obs)  # obsm-only rep; X unused (n_var=0 is fine, the node reads obsm/<k>)
    ctrl_ad.obsm[key.split("/", 1)[1]] = rep
    return ctrl_ad


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
) -> AnnbatchTraining:
    """Assemble the :class:`dagloader.Scheme` + ``condition_fn`` for the streaming path (obs only).

    ``source`` is an out-of-core :class:`annbatch.DatasetCollection` or an in-memory ``AnnData``.
    ``rep_dict`` holds the covariate embedding tables (as ``adata.uns`` would); pass :obj:`None` when the
    primary covariate is categorical (one-hot).

    ``control_in_memory`` materializes the control cells into an in-memory ``AnnData`` (a second Scheme
    source) so the matched source/control is served from RAM while the perturbed target keeps streaming
    out of core — a large dataloader speedup, since controls are re-drawn every batch. Only enable it when
    the controls fit in host RAM (they are the small population by design).
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

    def condition_fn(leaf: Leaf) -> dict[str, np.ndarray]:
        idx = cov_to_idx[tuple(str(leaf[i]) for i in reorder)]
        return {group: condition_data[group][[idx]] for group in condition_data}

    # The Scheme: root = perturbed combos, child = matched-control combos (bound on the context columns).
    ctrl_flag = obs[control_key].to_numpy().astype(bool)
    pert = [tuple(r) for r in obs.loc[~ctrl_flag, list(cols)].drop_duplicates().to_numpy()]
    ctrl = [tuple(r) for r in obs.loc[ctrl_flag, list(cols)].drop_duplicates().to_numpy()]
    # `control_in_memory`: serve the matched control from a RAM-resident AnnData (a second source) while
    # the perturbed target keeps streaming out of core. The bind matches by label across sources.
    if control_in_memory:
        sources = {"data": source, "ctrl_mem": _materialize_controls(source, obs, cols, control_key, key)}
        ctrl_node = Node("ctrl_mem", cols, key, uniform(ctrl))
    else:
        sources = {"data": source}
        ctrl_node = Node("data", cols, key, uniform(ctrl))
    scheme = Scheme(
        sources=sources,
        nodes={"pert": Node("data", cols, key, uniform(pert)), "ctrl": ctrl_node},
        root="pert",
        binds=(Bind("pert", "ctrl", common=context),),
        seed=seed,
    )

    data_dim = _backing_ncols(key_backings(source, key)[0])
    return AnnbatchTraining(
        scheme=scheme,
        condition_fn=condition_fn,
        condition_data=condition_data,
        data_manager=dm,
        data_dim=data_dim,
        max_combination_length=dm.max_combination_length,
    )
