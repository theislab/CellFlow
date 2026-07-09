"""Build the annbatch/dagloader streaming training path from a CellFlow covariate spec.

Turns the same ``prepare_data`` covariate arguments into a :class:`dagloader.Scheme` (perturbed root,
matched-control child) plus a ``condition_fn`` that maps a sampled leaf to its per-condition embedding.
The condition embeddings reuse the **in-memory** machinery unchanged: a cell-free "shell" ``AnnData``
(``obs`` + ``uns`` only, no ``X``) drives a :class:`~cellflow.data._datamanager.DataManager` purely as a
covariate-encoder factory, and :func:`~cellflow.data._condition.build_condition_data` assembles the
embeddings — byte-for-byte identical to the in-memory path (parity-tested), so nothing is duplicated.

Only ``obs`` (and the embedding tables) are read here; cells are streamed later by ``dagloader``.
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

__all__ = ["AnnbatchTraining", "build_annbatch_training", "sample_rep_to_key"]


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

    ``source`` is an out-of-core :class:`annbatch.DatasetCollection` or an in-memory ``AnnData`` (the
    dagloader is container-agnostic). ``rep_dict`` supplies the covariate embedding tables that
    ``adata.uns`` would hold in the in-memory path (keys match ``*_covariate_reps`` values); it may be
    :obj:`None` when the primary covariate is categorical (one-hot encoded, no external embeddings).
    """
    from dagloader import Bind, Node, Scheme, uniform
    from dagloader._io import key_backings, obs_columns

    context = tuple(split_covariates or ())
    pert_cols = tuple(c for grp in perturbation_covariates.values() for c in grp)
    samp_cols = tuple(sample_covariates or ())
    cols = tuple(dict.fromkeys((*context, *pert_cols, *samp_cols)))  # grouping cols (deduped, ordered)
    key = sample_rep_to_key(sample_rep)

    obs = obs_columns(source, [*cols, control_key])

    # DataManager as a pure covariate-encoder factory. It reads only obs + uns (never X), so a cell-free
    # shell suffices; `sample_rep="X"` sidesteps obsm verification (the real rep is streamed by the Scheme).
    shell = ad.AnnData(obs=obs.copy())
    shell.uns = dict(rep_dict or {})
    dm = DataManager(
        shell,
        sample_rep="X",
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

    # leaf (in `cols` order) → perturbation index → per-condition embedding. `enumerate_perturbations`
    # lays its covariate tuples out in `tuple_keys` order (pert + sample/split), which differs from
    # `cols`, so re-project the leaf onto that layout before the lookup. String-normalize both sides so
    # dtype quirks (categorical / numpy scalars) never break the match.
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
