"""Pure helpers for building condition data from ``obs`` + covariate representations.

Extracted from :class:`~cellflow.data._datamanager.DataManager` so the in-memory path and the
annbatch/binded streaming path share **one** implementation — they must produce identical
condition embeddings. Nothing here touches the cell matrix (``X``); only the covariate spec, the
``obs`` table, and the representation dict (``uns``) are used. ``DataManager`` delegates to these.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from cellflow._logging import logger
from cellflow.data._utils import _to_list

__all__ = ["build_condition_data", "enumerate_perturbations", "get_max_combination_length"]


def _key_layout(
    perturb_covar_keys: Sequence[str],
    split_covariates: Sequence[str],
    sample_covariates: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Column layouts used by DataManager's condition enumeration.

    Returns ``(all_combs_keys, tuple_keys)``: the sort order for assigning perturbation indices, and
    the order each ``perturbation_idx_to_covariates`` tuple is laid out in. Shared by
    :func:`enumerate_perturbations` and :func:`build_condition_data` so both agree with DataManager.
    """
    uniq_sample_keys = list(split_covariates) if len(split_covariates) else list(sample_covariates)
    perturbation_covariates_keys = [k for k in perturb_covar_keys if k not in uniq_sample_keys]
    if len(split_covariates):
        all_combs_keys = uniq_sample_keys + perturbation_covariates_keys
    else:
        all_combs_keys = perturbation_covariates_keys + uniq_sample_keys
    tuple_keys = perturbation_covariates_keys + uniq_sample_keys
    return all_combs_keys, tuple_keys


def get_max_combination_length(
    perturbation_covariates: dict[str, Sequence[str]],
    max_combination_length: int | None = None,
) -> int:
    """Maximum number of perturbations in a combination (a pure function of the spec).

    This is the largest covariate-group size in ``perturbation_covariates`` (e.g. ``{"drug":
    ("drug_1", "drug_2")}`` ⇒ ``2``). ``max_combination_length`` acts only as a floor: a value below
    the observed maximum is raised to it (with a warning); a larger value is kept as-is.
    """
    obs_max_combination_length = max(len(comb) for comb in perturbation_covariates.values())
    if max_combination_length is None:
        return obs_max_combination_length
    elif max_combination_length < obs_max_combination_length:
        logger.warning(
            f"Provided `max_combination_length` is smaller than the observed maximum combination length of the perturbation covariates. Setting maximum combination length to {obs_max_combination_length}.",
            stacklevel=2,
        )
        return obs_max_combination_length
    else:
        return max_combination_length


def enumerate_perturbations(
    obs: pd.DataFrame,
    *,
    control_key: str,
    perturb_covar_keys: Sequence[str],
    split_covariates: Sequence[str],
    sample_covariates: Sequence[str],
) -> dict[int, tuple]:
    """Map each perturbation index to its covariate tuple, matching ``DataManager`` (no dask/masks).

    Reproduces the target-combination enumeration inside
    :meth:`~cellflow.data._datamanager.DataManager._get_condition_data`: the unique non-control
    combinations, indexed by ``arange`` after sorting by ``all_combs_keys``, each tuple laid out in
    ``perturbation_covariates_keys + uniq_sample_keys`` order. Uses plain pandas (no per-cell masks,
    no dask), so it is cheap enough to run off a streamed ``obs`` table.
    """
    all_combs_keys, tuple_keys = _key_layout(perturb_covar_keys, split_covariates, sample_covariates)

    df = obs[[*all_combs_keys, control_key]].copy()
    for col in all_combs_keys:  # mirror DataManager: categorical sort order (only cast if needed)
        if df[col].dtype != "category":
            df[col] = df[col].astype("category")
    df = df[~df[control_key].astype(bool)]
    combos = df[all_combs_keys].drop_duplicates().sort_values(by=all_combs_keys).reset_index(drop=True)
    return {i: tuple(combos.loc[i, tuple_keys]) for i in range(len(combos))}


def build_condition_data(
    obs: pd.DataFrame,
    rep_dict: Mapping[str, Any],
    *,
    control_key: str,
    perturb_covar_keys: Sequence[str],
    split_covariates: Sequence[str],
    sample_covariates: Sequence[str],
    perturbation_covariates: Mapping[str, Sequence[str]],
    covariate_reps: Mapping[str, str],
    covar_to_idx: Mapping[str, int],
    is_categorical: bool,
    primary_one_hot_encoder: Any,
    primary_group: str,
    linked_perturb_covars: Mapping[str, Any],
    max_combination_length: int,
    null_value: float,
) -> dict[str, np.ndarray]:
    """Assemble the per-condition embeddings (``condition_data``), matching ``DataManager`` (no dask).

    Enumerates the perturbations (:func:`enumerate_perturbations`), then reuses the same per-condition
    embedding function as the in-memory path — :meth:`DataManager._get_embeddings` — over a plain
    serial loop instead of DataManager's ``dask.delayed`` fan-out. The returned dict maps each covariate
    group to an ``(n_perturbations, max_combination_length, dim)`` array, aligned to the perturbation
    index from :func:`enumerate_perturbations`. Values are identical to
    ``DataManager._get_condition_data(...).condition_data`` (parity-tested); only the orchestration
    differs, so it is cheap enough to run off a streamed ``obs`` table.
    """
    from cellflow.data._datamanager import DataManager  # lazy: reuse the shared per-condition embedding

    _, tuple_keys = _key_layout(perturb_covar_keys, split_covariates, sample_covariates)
    idx_to_covariates = enumerate_perturbations(
        obs,
        control_key=control_key,
        perturb_covar_keys=perturb_covar_keys,
        split_covariates=split_covariates,
        sample_covariates=sample_covariates,
    )
    perturb_covariates = OrderedDict({k: sorted(_to_list(v)) for k, v in perturbation_covariates.items()})
    condition_data: dict[str, list[np.ndarray]] = {k: [] for k in covar_to_idx}
    for idx in sorted(idx_to_covariates):
        tgt_cond = dict(zip(tuple_keys, idx_to_covariates[idx], strict=True))
        embeddings = DataManager._get_embeddings(
            condition_data=tgt_cond,
            rep_dict=rep_dict,
            perturb_covariates=perturb_covariates,
            covariate_reps=covariate_reps,
            is_categorical=is_categorical,
            primary_one_hot_encoder=primary_one_hot_encoder,
            null_value=null_value,
            max_combination_length=max_combination_length,
            linked_perturb_covars=linked_perturb_covars,
            sample_covariates=sample_covariates,
            primary_group=primary_group,
        )
        for pert_cov, emb in embeddings.items():
            condition_data[pert_cov].append(emb)
    return {pert_cov: np.array(emb) for pert_cov, emb in condition_data.items()}
