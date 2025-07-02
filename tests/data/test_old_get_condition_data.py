import logging
from collections import OrderedDict
from typing import Any

import anndata
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from tqdm import tqdm

from cellflow._types import ArrayLike
from cellflow.data._datamanager import (
    DataManager,
    ReturnData,
    _to_list,
)


def _get_perturbation_covariates(
    dm: DataManager,
    condition_data: pd.DataFrame,
    rep_dict: dict[str, dict[str, ArrayLike]],
    perturb_covariates: Any,  # TODO: check if we can save as attribtue
) -> dict[str, np.ndarray]:
    primary_covars = perturb_covariates[dm.primary_group]
    perturb_covar_emb: dict[str, list[np.ndarray]] = {group: [] for group in perturb_covariates}
    for primary_cov in primary_covars:
        value = condition_data[primary_cov]
        cov_name = value if dm.is_categorical else primary_cov
        prim_arr1 = None
        if dm.primary_group in dm._covariate_reps:
            rep_key = dm._covariate_reps[dm.primary_group]
            if cov_name not in rep_dict[rep_key]:
                raise ValueError(f"Representation for '{cov_name}' not found in `adata.uns['{rep_key}']`.")
            prim_arr1 = np.asarray(rep_dict[rep_key][cov_name])
        else:
            prim_arr1 = np.asarray(
                dm.primary_one_hot_encoder.transform(  # type: ignore[union-attr]
                    np.array(cov_name).reshape(-1, 1)
                )
            )

        prim_arr2 = None
        if not dm.is_categorical:
            prim_arr2 = value * prim_arr1.copy()
        else:
            prim_arr2 = prim_arr1.copy()

        prim_arr3 = dm._check_shape(prim_arr2.copy())
        perturb_covar_emb[dm.primary_group].append(prim_arr3)
        for linked_covar in dm._linked_perturb_covars[primary_cov].items():
            linked_group, linked_cov = list(linked_covar)

            if linked_cov is None:
                linked_arr = np.full((1, 1), dm._null_value)
                linked_arr = dm._check_shape(linked_arr)
                perturb_covar_emb[linked_group].append(linked_arr)
                continue
            cov_name = condition_data[linked_cov]
            if linked_group in dm._covariate_reps:
                rep_key = dm._covariate_reps[linked_group]
                if cov_name not in rep_dict[rep_key]:
                    raise ValueError(f"Representation for '{cov_name}' not found in `adata.uns['{linked_group}']`.")
                linked_arr = np.asarray(rep_dict[rep_key][cov_name])
            else:
                linked_arr = np.asarray(condition_data[linked_cov])

            linked_arr = dm._check_shape(linked_arr)
            perturb_covar_emb[linked_group].append(linked_arr)

    perturb_covar_emb = {
        k: dm._pad_to_max_length(
            np.concatenate(v, axis=0),
            dm._max_combination_length,
            dm._null_value,
        )
        for k, v in perturb_covar_emb.items()
    }

    sample_covar_emb: dict[str, np.ndarray] = {}
    for sample_cov in dm._sample_covariates:
        value = condition_data[sample_cov]
        if sample_cov in dm._covariate_reps:
            rep_key = dm._covariate_reps[sample_cov]
            if value not in rep_dict[rep_key]:
                raise ValueError(f"Representation for '{value}' not found in `adata.uns['{sample_cov}']`.")
            cov_arr = np.asarray(rep_dict[rep_key][value])
        else:
            cov_arr = np.asarray(value)

        cov_arr = dm._check_shape(cov_arr)
        sample_covar_emb[sample_cov] = np.tile(cov_arr, (dm._max_combination_length, 1))
    return perturb_covar_emb | sample_covar_emb


def _get_condition_data_old(
    dm: DataManager,
    split_cov_combs: np.ndarray | list[list[Any]],
    adata: anndata.AnnData | None,
    covariate_data: pd.DataFrame | None = None,
    rep_dict: dict[str, Any] | None = None,
    condition_id_key: str | None = None,
) -> ReturnData:
    # for prediction: adata is None, covariate_data is provided
    # for training/validation: adata is provided and used to get cell masks, covariate_data is None
    if adata is None and covariate_data is None:
        raise ValueError("Either `adata` or `covariate_data` must be provided.")
    covariate_data = covariate_data if covariate_data is not None else adata.obs  # type: ignore[union-attr]
    if rep_dict is None:
        rep_dict = adata.uns if adata is not None else {}
    # check if all perturbation/split covariates and control cells are present in the input
    dm._verify_covariate_data(
        covariate_data,
        OrderedDict({covar: sorted(_to_list(covar)) for covar in dm._sample_covariates}),
    )
    dm._verify_control_data(adata)
    dm._verify_covariate_data(covariate_data, sorted(_to_list(dm._split_covariates)))
    # extract unique combinations of perturbation covariates
    if condition_id_key is not None:
        dm._verify_condition_id_key(covariate_data, condition_id_key)
        select_keys = dm._perturb_covar_keys + [condition_id_key]
    else:
        select_keys = dm._perturb_covar_keys
    perturb_covar_df = covariate_data[select_keys].drop_duplicates()
    if condition_id_key is not None:
        perturb_covar_df = perturb_covar_df.set_index(condition_id_key)
    else:
        perturb_covar_df = perturb_covar_df.reset_index()

    # get indices of cells belonging to each unique condition
    _perturb_covar_df, _covariate_data = (
        perturb_covar_df[dm._perturb_covar_keys],
        covariate_data[dm._perturb_covar_keys],
    )
    _perturb_covar_df["row_id"] = range(len(perturb_covar_df))
    _covariate_data["cell_index"] = _covariate_data.index
    _perturb_covar_merged = _perturb_covar_df.merge(_covariate_data, on=dm._perturb_covar_keys, how="inner")
    perturb_covar_to_cells = _perturb_covar_merged.groupby("row_id")["cell_index"].apply(list).to_list()
    # intialize data containers
    if adata is not None:
        split_covariates_mask = np.full((len(adata),), -1, dtype=np.int32)
        perturbation_covariates_mask = np.full((len(adata),), -1, dtype=np.int32)
        control_mask = covariate_data[dm._control_key]
    else:
        split_covariates_mask = None
        perturbation_covariates_mask = None
        control_mask = np.ones((len(covariate_data),))

    condition_data_temp: dict[str, list[tuple[int, np.ndarray]]] = {i: [] for i in dm._covar_to_idx.keys()}

    control_to_perturbation: dict[int, np.ndarray] = {}
    split_idx_to_covariates: dict[int, tuple[Any]] = {}
    perturbation_idx_to_covariates: dict[int, tuple[Any]] = {}
    perturbation_idx_to_id: dict[int, Any] = {}

    src_counter = 0
    tgt_counter = 0

    # iterate over unique split covariate combinations
    for split_combination in split_cov_combs:
        # get masks for split covariates; for prediction, it's done outside this method
        if adata is not None:
            split_covariates_mask, split_idx_to_covariates, split_cov_mask = dm._get_split_combination_mask(
                covariate_data=adata.obs,
                split_covariates_mask=split_covariates_mask,  # type: ignore[arg-type]
                split_combination=split_combination,
                split_idx_to_covariates=split_idx_to_covariates,
                control_mask=control_mask,
                src_counter=src_counter,
            )
        conditional_distributions = []

        # iterate over target conditions
        filter_dict = dict(zip(dm.split_covariates, split_combination, strict=False))
        pc_df = perturb_covar_df[(perturb_covar_df[list(filter_dict.keys())] == list(filter_dict.values())).all(axis=1)]
        pc_df = pc_df.sort_values(by=dm._perturb_covar_keys)
        pbar = tqdm(pc_df.iterrows(), total=pc_df.shape[0])
        perturb_covariates = OrderedDict({k: sorted(_to_list(v)) for k, v in dm._perturbation_covariates.items()})
        for i, tgt_cond in pbar:
            tgt_cond = tgt_cond[dm._perturb_covar_keys]
            # for train/validation, only extract covariate combinations that are present in adata
            if adata is not None:
                mask = covariate_data.index.isin(perturb_covar_to_cells[i])
                mask *= (1 - control_mask) * split_cov_mask
                mask = np.array(mask == 1)
                if mask.sum() == 0:
                    continue
                # map unique condition id to target id
                perturbation_covariates_mask[mask] = tgt_counter  # type: ignore[index]

            # map target id to unique conditions and their ids
            conditional_distributions.append(tgt_counter)
            perturbation_idx_to_covariates[tgt_counter] = tgt_cond.values
            if condition_id_key is not None:
                perturbation_idx_to_id[tgt_counter] = i

            # get embeddings for conditions
            embedding = _get_perturbation_covariates(
                dm=dm,
                condition_data=dict(tgt_cond),
                rep_dict=rep_dict.copy(),
                perturb_covariates=perturb_covariates,
            )
            for pert_cov, emb in embedding.items():
                condition_data_temp[pert_cov].append((tgt_counter, emb))

            tgt_counter += 1

        # map source (control) to target condition ids
        control_to_perturbation[src_counter] = np.array(sorted(conditional_distributions))
        src_counter += 1
    # convert outputs to numpy arrays

    condition_data = {}
    for pert_cov, emb_list in condition_data_temp.items():
        emb_list = sorted(emb_list, key=lambda x: x[0])
        condition_data[pert_cov] = np.array([emb for _, emb in emb_list])
    split_covariates_mask = np.asarray(split_covariates_mask) if split_covariates_mask is not None else None
    perturbation_covariates_mask = (
        np.asarray(perturbation_covariates_mask) if perturbation_covariates_mask is not None else None
    )

    return ReturnData(
        split_covariates_mask=split_covariates_mask,
        split_idx_to_covariates=split_idx_to_covariates,
        perturbation_covariates_mask=perturbation_covariates_mask,
        perturbation_idx_to_covariates=perturbation_idx_to_covariates,
        perturbation_idx_to_id=perturbation_idx_to_id,
        condition_data=condition_data,  # type: ignore[arg-type]
        control_to_perturbation=control_to_perturbation,
        max_combination_length=dm._max_combination_length,
    )


@pytest.fixture(autouse=True)
def setup_logging(caplog):
    # This ensures we capture all levels of logging
    caplog.set_level(logging.INFO)
    # This ensures we see the output even if a test fails
    # logging.getLogger().setLevel(logging.DEBUG)


perturbation_covariates_args = [
    OrderedDict({"drug": ["drug1"]}),
    OrderedDict({"drug": ["drug1"], "dosage": ["dosage_a"]}),
    OrderedDict(
        {
            "drug": ["drug_a"],
            "dosage": ["dosage_a"],
        }
    ),
]

perturbation_covariate_comb_args = [
    OrderedDict({"drug": ["drug1", "drug2"]}),
    OrderedDict({"drug": ["drug1", "drug2"], "dosage": ["dosage_a", "dosage_b"]}),
    OrderedDict(
        {
            "drug": ["drug_a", "drug_b", "drug_c"],
            "dosage": ["dosage_a", "dosage_b", "dosage_c"],
        }
    ),
]


def compare_masks(a: np.ndarray, b: np.ndarray, name: str):
    uniq_a = np.unique(a)
    uniq_b = np.unique(b)

    # get first occurence of each unique value
    a_ = [(e, next(i for i, x in enumerate(a) if x == e)) for e in uniq_a]
    b_ = [(e, next(i for i, x in enumerate(b) if x == e)) for e in uniq_b]

    a_ = sorted(a_, key=lambda x: x[1])
    b_ = sorted(b_, key=lambda x: x[1])

    a1 = [aa[1] for aa in a_]
    b1 = [bb[1] for bb in b_]
    assert a1 == b1, f"{name}: a: {a1}, b: {b1}, can't be mapped"

    a2b = {aa[0]: bb[0] for aa, bb in zip(a_, b_, strict=False)}

    for k, v in a2b.items():
        a_idx = np.argwhere(a == k)
        b_idx = np.argwhere(b == v)
        assert a_idx.shape == b_idx.shape, f"{name}: a: {a_idx.shape}, b: {b_idx.shape}"
        assert (a_idx == b_idx).all(), f"{name}: a: {a_idx}, b: {b_idx}"

    return a2b


def compare_train_data(a, b):
    a2b_perturbation = compare_masks(
        a.perturbation_covariates_mask, b.perturbation_covariates_mask, "perturbation_covariates_mask"
    )
    a2b_split = compare_masks(a.split_covariates_mask, b.split_covariates_mask, "split_covariates_mask")
    assert a.split_idx_to_covariates.keys() == b.split_idx_to_covariates.keys(), "split_idx_to_covariates"
    for k in a.split_idx_to_covariates.keys():
        if a2b_split:
            b_k = a2b_split[k]
        else:
            b_k = k
        assert a.split_idx_to_covariates[k] == b.split_idx_to_covariates[b_k], (
            f"split_idx_to_covariates[{k}] {a.split_idx_to_covariates[k]}, {b.split_idx_to_covariates[b_k]}"
        )
    assert a.perturbation_idx_to_covariates.keys() == b.perturbation_idx_to_covariates.keys(), (
        "perturbation_idx_to_covariates"
    )
    for k in a.perturbation_idx_to_covariates.keys():
        if a2b_perturbation:
            b_k = a2b_perturbation[k]
        else:
            b_k = k
        elem_a = a.perturbation_idx_to_covariates[k]
        elem_a = elem_a.tolist() if isinstance(elem_a, np.ndarray) else elem_a
        elem_b = b.perturbation_idx_to_covariates[b_k]
        elem_b = elem_b.tolist() if isinstance(elem_b, np.ndarray) else elem_b
        assert elem_a == elem_b, f"perturbation_idx_to_covariates[{k}] {elem_a}, {elem_b}"
    assert a.control_to_perturbation.keys() == b.control_to_perturbation.keys(), "control_to_perturbation"
    for k in a.control_to_perturbation.keys():
        elem_a = a.control_to_perturbation[k]
        elem_a = elem_a.tolist() if isinstance(elem_a, np.ndarray) else elem_a
        elem_b = b.control_to_perturbation[k]
        elem_b = elem_b.tolist() if isinstance(elem_b, np.ndarray) else elem_b
        assert len(elem_a) == len(elem_b), f"control_to_perturbation[{k}] {elem_a}, {elem_b}"
        for a_elem, b_elem in zip(elem_a, elem_b, strict=False):
            error_str = f"control_to_perturbation[{k}] {a_elem}, {b_elem}, {a.control_to_perturbation}, {b.control_to_perturbation}"
            if a2b_perturbation:
                error_str += f", a2b_perturbation[{a_elem}] {a2b_perturbation[a_elem]}"
            assert a_elem == b_elem, error_str
    assert a.condition_data.keys() == b.condition_data.keys(), "condition_data"
    for k in a.condition_data.keys():
        assert a.condition_data[k].shape == b.condition_data[k].shape, (
            f"condition_data[{k}].shape {a.condition_data[k].shape}, {b.condition_data[k].shape}"
        )
        assert np.allclose(a.condition_data[k].sum(), b.condition_data[k].sum()), (
            f"condition_data[{k}].sum {a.condition_data[k].sum()}, {b.condition_data[k].sum()}"
        )
        assert np.allclose(a.condition_data[k], b.condition_data[k]), (
            f"condition_data[{k}], {a.condition_data[k]}, {b.condition_data[k]}"
        )


class TestDataManager:
    @pytest.mark.parametrize("sample_rep", ["X", "X_pca"])
    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize("perturbation_covariates", perturbation_covariates_args)
    @pytest.mark.parametrize("perturbation_covariate_reps", [{}, {"drug": "drug"}])
    @pytest.mark.parametrize("sample_covariates", [[], ["dosage_c"]])
    def test_get_train_data(
        self,
        adata_perturbation: ad.AnnData,
        sample_rep,
        split_covariates,
        perturbation_covariates,
        perturbation_covariate_reps,
        sample_covariates,
    ):
        primary_group, _ = next(iter(perturbation_covariates.items()))
        dm = DataManager(
            adata_perturbation,
            sample_rep=sample_rep,
            split_covariates=split_covariates,
            control_key="control",
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=sample_covariates,
            primary_group=primary_group,
        )
        assert isinstance(dm, DataManager)
        assert dm._sample_rep == sample_rep
        assert dm._control_key == "control"
        assert dm._split_covariates == split_covariates
        assert dm._perturbation_covariates == perturbation_covariates
        assert dm._sample_covariates == sample_covariates

        old = _get_condition_data_old(
            dm=dm,
            split_cov_combs=dm._get_split_cov_combs(adata_perturbation.obs),
            adata=adata_perturbation,
        )
        new = dm._get_condition_data(
            adata=adata_perturbation,
        )

        compare_train_data(old, new)

    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize("perturbation_covariates", perturbation_covariate_comb_args)
    @pytest.mark.parametrize("perturbation_covariate_reps", [{}, {"drug": "drug"}])
    def test_get_train_data_with_combinations(
        self,
        adata_perturbation: ad.AnnData,
        split_covariates,
        perturbation_covariates,
        perturbation_covariate_reps,
        caplog,
    ):
        primary_group, _ = next(iter(perturbation_covariates.items()))
        dm_old = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=split_covariates,
            control_key="control",
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=["cell_type"],
            sample_covariate_reps={"cell_type": "cell_type"},
            primary_group=primary_group,
        )
        dm_new = DataManager(
            adata_perturbation.copy(),
            sample_rep="X",
            split_covariates=split_covariates,
            control_key="control",
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=["cell_type"],
            sample_covariate_reps={"cell_type": "cell_type"},
            primary_group=primary_group,
        )
        old = _get_condition_data_old(
            dm=dm_old,
            split_cov_combs=dm_old._get_split_cov_combs(adata_perturbation.obs),
            adata=adata_perturbation.copy(),
        )
        new = dm_new._get_condition_data(
            adata=adata_perturbation.copy(),
        )
        compare_train_data(old, new)


class TestValidationData:
    @pytest.mark.parametrize("sample_rep", ["X", "X_pca"])
    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize("perturbation_covariates", perturbation_covariate_comb_args)
    @pytest.mark.parametrize("perturbation_covariate_reps", [{}, {"drug": "drug"}])
    def test_get_validation_data(
        self,
        adata_perturbation: ad.AnnData,
        sample_rep,
        split_covariates,
        perturbation_covariates,
        perturbation_covariate_reps,
    ):
        control_key = "control"
        sample_covariates = ["cell_type"]
        sample_covariate_reps = {"cell_type": "cell_type"}
        primary_group, _ = next(iter(perturbation_covariates.items()))
        dm_old = DataManager(
            adata_perturbation.copy(),
            sample_rep=sample_rep,
            split_covariates=split_covariates.copy(),
            control_key=control_key,
            perturbation_covariates=perturbation_covariates.copy(),
            perturbation_covariate_reps=perturbation_covariate_reps.copy(),
            sample_covariates=sample_covariates.copy(),
            sample_covariate_reps=sample_covariate_reps.copy(),
            primary_group=primary_group,
        )
        dm_new = DataManager(
            adata_perturbation.copy(),
            sample_rep=sample_rep,
            split_covariates=split_covariates.copy(),
            control_key=control_key,
            perturbation_covariates=perturbation_covariates.copy(),
            perturbation_covariate_reps=perturbation_covariate_reps.copy(),
            sample_covariates=sample_covariates.copy(),
            sample_covariate_reps=sample_covariate_reps.copy(),
            primary_group=primary_group,
        )

        old = _get_condition_data_old(
            dm=dm_old,
            split_cov_combs=dm_old._get_split_cov_combs(adata_perturbation.obs.copy()),
            adata=adata_perturbation.copy(),
        )
        new = dm_new._get_condition_data(
            adata=adata_perturbation.copy(),
        )
        compare_train_data(old, new)
