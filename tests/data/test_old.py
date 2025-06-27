import logging
from collections import OrderedDict

import anndata as ad
import numpy as np
import pytest

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture(autouse=True)
def setup_logging(caplog):
    # This ensures we capture all levels of logging
    caplog.set_level(logging.INFO)
    # This ensures we see the output even if a test fails
    # logging.getLogger().setLevel(logging.DEBUG)


from cellflow.data._datamanager import DataManager

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
    # first print if they are different
    for k in a.condition_data.keys():
        if a.condition_data[k].shape != b.condition_data[k].shape:
            print(f"condition_data[{k}].shape {a.condition_data[k].shape}, {b.condition_data[k].shape}")
        if not np.allclose(a.condition_data[k], b.condition_data[k]):
            print(f"condition_data[{k}] {a.condition_data[k]}, {b.condition_data[k]}")
        else:
            print(f"they are the same: {k}")
        #     f"condition_data[{k}].sum {a.condition_data[k].sum()}, {b.condition_data[k].sum()}"
        # )
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

        old = dm._get_condition_data_old(
            split_cov_combs=dm._get_split_cov_combs(adata_perturbation.obs),
            adata=adata_perturbation,
        )
        new = dm._get_condition_data(
            split_cov_combs=dm._get_split_cov_combs(adata_perturbation.obs),
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
        old = dm_old._get_condition_data_old(
            split_cov_combs=dm_old._get_split_cov_combs(adata_perturbation.obs),
            adata=adata_perturbation.copy(),
        )
        new = dm_new._get_condition_data(
            split_cov_combs=dm_new._get_split_cov_combs(adata_perturbation.obs),
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

        old = dm_old._get_condition_data_old(
            split_cov_combs=dm_old._get_split_cov_combs(adata_perturbation.obs.copy()),
            adata=adata_perturbation.copy(),
        )
        new = dm_new._get_condition_data(
            split_cov_combs=dm_new._get_split_cov_combs(adata_perturbation.obs),
            adata=adata_perturbation.copy(),
        )
        compare_train_data(old, new)
