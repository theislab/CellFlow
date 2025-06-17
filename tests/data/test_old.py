import anndata as ad
import numpy as np
import pytest

from cellflow.data._datamanager import DataManager

perturbation_covariates_args = [
    {"drug": ["drug1"]},
    {"drug": ["drug1"], "dosage": ["dosage_a"]},
    {
        "drug": ["drug_a"],
        "dosage": ["dosage_a"],
    },
]

perturbation_covariate_comb_args = [
    {"drug": ["drug1", "drug2"]},
    {"drug": ["drug1", "drug2"], "dosage": ["dosage_a", "dosage_b"]},
    {
        "drug": ["drug_a", "drug_b", "drug_c"],
        "dosage": ["dosage_a", "dosage_b", "dosage_c"],
    },
]


def compare_train_data(a, b):
    assert (a.perturbation_covariates_mask == b.perturbation_covariates_mask).all(), f"perturbation_covariates_mask {a.perturbation_covariates_mask}, {b.perturbation_covariates_mask}"
    assert (a.split_covariates_mask == b.split_covariates_mask).all(), f"split_covariates_mask {a.split_covariates_mask}, {b.split_covariates_mask}"
    # compare split_idx_to_covariates and perturbation_idx_to_covariates dicts
    assert a.split_idx_to_covariates.keys() == b.split_idx_to_covariates.keys(), "split_idx_to_covariates"
    for k in a.split_idx_to_covariates.keys():
        assert (a.split_idx_to_covariates[k] == b.split_idx_to_covariates[k]), f"split_idx_to_covariates[{k}] {a.split_idx_to_covariates[k]}, {b.split_idx_to_covariates[k]}"
    assert a.perturbation_idx_to_covariates.keys() == b.perturbation_idx_to_covariates.keys(), "perturbation_idx_to_covariates"
    for k in a.perturbation_idx_to_covariates.keys():
        elem_a = a.perturbation_idx_to_covariates[k]
        elem_a = elem_a.tolist() if isinstance(elem_a, np.ndarray) else elem_a
        elem_b = b.perturbation_idx_to_covariates[k]
        elem_b = elem_b.tolist() if isinstance(elem_b, np.ndarray) else elem_b
        assert (elem_a == elem_b), f"perturbation_idx_to_covariates[{k}] {elem_a}, {elem_b}"
    assert a.control_to_perturbation.keys() == b.control_to_perturbation.keys(), "control_to_perturbation"
    for k in a.control_to_perturbation.keys():
        assert (a.control_to_perturbation[k] == b.control_to_perturbation[k]).all().all(), f"control_to_perturbation[{k}]"
    assert a.condition_data.keys() == b.condition_data.keys(), "condition_data"
    for k in a.condition_data.keys():
        assert (a.condition_data[k] == b.condition_data[k]).all(), f"condition_data[{k}]"


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
        from cellflow.data._data import TrainingData
        from cellflow.data._datamanager import DataManager

        dm = DataManager(
            adata_perturbation,
            sample_rep=sample_rep,
            split_covariates=split_covariates,
            control_key="control",
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=sample_covariates,
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
    ):
        from cellflow.data._datamanager import DataManager

        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=split_covariates,
            control_key="control",
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=["cell_type"],
            sample_covariate_reps={"cell_type": "cell_type"},
        )
        old = dm._get_condition_data_old(
            split_cov_combs=dm._get_split_cov_combs(adata_perturbation.obs),
            adata=adata_perturbation,
        )
        new = dm._get_condition_data(
            split_cov_combs=dm._get_split_cov_combs(adata_perturbation.obs),
            adata=adata_perturbation,
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
        from cellflow.data._datamanager import DataManager

        control_key = "control"
        sample_covariates = ["cell_type"]
        sample_covariate_reps = {"cell_type": "cell_type"}

        dm = DataManager(
            adata_perturbation,
            sample_rep=sample_rep,
            split_covariates=split_covariates,
            control_key=control_key,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=sample_covariates,
            sample_covariate_reps=sample_covariate_reps,
        )

        old = dm._get_condition_data_old(
            split_cov_combs=dm._get_split_cov_combs(adata_perturbation.obs),
            adata=adata_perturbation,
        )
        new = dm._get_condition_data(
            split_cov_combs=dm._get_split_cov_combs(adata_perturbation.obs),
            adata=adata_perturbation,
        )
        compare_train_data(old, new)
