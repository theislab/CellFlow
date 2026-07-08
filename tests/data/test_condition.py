"""Tests for the extracted condition helpers in :mod:`cellflow.data._condition`.

These guard the "act the same" contract: the standalone functions must agree with the
:class:`~cellflow.data._datamanager.DataManager` methods they were extracted from (DataManager
delegates to them, so any divergence is a regression).
"""

import anndata as ad
import numpy as np
import pytest

from cellflow.data._condition import build_condition_data, enumerate_perturbations, get_max_combination_length
from cellflow.data._datamanager import DataManager

SPECS = [
    {"drug": ["drug1"]},
    {"drug": ["drug1", "drug2"]},
    {"drug": ["drug1", "drug2"], "dosage": ["dosage_a", "dosage_b"]},
    {"drug": ["drug_a", "drug_b", "drug_c"], "dosage": ["dosage_a", "dosage_b", "dosage_c"]},
]


class TestGetMaxCombinationLength:
    @pytest.mark.parametrize("spec", SPECS)
    @pytest.mark.parametrize("override", [None, 1, 2, 5, 100])
    def test_matches_datamanager(self, spec, override):
        # single source of truth: standalone == DataManager (which delegates to it)
        assert get_max_combination_length(spec, override) == DataManager._get_max_combination_length(spec, override)

    def test_no_override_is_observed_max(self):
        assert get_max_combination_length({"drug": ["d1", "d2"], "dose": ["x1"]}) == 2

    def test_larger_override_kept(self):
        assert get_max_combination_length({"drug": ["d1", "d2"]}, 5) == 5

    def test_smaller_override_raised_to_observed_with_warning(self, caplog):
        assert get_max_combination_length({"drug": ["d1", "d2"]}, 1) == 2


ENUM_CASES = [
    ({"drug": ["drug1"]}, ["cell_type"], []),
    ({"drug": ["drug1", "drug2"]}, ["cell_type"], []),
    ({"drug": ["drug1"]}, [], []),
    ({"drug": ["drug1"]}, [], ["dosage_c"]),
    ({"drug": ["drug1"], "dosage": ["dosage_a"]}, ["cell_type"], []),
]


class TestEnumeratePerturbations:
    @pytest.mark.parametrize(("perturbation_covariates", "split_covariates", "sample_covariates"), ENUM_CASES)
    def test_matches_datamanager_idx_to_covariates(
        self,
        adata_perturbation: ad.AnnData,
        perturbation_covariates,
        split_covariates,
        sample_covariates,
    ):
        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            control_key="control",
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps={"drug": "drug"},
            split_covariates=split_covariates,
            sample_covariates=sample_covariates,
        )
        expected = {
            int(k): tuple(v)
            for k, v in dm._get_condition_data(adata=adata_perturbation).perturbation_idx_to_covariates.items()
        }
        got = enumerate_perturbations(
            adata_perturbation.obs,
            control_key="control",
            perturb_covar_keys=dm._perturb_covar_keys,
            split_covariates=dm._split_covariates,
            sample_covariates=dm._sample_covariates,
        )
        assert got == expected


BUILD_CASES = [
    ({"drug": ["drug1"]}, {"drug": "drug"}, ["cell_type"], []),
    ({"drug": ["drug1", "drug2"]}, {"drug": "drug"}, ["cell_type"], []),
    ({"drug": ["drug1"]}, {"drug": "drug"}, [], ["dosage_c"]),
    ({"drug": ["drug1"], "dosage": ["dosage_a"]}, {"drug": "drug"}, ["cell_type"], []),
    ({"drug": ["drug1"]}, {}, ["cell_type"], []),  # no rep -> primary one-hot-encoder path
]


class TestBuildConditionData:
    @pytest.mark.parametrize(
        ("perturbation_covariates", "perturbation_covariate_reps", "split_covariates", "sample_covariates"),
        BUILD_CASES,
    )
    def test_matches_datamanager_condition_data(
        self,
        adata_perturbation: ad.AnnData,
        perturbation_covariates,
        perturbation_covariate_reps,
        split_covariates,
        sample_covariates,
    ):
        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            control_key="control",
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            split_covariates=split_covariates,
            sample_covariates=sample_covariates,
        )
        ref = dm._get_condition_data(adata=adata_perturbation).condition_data
        got = build_condition_data(
            adata_perturbation.obs,
            adata_perturbation.uns,
            control_key="control",
            perturb_covar_keys=dm.perturb_covar_keys,
            split_covariates=dm.split_covariates,
            sample_covariates=dm.sample_covariates,
            perturbation_covariates=dm.perturbation_covariates,
            covariate_reps=dm.covariate_reps,
            covar_to_idx=dm._covar_to_idx,
            is_categorical=dm.is_categorical,
            primary_one_hot_encoder=dm.primary_one_hot_encoder,
            primary_group=dm.primary_group,
            linked_perturb_covars=dm.linked_perturb_covars,
            max_combination_length=dm.max_combination_length,
            null_value=dm.null_value,
        )
        assert set(got) == set(ref)
        for key in ref:
            np.testing.assert_array_equal(got[key], ref[key])
