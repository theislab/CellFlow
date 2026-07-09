"""Tests for ``condition_id_key`` handling in :class:`~cellflow.data.DataManager`.

Includes regression coverage for:

- issue #284: a perturbation covariate key used directly as the ``condition_id_key``
  (``condition_id_key`` coincides with one of ``perturb_covar_keys``);
- the pandas>=3 ``.loc`` lookup on a flat index (single perturbation covariate key).
"""

import pandas as pd
import pytest

from cellflow.data._data import ConditionData
from cellflow.data._datamanager import DataManager


def _make_dm(adata, perturbation_covariates):
    return DataManager(
        adata,
        sample_rep="X",
        split_covariates=[],
        control_key="control",
        perturbation_covariates=perturbation_covariates,
    )


def _covariate_data(adata, dm):
    """Unique perturbation conditions, keeping all obs columns (incl. ``control``)."""
    return adata.obs.drop_duplicates(subset=dm.perturb_covar_keys).copy().reset_index(drop=True)


class TestConditionIdKeyValidation:
    def test_missing_column_raises(self, adata_perturbation):
        dm = _make_dm(adata_perturbation, {"drug": ["drug1"]})
        covariate_data = _covariate_data(adata_perturbation, dm)
        with pytest.raises(ValueError, match=r".*condition_id_key.*not found.*"):
            dm.get_condition_data(
                covariate_data=covariate_data,
                rep_dict=adata_perturbation.uns,
                condition_id_key="does_not_exist",
            )

    def test_non_unique_values_raises(self, adata_perturbation):
        dm = _make_dm(adata_perturbation, {"drug": ["drug1"]})
        covariate_data = _covariate_data(adata_perturbation, dm)
        covariate_data["condition_id"] = "same_value"
        with pytest.raises(ValueError, match=r".*must contain unique values.*"):
            dm.get_condition_data(
                covariate_data=covariate_data,
                rep_dict=adata_perturbation.uns,
                condition_id_key="condition_id",
            )


class TestConditionIdKey:
    @pytest.mark.parametrize(
        "perturbation_covariates",
        [
            {"drug": ["drug1"]},  # single key -> flat index (pandas>=3 `.loc`)
            {"drug": ["drug1", "drug2"]},  # combination -> MultiIndex
            {"drug": ["drug1"], "dosage": ["dosage_a"]},  # multiple keys -> MultiIndex
        ],
    )
    def test_separate_condition_id_column(self, adata_perturbation, perturbation_covariates):
        dm = _make_dm(adata_perturbation, perturbation_covariates)
        covariate_data = _covariate_data(adata_perturbation, dm)
        covariate_data["condition_id"] = [f"cond_{i}" for i in range(len(covariate_data))]

        cond_data = dm.get_condition_data(
            covariate_data=covariate_data,
            rep_dict=adata_perturbation.uns,
            condition_id_key="condition_id",
        )
        assert isinstance(cond_data, ConditionData)
        assert len(cond_data.perturbation_idx_to_id) == len(covariate_data)
        assert set(cond_data.perturbation_idx_to_id.values()) == set(covariate_data["condition_id"])

    def test_condition_id_key_equals_perturb_covar_key(self, adata_perturbation):
        """Regression for #284: ``condition_id_key`` is itself a perturbation covariate key."""
        dm = _make_dm(adata_perturbation, {"drug": ["drug1"]})
        covariate_data = _covariate_data(adata_perturbation, dm)

        cond_data = dm.get_condition_data(
            covariate_data=covariate_data,
            rep_dict=adata_perturbation.uns,
            condition_id_key="drug1",
        )
        assert isinstance(cond_data, ConditionData)
        assert len(cond_data.perturbation_idx_to_id) == len(covariate_data)
        # the ids are exactly the `drug1` values of each condition
        expected = set(covariate_data["drug1"].astype(str))
        actual = {str(v) for v in cond_data.perturbation_idx_to_id.values()}
        assert actual == expected

    def test_none_returns_empty(self, adata_perturbation):
        dm = _make_dm(adata_perturbation, {"drug": ["drug1"]})
        covariate_data = _covariate_data(adata_perturbation, dm)
        cond_data = dm.get_condition_data(
            covariate_data=covariate_data,
            rep_dict=adata_perturbation.uns,
            condition_id_key=None,
        )
        assert len(cond_data.perturbation_idx_to_id) == 0


class TestGetPerturbCovarDf:
    def test_distinct_condition_id_key(self):
        covariate_data = pd.DataFrame(
            {
                "drug1": ["drug_a", "drug_b", "drug_a"],
                "dosage": [10, 20, 10],
                "condition_id": ["cond_0", "cond_1", "cond_0"],
            }
        )
        result = DataManager._get_perturb_covar_df(
            covariate_data=covariate_data,
            perturb_covar_keys=["drug1", "dosage"],
            condition_id_key="condition_id",
        )
        assert result.index.name == "condition_id"
        assert set(result.index) == {"cond_0", "cond_1"}
        assert "drug1" in result.columns
        assert "dosage" in result.columns

    def test_condition_id_key_equals_covar_key(self):
        """#284 at the unit level: ``condition_id_key`` is one of ``perturb_covar_keys``."""
        covariate_data = pd.DataFrame({"drug1": ["drug_a", "drug_b", "drug_a"]})
        result = DataManager._get_perturb_covar_df(
            covariate_data=covariate_data,
            perturb_covar_keys=["drug1"],
            condition_id_key="drug1",
        )
        # must not raise, and the covariate column must survive for downstream re-indexing
        assert "drug1" in result.columns
        assert set(result.index) == {"drug_a", "drug_b"}

    def test_without_condition_id_key(self):
        covariate_data = pd.DataFrame({"drug1": ["drug_a", "drug_b", "drug_a"], "dosage": [10, 20, 10]})
        result = DataManager._get_perturb_covar_df(
            covariate_data=covariate_data,
            perturb_covar_keys=["drug1", "dosage"],
            condition_id_key=None,
        )
        assert "drug1" in result.columns
        assert "dosage" in result.columns
        assert len(result) == 2
