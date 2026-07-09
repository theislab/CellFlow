"""Tests for the annbatch-path scaffolding on :class:`~cellflow.model.CellFlow`.

Covers the ``adata``-optional constructor (deprecation of the constructor argument), passing
``adata`` to :meth:`prepare_data`, and the mode guard on :meth:`prepare_annbatch_data`.
"""

import anndata as ad
import pytest

import cellflow

PERT_COVARS = {"drug": ["drug1"]}
PERT_COVAR_REPS = {"drug": "drug"}


class TestAnnbatchPathScaffolding:
    def test_constructor_adata_emits_futurewarning(self, adata_perturbation: ad.AnnData):
        with pytest.warns(FutureWarning, match="prepare_data"):
            cellflow.model.CellFlow(adata_perturbation)

    def test_constructor_without_adata_is_silent(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            cf = cellflow.model.CellFlow()
        assert cf.adata is None

    def test_prepare_data_with_adata_kwarg(self, adata_perturbation: ad.AnnData):
        cf = cellflow.model.CellFlow()
        cf.prepare_data(
            sample_rep="X",
            control_key="control",
            perturbation_covariates=PERT_COVARS,
            perturbation_covariate_reps=PERT_COVAR_REPS,
            adata=adata_perturbation,
        )
        assert cf.train_data is not None
        assert cf.adata is adata_perturbation

    def test_prepare_data_without_any_adata_raises(self):
        cf = cellflow.model.CellFlow()
        with pytest.raises(ValueError, match="No `adata` provided"):
            cf.prepare_data(sample_rep="X", control_key="control", perturbation_covariates=PERT_COVARS)

    def test_prepare_annbatch_rejects_adata_model(self, adata_perturbation: ad.AnnData):
        with pytest.warns(FutureWarning):
            cf = cellflow.model.CellFlow(adata_perturbation)
        with pytest.raises(ValueError, match="without `adata`"):
            cf.prepare_annbatch_data(
                source=None, sample_rep="X", control_key="control", perturbation_covariates=PERT_COVARS
            )
