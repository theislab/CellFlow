import numpy as np
import pytest

from cellflow.data._dataloader import OOCTrainSampler, PredictionSampler, TrainSampler
from cellflow.data._datamanager import DataManager


class TestTrainSampler:
    @pytest.mark.parametrize("batch_size", [1, 31])
    def test_sampling_no_combinations(self, adata_perturbation, batch_size: int):
        sample_rep = "X"
        split_covariates = ["cell_type"]
        control_key = "control"
        perturbation_covariates = {
            "drug": ("drug1", "drug2"),
            "dosage": ("dosage_a", "dosage_b"),
        }
        perturbation_covariate_reps = {"drug": "drug"}

        dm = DataManager(
            adata_perturbation,
            sample_rep=sample_rep,
            split_covariates=split_covariates,
            control_key=control_key,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
        )

        train_data = dm.get_train_data(adata_perturbation)
        sampler = TrainSampler(data=train_data, batch_size=batch_size)
        rng_1 = np.random.default_rng(0)
        rng_2 = np.random.default_rng(1)

        sample_1 = sampler.sample(rng_1)
        sample_2 = sampler.sample(rng_2)

        assert "src_cell_data" in sample_1
        assert "tgt_cell_data" in sample_1
        assert "condition" in sample_1
        assert sample_1["src_cell_data"].shape[0] == batch_size
        assert sample_2["src_cell_data"].shape[0] == batch_size
        assert sample_1["tgt_cell_data"].shape[0] == batch_size
        assert sample_2["tgt_cell_data"].shape[0] == batch_size
        assert sample_1["condition"]["dosage"].shape[0] == 1
        assert sample_2["condition"]["dosage"].shape[0] == 1

class TestOOCTrainSampler:
    @pytest.mark.parametrize("batch_size", [1, 31])
    def test_sampling_no_combinations(self, adata_perturbation, batch_size: int):
        sample_rep = "X"
        split_covariates = ["cell_type"]
        control_key = "control"
        perturbation_covariates = {
            "drug": ("drug1", "drug2"),
            "dosage": ("dosage_a", "dosage_b"),
        }
        perturbation_covariate_reps = {"drug": "drug"}

        dm = DataManager(
            adata_perturbation,
            sample_rep=sample_rep,
            split_covariates=split_covariates,
            control_key=control_key,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
        )

        train_data = dm.get_train_data(adata_perturbation)
        sampler = OOCTrainSampler(data=train_data, batch_size=batch_size, seed=0)
        sampler.set_sampler(num_iterations=2)
        sample_1 = sampler.sample()
        sample_2 = sampler.sample()

        assert "src_cell_data" in sample_1
        assert "tgt_cell_data" in sample_1
        assert "condition" in sample_1
        assert sample_1["src_cell_data"].shape[0] == batch_size
        assert sample_2["src_cell_data"].shape[0] == batch_size
        assert sample_1["tgt_cell_data"].shape[0] == batch_size
        assert sample_2["tgt_cell_data"].shape[0] == batch_size
        assert sample_1["condition"]["dosage"].shape[0] == 1
        assert sample_2["condition"]["dosage"].shape[0] == 1


class TestValidationSampler:
    @pytest.mark.parametrize("n_conditions_on_log_iteration", [None, 1, 3])
    def test_valid_sampler(self, adata_perturbation, n_conditions_on_log_iteration):
        from cellflow.data._dataloader import ValidationSampler
        from cellflow.data._datamanager import DataManager

        control_key = "control"
        sample_covariates = ["cell_type"]
        sample_covariate_reps = {"cell_type": "cell_type"}

        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key=control_key,
            perturbation_covariates={"drug": ["drug1"]},
            perturbation_covariate_reps={"drug": "drug"},
            sample_covariates=sample_covariates,
            sample_covariate_reps=sample_covariate_reps,
        )

        val_data = dm.get_validation_data(
            adata_perturbation,
            n_conditions_on_log_iteration=n_conditions_on_log_iteration,
        )

        s = ValidationSampler(val_data)
        out = s.sample(mode="on_log_iteration")
        assert "source" in out
        assert "target" in out
        assert "condition" in out
        assert len(out["target"]) == len(out["source"])
        assert len(out["target"]) == len(out["condition"])
        assert (
            len(out["target"]) == n_conditions_on_log_iteration
            if n_conditions_on_log_iteration is not None
            else val_data.n_perturbations
        )
        assert out["source"].keys() == out["target"].keys()
        assert out["source"].keys() == out["condition"].keys()


class TestPredictionSampler:
    @pytest.mark.parametrize("sample_rep", ["X", "X_pca"])
    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize("perturbation_covariate_reps", [{}, {"drug": "drug"}])
    def test_pred_sampler(
        self,
        adata_perturbation,
        sample_rep,
        split_covariates,
        perturbation_covariate_reps,
    ):
        from cellflow.data._datamanager import DataManager

        perturbation_covariates = {"drug": ["drug1", "drug2"]}

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

        adata_pred = adata_perturbation[:10].copy()
        adata_pred.obs["control"] = True
        pred_data = dm.get_prediction_data(adata_pred, covariate_data=adata_pred.obs, sample_rep=sample_rep)
        s = PredictionSampler(pred_data)
        out = s.sample()
        assert "source" in out
        assert "condition" in out
        assert len(out["source"]) == len(out["condition"])
