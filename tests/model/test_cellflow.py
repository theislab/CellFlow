import jax
import pandas as pd
import pytest

import cfp

perturbation_covariate_comb_args = [
    {"drug": ["drug1"]},
    {"drug": ["drug1", "drug2"], "dosage": ["dosage_a", "dosage_b"]},
    {
        "drug": ["drug_a", "drug_b", "drug_c"],
        "dosage": ["dosage_a", "dosage_b", "dosage_c"],
    },
]


class TestCellFlow:
    @pytest.mark.parametrize("solver", ["otfm", "genot"])
    @pytest.mark.parametrize("time_encoder_batchnorm", [True, False])
    @pytest.mark.parametrize("hidden_batchnorm", [True, False])
    @pytest.mark.parametrize("decoder_batchnorm", [True, False])
    def test_cellflow_solver(
        self,
        adata_perturbation,
        solver,
        time_encoder_batchnorm,
        hidden_batchnorm,
        decoder_batchnorm,
    ):
        sample_rep = "X"
        control_key = "control"
        perturbation_covariates = {"drug": ["drug1", "drug2"]}
        perturbation_covariate_reps = {"drug": "drug"}
        condition_embedding_dim = 32

        cf = cfp.model.CellFlow(adata_perturbation, solver=solver)
        cf.prepare_data(
            sample_rep=sample_rep,
            control_key=control_key,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
        )
        assert cf.train_data is not None
        assert hasattr(cf, "_data_dim")

        condition_encoder_kwargs = {}
        if solver == "genot":
            condition_encoder_kwargs["genot_source_layers"] = (({"dims": (32, 32)}),)
            condition_encoder_kwargs["genot_source_dim"] = 32

        cf.prepare_model(
            condition_embedding_dim=condition_embedding_dim,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
            condition_encoder_kwargs=condition_encoder_kwargs,
            time_encoder_batchnorm=time_encoder_batchnorm,
            hidden_batchnorm=hidden_batchnorm,
            decoder_batchnorm=decoder_batchnorm,
        )
        assert cf._trainer is not None

        cf.train(num_iterations=3)
        assert cf._dataloader is not None

        # we assume these are all source cells now in adata_perturbation
        adata_perturbation_pred = adata_perturbation.copy()
        adata_perturbation_pred.obs["control"] = True
        pred = cf.predict(
            adata_perturbation_pred,
            sample_rep=sample_rep,
            covariate_data=adata_perturbation_pred.obs,
        )
        assert isinstance(pred, dict)
        key, out = next(iter(pred.items()))
        assert out.shape[0] == adata_perturbation.n_obs
        assert out.shape[1] == cf._data_dim

        pred_stored = cf.predict(
            adata_perturbation_pred,
            sample_rep=sample_rep,
            covariate_data=adata_perturbation_pred.obs,
            key_added_prefix="MY_PREDICTION_",
        )

        assert pred_stored is None
        if solver == "otfm":
            assert "MY_PREDICTION_" + str(key) in adata_perturbation_pred.obsm

        if solver == "genot":
            assert "MY_PREDICTION_" + str(key) + "_0" in adata_perturbation_pred.obsm
            pred2 = cf.predict(
                adata_perturbation_pred,
                sample_rep=sample_rep,
                covariate_data=adata_perturbation_pred.obs,
                n_samples=2,
            )
            assert isinstance(pred2, dict)
            out = next(iter(pred2.values()))
            assert out.shape[0] == adata_perturbation.n_obs
            assert out.shape[1] == cf._data_dim
            assert out.shape[2] == 2

        conds = adata_perturbation.obs.drop_duplicates(subset=["drug1", "drug2"])
        cond_embed = cf.get_condition_embedding(conds, rep_dict=adata_perturbation.uns)
        assert isinstance(cond_embed, pd.DataFrame)
        assert cond_embed.shape[0] == conds.shape[0]
        assert cond_embed.shape[1] == condition_embedding_dim

    @pytest.mark.parametrize("solver", ["otfm", "genot"])
    @pytest.mark.parametrize("perturbation_covariate_reps", [{}, {"drug": "drug"}])
    def test_cellflow_covar_reps(
        self,
        adata_perturbation,
        perturbation_covariate_reps,
        solver,
    ):
        sample_rep = "X"
        control_key = "control"
        perturbation_covariates = {"drug": ["drug1"]}
        perturbation_covariate_reps = {"drug": "drug"}
        condition_embedding_dim = 32

        cf = cfp.model.CellFlow(adata_perturbation, solver=solver)
        cf.prepare_data(
            sample_rep=sample_rep,
            control_key=control_key,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
        )
        assert cf.train_data is not None
        assert hasattr(cf, "_data_dim")

        condition_encoder_kwargs = {}
        if solver == "genot":
            condition_encoder_kwargs["genot_source_layers"] = (({"dims": (32, 32)}),)
            condition_encoder_kwargs["genot_source_dim"] = 32

        cf.prepare_model(
            condition_embedding_dim=condition_embedding_dim,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
            condition_encoder_kwargs=condition_encoder_kwargs,
        )
        assert cf._trainer is not None

        cf.train(num_iterations=3)
        assert cf._dataloader is not None

        # we assume these are all source cells now in adata_perturbation
        adata_perturbation_pred = adata_perturbation.copy()
        adata_perturbation_pred.obs["control"] = True
        pred = cf.predict(
            adata_perturbation_pred,
            sample_rep=sample_rep,
            covariate_data=adata_perturbation_pred.obs,
        )
        assert isinstance(pred, dict)
        out = next(iter(pred.values()))
        assert out.shape[0] == adata_perturbation.n_obs
        assert out.shape[1] == cf._data_dim

        covs = adata_perturbation.obs.drop_duplicates(subset=["drug1"])
        cond_embed = cf.get_condition_embedding(
            covariate_data=covs, rep_dict=adata_perturbation.uns
        )

        assert isinstance(cond_embed, pd.DataFrame)
        assert cond_embed.shape[0] == len(covs)
        assert cond_embed.shape[1] == condition_embedding_dim

    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize(
        "perturbation_covariates", perturbation_covariate_comb_args
    )
    @pytest.mark.parametrize("solver", ["otfm", "genot"])
    @pytest.mark.parametrize("n_conditions_on_log_iteration", [None, 0, 2])
    @pytest.mark.parametrize("n_conditions_on_train_end", [None, 0, 2])
    def test_cellflow_val_data_loading(
        self,
        adata_perturbation,
        split_covariates,
        perturbation_covariates,
        solver,
        n_conditions_on_log_iteration,
        n_conditions_on_train_end,
    ):
        cf = cfp.model.CellFlow(adata_perturbation, solver=solver)
        cf.prepare_data(
            sample_rep="X",
            control_key="control",
            split_covariates=split_covariates,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps={"drug": "drug"},
        )
        assert cf.train_data is not None
        assert hasattr(cf, "_data_dim")

        for k in perturbation_covariates.keys():
            assert k in cf.train_data.condition_data.keys()
            assert cf.train_data.condition_data[k].ndim == 3
            assert (
                cf.train_data.condition_data[k].shape[1]
                == cf.train_data.max_combination_length
            )
            assert (
                cf.train_data.condition_data[k].shape[0]
                == cf.train_data.n_perturbations
            )

        cf.prepare_validation_data(
            adata_perturbation,
            name="val",
            n_conditions_on_log_iteration=n_conditions_on_log_iteration,
            n_conditions_on_train_end=n_conditions_on_train_end,
        )
        assert isinstance(cf._validation_data, dict)
        assert "val" in cf._validation_data
        assert isinstance(cf._validation_data["val"].cell_data, jax.Array)
        assert isinstance(cf._validation_data["val"].condition_data, dict)

        cond_data = cf._validation_data["val"].condition_data
        assert (
            cf._validation_data["val"].n_conditions_on_log_iteration
            == n_conditions_on_log_iteration
        )
        assert (
            cf._validation_data["val"].n_conditions_on_train_end
            == n_conditions_on_train_end
        )
        for k in perturbation_covariates.keys():
            assert k in cond_data.keys()
            assert cond_data[k].ndim == 3
            assert cond_data[k].shape[1] == cf.train_data.max_combination_length

    @pytest.mark.parametrize("solver", ["otfm", "genot"])
    @pytest.mark.parametrize("n_conditions_on_log_iteration", [None, 0, 1])
    @pytest.mark.parametrize("n_conditions_on_train_end", [None, 0, 1])
    @pytest.mark.parametrize("time_encoder_batchnorm", [True, False])
    @pytest.mark.parametrize("hidden_batchnorm", [True, False])
    @pytest.mark.parametrize("decoder_batchnorm", [True, False])
    def test_cellflow_with_validation(
        self,
        adata_perturbation,
        solver,
        n_conditions_on_log_iteration,
        n_conditions_on_train_end,
        time_encoder_batchnorm,
        hidden_batchnorm,
        decoder_batchnorm,
    ):
        # TODO(@MUCDK) after PR #33 check for larger n_conditions_on...
        cf = cfp.model.CellFlow(adata_perturbation, solver=solver)
        cf.prepare_data(
            sample_rep="X",
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
            perturbation_covariate_reps={"drug": "drug"},
        )
        assert cf.train_data is not None
        assert hasattr(cf, "_data_dim")

        cf.prepare_validation_data(
            adata_perturbation,
            name="val",
            n_conditions_on_log_iteration=n_conditions_on_log_iteration,
            n_conditions_on_train_end=n_conditions_on_train_end,
        )
        assert isinstance(cf._validation_data, dict)
        assert "val" in cf._validation_data
        assert isinstance(cf._validation_data["val"].cell_data, jax.Array)
        assert isinstance(cf._validation_data["val"].condition_data, dict)
        assert (
            cf._validation_data["val"].max_combination_length
            == cf.train_data.max_combination_length
        )
        cond_data = cf._validation_data["val"].condition_data
        assert "drug" in cond_data.keys()
        assert cond_data["drug"].ndim == 3
        assert cond_data["drug"].shape[1] == cf.train_data.max_combination_length

        condition_encoder_kwargs = {}
        if solver == "genot":
            condition_encoder_kwargs["genot_source_layers"] = (({"dims": (32, 32)}),)
            condition_encoder_kwargs["genot_source_dim"] = 32

        cf.prepare_model(
            condition_embedding_dim=32,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
            condition_encoder_kwargs=condition_encoder_kwargs,
            time_encoder_batchnorm=time_encoder_batchnorm,
            hidden_batchnorm=hidden_batchnorm,
            decoder_batchnorm=decoder_batchnorm,
        )
        assert cf._trainer is not None

        metric_to_compute = "r_squared"
        metrics_callback = cfp.training.Metrics(metrics=[metric_to_compute])

        cf.train(num_iterations=3, callbacks=[metrics_callback], valid_freq=1)
        assert cf._dataloader is not None
        assert f"val_{metric_to_compute}_mean" in cf._trainer.training_logs

    @pytest.mark.parametrize("solver", ["otfm", "genot"])
    @pytest.mark.parametrize("time_encoder_batchnorm", [True, False])
    @pytest.mark.parametrize("hidden_batchnorm", [True, False])
    @pytest.mark.parametrize("decoder_batchnorm", [True, False])
    def test_cellflow_predict(
        self,
        adata_perturbation,
        solver,
        time_encoder_batchnorm,
        hidden_batchnorm,
        decoder_batchnorm,
    ):
        cf = cfp.model.CellFlow(adata_perturbation, solver=solver)
        cf.prepare_data(
            sample_rep="X",
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
            perturbation_covariate_reps={"drug": "drug"},
        )

        condition_encoder_kwargs = {}
        if solver == "genot":
            condition_encoder_kwargs["genot_source_layers"] = (({"dims": (32, 32)}),)
            condition_encoder_kwargs["genot_source_dim"] = 32

        cf.prepare_model(
            condition_embedding_dim=32,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
            condition_encoder_kwargs=condition_encoder_kwargs,
            time_encoder_batchnorm=time_encoder_batchnorm,
            hidden_batchnorm=hidden_batchnorm,
            decoder_batchnorm=decoder_batchnorm,
        )

        cf.train(num_iterations=3)

        adata_pred = adata_perturbation[:100].copy()
        adata_pred.obs["control"] = True
        covariate_data = adata_perturbation.obs.iloc[:3]

        pred = cf.predict(adata_pred, sample_rep="X", covariate_data=covariate_data)

        assert isinstance(pred, dict)
        out = next(iter(pred.values()))
        assert out.shape[0] == 100
        assert out.shape[1] == cf._data_dim

        adata_pred.obs["control"].iloc[0:20] = False
        with pytest.raises(
            ValueError,
            match=r".*If both `adata` and `covariate_data` are given, all samples in `adata` must be control samples*",
        ):
            cf.predict(adata_pred, sample_rep="X", covariate_data=covariate_data)

    def test_raise_otfm_genot_layers_passed(self, adata_perturbation):
        cf = cfp.model.CellFlow(adata_perturbation, solver="otfm")
        cf.prepare_data(
            sample_rep="X",
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
            perturbation_covariate_reps={"drug": "drug"},
        )
        with pytest.raises(
            ValueError,
            match=r".*For OTFlowMatching, 'genot_source_layers' must be `None`.'*",
        ):
            cf.prepare_model(
                condition_embedding_dim=32,
                hidden_dims=(32, 32),
                decoder_dims=(32, 32),
                genot_source_layers=({"layer_type": "mlp", "dims": (32, 32)},),
            )

    @pytest.mark.parametrize(
        "sample_covariate_and_reps",
        [(None, None), (["cell_type"], {"cell_type": "cell_type"})],
    )
    @pytest.mark.parametrize("split_covariates", [None, ["cell_type"]])
    @pytest.mark.parametrize(
        "perturbation_covariates", perturbation_covariate_comb_args
    )
    def test_cellflow_get_condition_embedding(
        self,
        adata_perturbation,
        sample_covariate_and_reps,
        split_covariates,
        perturbation_covariates,
    ):
        sample_rep = "X"
        control_key = "control"
        perturbation_covariate_reps = {"drug": "drug"}
        sample_covariates = sample_covariate_and_reps[0]
        sample_covariate_reps = sample_covariate_and_reps[1]
        condition_embedding_dim = 32
        solver = "otfm"

        cf = cfp.model.CellFlow(adata_perturbation, solver=solver)
        cf.prepare_data(
            sample_rep=sample_rep,
            control_key=control_key,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=sample_covariates,
            sample_covariate_reps=sample_covariate_reps,
        )
        assert cf.train_data is not None
        assert isinstance(cf._dm.perturb_covar_keys, list)
        assert hasattr(cf, "_data_dim")

        cf.prepare_model(
            condition_embedding_dim=condition_embedding_dim,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        assert cf._trainer is not None

        cf.train(num_iterations=3)
        assert cf._dataloader is not None

        conds = adata_perturbation.obs.drop_duplicates(subset=cf._dm.perturb_covar_keys)
        cond_embed = cf.get_condition_embedding(conds, rep_dict=adata_perturbation.uns)
        assert isinstance(cond_embed, pd.DataFrame)
        assert cond_embed.shape[0] == conds.shape[0]
        assert cond_embed.shape[1] == condition_embedding_dim

        # Test if condition_id_key works
        condition_id_key = "condition_id"
        conds[condition_id_key] = range(len(conds))
        conds[condition_id_key] = "cond_" + conds[condition_id_key].astype(str)
        cond_embed = cf.get_condition_embedding(
            conds, rep_dict=adata_perturbation.uns, condition_id_key=condition_id_key
        )
        assert isinstance(cond_embed, pd.DataFrame)
        assert cond_embed.shape[0] == conds.shape[0]
        assert cond_embed.index.name == condition_id_key
        cond_id_vals = conds[condition_id_key].values
        assert cond_embed.index.isin(cond_id_vals).all()
