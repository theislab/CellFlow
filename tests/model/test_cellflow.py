import jax
import pandas as pd
import pytest

import cellflow
from cellflow.networks import _velocity_field

perturbation_covariate_comb_args = [
    {"drug": ["drug1"]},
    {"drug": ["drug1", "drug2"], "dosage": ["dosage_a", "dosage_b"]},
    {
        "drug": ["drug_a", "drug_b", "drug_c"],
        "dosage": ["dosage_a", "dosage_b", "dosage_c"],
    },
]


class TestCellFlow:
    @pytest.mark.skip(reason="temporarily disabled")
    @pytest.mark.parametrize("solver", ["otfm"])  # , "genot"])
    @pytest.mark.parametrize("condition_mode", ["deterministic", "stochastic"])
    @pytest.mark.parametrize("regularization", [0.0, 0.1])
    @pytest.mark.parametrize("conditioning", ["concatenation", "film", "resnet"])
    def test_cellflow_solver(
        self,
        adata_perturbation,
        solver,
        condition_mode,
        regularization,
        conditioning,
    ):
        if solver == "genot" and ((condition_mode == "stochastic") or (regularization > 0.0)):
            return None
        sample_rep = "X"
        control_key = "control"
        perturbation_covariates = {"drug": ["drug1", "drug2"]}
        perturbation_covariate_reps = {"drug": "drug"}
        condition_embedding_dim = 32
        vf_kwargs = {"genot_source_dims": (32, 32), "genot_source_dropout": 0.1} if solver == "genot" else None

        cf = cellflow.model.CellFlow(adata_perturbation, solver=solver)
        cf.prepare_data(
            sample_rep=sample_rep,
            control_key=control_key,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
        )
        assert cf.train_data is not None
        assert hasattr(cf, "_data_dim")

        if regularization == 0.0 and condition_mode == "stochastic":
            with pytest.raises(
                ValueError,
                match=r".*Stochastic condition embeddings require `regularization`>0*",
            ):
                cf.prepare_model(
                    condition_mode=condition_mode,
                    regularization=regularization,
                    condition_embedding_dim=condition_embedding_dim,
                    hidden_dims=(32, 32),
                    decoder_dims=(32, 32),
                    vf_kwargs=vf_kwargs,
                    conditioning=conditioning,
                )
            return None
        cf.prepare_model(
            condition_mode=condition_mode,
            regularization=regularization,
            condition_embedding_dim=condition_embedding_dim,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
            vf_kwargs=vf_kwargs,
            conditioning=conditioning,
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
            max_steps=3,
            throw=False,
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
            max_steps=3,
            throw=False,
        )

        assert pred_stored is None
        if solver == "otfm":
            assert "MY_PREDICTION_" + str(key) in adata_perturbation_pred.obsm

        if solver == "genot":
            assert "MY_PREDICTION_" + str(key) in adata_perturbation_pred.obsm
            pred2 = cf.predict(
                adata_perturbation_pred,
                sample_rep=sample_rep,
                covariate_data=adata_perturbation_pred.obs,
                max_steps=3,
                throw=False,
            )
            assert isinstance(pred2, dict)
            out = next(iter(pred2.values()))
            assert out.shape[0] == adata_perturbation.n_obs
            assert out.shape[1] == cf._data_dim

        conds = adata_perturbation.obs.drop_duplicates(subset=["drug1", "drug2"])
        cond_embed_mean, cond_embed_var = cf.get_condition_embedding(conds, rep_dict=adata_perturbation.uns)
        assert isinstance(cond_embed_mean, pd.DataFrame)
        assert isinstance(cond_embed_var, pd.DataFrame)
        assert cond_embed_mean.shape[0] == conds.shape[0]
        assert cond_embed_mean.shape[1] == condition_embedding_dim
        assert cond_embed_var.shape[0] == conds.shape[0]
        assert cond_embed_var.shape[1] == condition_embedding_dim
        assert cond_embed_mean.shape[0] == conds.shape[0]
        assert cond_embed_mean.shape[1] == condition_embedding_dim
        assert cond_embed_var.shape[0] == conds.shape[0]
        assert cond_embed_var.shape[1] == condition_embedding_dim

    @pytest.mark.skip(reason="temporarily disabled")
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
        vf_kwargs = {"genot_source_dims": (32, 32), "genot_source_dropout": 0.1} if solver == "genot" else None

        cf = cellflow.model.CellFlow(adata_perturbation, solver=solver)
        cf.prepare_data(
            sample_rep=sample_rep,
            control_key=control_key,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
        )
        assert cf.train_data is not None
        assert hasattr(cf, "_data_dim")

        cf.prepare_model(
            condition_embedding_dim=condition_embedding_dim,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
            vf_kwargs=vf_kwargs,
        )
        assert cf._trainer is not None

        vector_field_class = (
            _velocity_field.ConditionalVelocityField
            if solver == "otfm"
            else _velocity_field.GENOTConditionalVelocityField
        )
        assert cf._vf_class == vector_field_class
        cf.train(num_iterations=3)
        assert cf._dataloader is not None

        # we assume these are all source cells now in adata_perturbation
        adata_perturbation_pred = adata_perturbation.copy()
        adata_perturbation_pred.obs["control"] = True
        pred = cf.predict(
            adata_perturbation_pred,
            sample_rep=sample_rep,
            covariate_data=adata_perturbation_pred.obs,
            max_steps=3,
            throw=False,
        )
        assert isinstance(pred, dict)
        out = next(iter(pred.values()))
        assert out.shape[0] == adata_perturbation.n_obs
        assert out.shape[1] == cf._data_dim

        covs = adata_perturbation.obs.drop_duplicates(subset=["drug1"])
        out = cf.get_condition_embedding(covariate_data=covs, rep_dict=adata_perturbation.uns)
        assert isinstance(out, tuple)
        assert isinstance(out[0], pd.DataFrame)
        assert isinstance(out[1], pd.DataFrame)
        assert out[0].shape[0] == len(covs)
        assert out[0].shape[1] == condition_embedding_dim

    @pytest.mark.skip(reason="temporarily disabled")
    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize("perturbation_covariates", perturbation_covariate_comb_args)
    @pytest.mark.parametrize("n_conditions_on_log_iteration", [None, 0, 2])
    @pytest.mark.parametrize("n_conditions_on_train_end", [None, 0, 2])
    def test_cellflow_val_data_loading(
        self,
        adata_perturbation,
        split_covariates,
        perturbation_covariates,
        n_conditions_on_log_iteration,
        n_conditions_on_train_end,
    ):
        cf = cellflow.model.CellFlow(adata_perturbation)
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
            assert cf.train_data.condition_data[k].shape[1] == cf.train_data.max_combination_length
            assert cf.train_data.condition_data[k].shape[0] == cf.train_data.n_perturbations

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
        assert cf._validation_data["val"].n_conditions_on_log_iteration == n_conditions_on_log_iteration
        assert cf._validation_data["val"].n_conditions_on_train_end == n_conditions_on_train_end
        for k in perturbation_covariates.keys():
            assert k in cond_data.keys()
            assert cond_data[k].ndim == 3
            assert cond_data[k].shape[1] == cf.train_data.max_combination_length

    @pytest.mark.skip()
    @pytest.mark.parametrize("solver", ["otfm", "genot"])
    @pytest.mark.parametrize("n_conditions_on_log_iteration", [None, 0, 1])
    @pytest.mark.parametrize("n_conditions_on_train_end", [None, 0, 1])
    def test_cellflow_with_validation(
        self,
        adata_perturbation,
        solver,
        n_conditions_on_log_iteration,
        n_conditions_on_train_end,
    ):
        vf_kwargs = {"genot_source_dims": (32, 32), "genot_source_dropout": 0.1} if solver == "genot" else None
        cf = cellflow.model.CellFlow(adata_perturbation, solver=solver)
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
        assert cf._validation_data["val"].max_combination_length == cf.train_data.max_combination_length
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
            vf_kwargs=vf_kwargs,
        )
        assert cf._trainer is not None

        metric_to_compute = "r_squared"
        metrics_callback = cellflow.training.Metrics(metrics=[metric_to_compute])

        cf.train(num_iterations=3, callbacks=[metrics_callback], valid_freq=1)
        assert cf._dataloader is not None
        assert f"val_{metric_to_compute}_mean" in cf._trainer.training_logs

    @pytest.mark.skip()
    @pytest.mark.parametrize("solver", ["otfm", "genot"])
    @pytest.mark.parametrize("condition_mode", ["deterministic", "stochastic"])
    @pytest.mark.parametrize("regularization", [0.0, 0.1])
    def test_cellflow_predict(
        self,
        adata_perturbation,
        solver,
        condition_mode,
        regularization,
    ):
        cf = cellflow.model.CellFlow(adata_perturbation, solver=solver)
        cf.prepare_data(
            sample_rep="X",
            control_key="control",
            perturbation_covariates={"drug": ["drug1"], "cell_type": ["cell_type"]},
            perturbation_covariate_reps={"drug": "drug", "cell_type": "cell_type"},
            split_covariates=["cell_type"],
        )

        vf_kwargs = {"genot_source_dims": (32, 32), "genot_source_dropout": 0.1} if solver == "genot" else None
        if condition_mode == "stochastic" and regularization == 0.0:
            with pytest.raises(
                ValueError,
                match=r".*Stochastic condition embeddings require `regularization`>0*",
            ):
                cf.prepare_model(
                    condition_mode=condition_mode,
                    regularization=regularization,
                    hidden_dims=(32, 32),
                    decoder_dims=(32, 32),
                )
            return None
        cf.prepare_model(
            condition_mode=condition_mode,
            regularization=regularization,
            condition_embedding_dim=32,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
            vf_kwargs=vf_kwargs,
        )

        cf.train(num_iterations=3)

        adata_pred = adata_perturbation
        adata_pred.obs["control"] = True
        covariate_data = adata_perturbation.obs.iloc[:3]

        pred = cf.predict(adata_pred, sample_rep="X", covariate_data=covariate_data, max_steps=3, throw=False)

        assert isinstance(pred, dict)
        out = next(iter(pred.values()))
        assert out.shape[1] == cf._data_dim

        adata_pred.obs["control"].iloc[0:20] = False
        with pytest.raises(
            ValueError,
            match=r".*If both `adata` and `covariate_data` are given, all samples in `adata` must be control samples*",
        ):
            cf.predict(adata_pred, sample_rep="X", covariate_data=covariate_data, max_steps=3, throw=False)

        with pytest.raises(ValueError, match="`covariate_data` is empty."):
            empty_covariate_data = covariate_data.head(0)
            cf.predict(adata_pred, sample_rep="X", covariate_data=empty_covariate_data, max_steps=3, throw=False)

        with pytest.raises(
            ValueError,
            match=r".*No cells found in `adata` for split covariates*",
        ):
            covariate_data_a = covariate_data.copy()
            covariate_data_a["cell_type"][0] = "cell_line_a"
            cov_data_cell_type_1 = covariate_data_a[covariate_data_a["cell_type"] == "cell_line_a"]
            adata_pred_cell_type_2 = adata_pred[adata_pred.obs["cell_type"] == "cell_line_b"]
            adata_pred_cell_type_2.obs["control"] = True
            cf.predict(
                adata_pred_cell_type_2, sample_rep="X", covariate_data=cov_data_cell_type_1, max_steps=3, throw=False
            )

    @pytest.mark.skip()
    def test_raise_otfm_vf_kwargs_passed(self, adata_perturbation):
        vf_kwargs = {"genot_source_dims": (32, 32), "genot_source_dropouts": 0.1}
        cf = cellflow.model.CellFlow(adata_perturbation, solver="otfm")
        cf.prepare_data(
            sample_rep="X",
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
            perturbation_covariate_reps={"drug": "drug"},
        )
        with pytest.raises(
            ValueError,
            match=r".*For `solver='otfm'`, `vf_kwargs` must be `None`.*",
        ):
            cf.prepare_model(
                condition_embedding_dim=32,
                hidden_dims=(32, 32),
                decoder_dims=(32, 32),
                vf_kwargs=vf_kwargs,
            )

    @pytest.mark.skip()
    @pytest.mark.parametrize(
        "sample_covariate_and_reps",
        [(None, None), (["cell_type"], {"cell_type": "cell_type"})],
    )
    @pytest.mark.parametrize("split_covariates", [None, ["cell_type"]])
    @pytest.mark.parametrize("perturbation_covariates", perturbation_covariate_comb_args)
    @pytest.mark.parametrize("condition_mode", ["deterministic", "stochastic"])
    @pytest.mark.parametrize("regularization", [0.0, 0.1])
    def test_cellflow_get_condition_embedding(
        self,
        adata_perturbation,
        sample_covariate_and_reps,
        condition_mode,
        regularization,
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

        cf = cellflow.model.CellFlow(adata_perturbation, solver=solver)
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

        if condition_mode == "stochastic" and regularization == 0.0:
            with pytest.raises(
                ValueError,
                match=r".*Stochastic condition embeddings require `regularization`>0*",
            ):
                cf.prepare_model(
                    condition_mode=condition_mode,
                    regularization=regularization,
                    condition_embedding_dim=condition_embedding_dim,
                    hidden_dims=(32, 32),
                    decoder_dims=(32, 32),
                )
            return None
        cf.prepare_model(
            condition_mode=condition_mode,
            regularization=regularization,
            condition_embedding_dim=condition_embedding_dim,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        assert cf._trainer is not None

        cf.train(num_iterations=3)
        assert cf._dataloader is not None

        conds = adata_perturbation.obs.drop_duplicates(subset=cf._dm.perturb_covar_keys)
        out = cf.get_condition_embedding(conds, rep_dict=adata_perturbation.uns)
        assert isinstance(out, tuple)
        assert isinstance(out[0], pd.DataFrame)
        assert isinstance(out[1], pd.DataFrame)
        assert out[0].shape[0] == conds.shape[0]
        assert out[0].shape[1] == condition_embedding_dim

        # Test if condition_id_key works
        condition_id_key = "condition_id"
        conds[condition_id_key] = range(len(conds))
        conds[condition_id_key] = "cond_" + conds[condition_id_key].astype(str)
        out = cf.get_condition_embedding(conds, rep_dict=adata_perturbation.uns, condition_id_key=condition_id_key)
        assert isinstance(out, tuple)
        assert isinstance(out[0], pd.DataFrame)
        assert out[0].shape[0] == out[0].shape[0]
        assert out[0].index.name == condition_id_key
        cond_id_vals = conds[condition_id_key].values
        assert out[0].index.isin(cond_id_vals).all()
        assert isinstance(out[1], pd.DataFrame)
        assert out[1].shape[0] == out[1].shape[0]
        assert out[1].index.name == condition_id_key
        cond_id_vals = conds[condition_id_key].values
        assert out[1].index.isin(cond_id_vals).all()
