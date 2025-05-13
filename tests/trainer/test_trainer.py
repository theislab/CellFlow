import time

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from ott.neural.methods.flows import dynamics

import cellflow
from cellflow.solvers import _otfm
from cellflow.utils import match_linear

x_test = jnp.ones((10, 5)) * 10
t_test = jnp.ones((10, 1))
cond = {"pert1": jnp.ones((1, 2, 3))}
vf_rng = jax.random.PRNGKey(111)


class TestTrainer:
    @pytest.mark.parametrize("valid_freq", [10, 1])
    def test_cellflow_trainer(self, dataloader, valid_freq):
        opt = optax.adam(1e-3)
        vf = cellflow.networks.ConditionalVelocityField(
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        model = _otfm.OTFlowMatching(
            vf=vf,
            match_fn=match_linear,
            probability_path=dynamics.ConstantNoiseFlow(0.0),
            optimizer=opt,
            conditions=cond,
            rng=vf_rng,
        )

        trainer = cellflow.training.CellFlowTrainer(solver=model)
        trainer.train(
            dataloader=dataloader,
            num_iterations=2,
            valid_freq=valid_freq,
        )
        x_pred = model.predict(x_test, cond)
        assert x_pred.shape == x_test.shape

        out = model.get_condition_embedding(cond)
        assert isinstance(out, tuple)
        assert isinstance(out[0], np.ndarray)
        assert isinstance(out[1], np.ndarray)
        assert out[0].shape == (1, 12)
        assert out[1].shape == (1, 12)

    @pytest.mark.parametrize("use_validdata", [True, False])
    def test_cellflow_trainer_with_callback(self, dataloader, valid_loader, use_validdata):
        opt = optax.adam(1e-3)
        vf = cellflow.networks.ConditionalVelocityField(
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        model = _otfm.OTFlowMatching(
            vf=vf,
            match_fn=match_linear,
            probability_path=dynamics.ConstantNoiseFlow(0.0),
            optimizer=opt,
            conditions=cond,
            rng=vf_rng,
        )

        metric_to_compute = "e_distance"
        metrics_callback = cellflow.training.Metrics(metrics=[metric_to_compute])

        trainer = cellflow.training.CellFlowTrainer(solver=model)
        trainer.train(
            dataloader=dataloader,
            valid_loaders=valid_loader if use_validdata else None,
            num_iterations=2,
            valid_freq=1,
            callbacks=[metrics_callback],
        )

        assert "loss" in trainer.training_logs
        if use_validdata:
            assert f"val_{metric_to_compute}_mean" in trainer.training_logs

        x_pred = model.predict(x_test, cond)
        assert x_pred.shape == x_test.shape

        out = model.get_condition_embedding(cond)
        assert isinstance(out, tuple)
        assert isinstance(out[0], np.ndarray)
        assert out[0].shape == (1, 12)
        assert isinstance(out[1], np.ndarray)
        assert out[1].shape == (1, 12)

    def test_predict_kwargs_iter(self, dataloader, valid_loader):
        opt_1 = optax.adam(1e-3)
        opt_2 = optax.adam(1e-3)
        vf_1 = cellflow.networks.ConditionalVelocityField(
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        vf_2 = cellflow.networks.ConditionalVelocityField(
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        model_1 = _otfm.OTFlowMatching(
            vf=vf_1,
            match_fn=match_linear,
            probability_path=dynamics.ConstantNoiseFlow(0.0),
            optimizer=opt_1,
            conditions=cond,
            rng=vf_rng,
        )
        model_2 = _otfm.OTFlowMatching(
            vf=vf_2,
            match_fn=match_linear,
            probability_path=dynamics.ConstantNoiseFlow(0.0),
            optimizer=opt_2,
            conditions=cond,
            rng=vf_rng,
        )

        metric_to_compute = "e_distance"
        metrics_callback = cellflow.training.Metrics(metrics=[metric_to_compute])

        predict_kwargs_1 = {"stepsize_controller": diffrax.PIDController(rtol=0.1, atol=0.1)}
        predict_kwargs_2 = {"stepsize_controller": diffrax.PIDController(rtol=1e-5, atol=1e-5)}

        trainer_1 = cellflow.training.CellFlowTrainer(solver=model_1, predict_kwargs=predict_kwargs_1)
        trainer_2 = cellflow.training.CellFlowTrainer(solver=model_2, predict_kwargs=predict_kwargs_2)

        start_1 = time.time()
        trainer_1.train(
            dataloader=dataloader,
            valid_loaders=valid_loader,
            num_iterations=2,
            valid_freq=1,
            callbacks=[metrics_callback],
        )
        end_1 = time.time()
        diff_1 = end_1 - start_1

        start_2 = time.time()
        trainer_2.train(
            dataloader=dataloader,
            valid_loaders=valid_loader,
            num_iterations=2,
            valid_freq=1,
            callbacks=[metrics_callback],
        )
        end_2 = time.time()
        diff_2 = end_2 - start_2

        assert diff_2 - diff_1 > 5
