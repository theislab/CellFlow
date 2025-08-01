import functools
import time

import jax
import numpy as np
import optax
import pytest
from ott.neural.methods.flows import dynamics

import cellflow
from cellflow.solvers import _genot, _otfm
from cellflow.utils import match_linear

src = {
    ("drug_1",): np.random.rand(10, 5),
    ("drug_2",): np.random.rand(10, 5),
}
cond = {
    ("drug_1",): {"drug": np.random.rand(1, 1, 3)},
    ("drug_2",): {"drug": np.random.rand(1, 1, 3)},
}
vf_rng = jax.random.PRNGKey(111)


class TestSolver:
    @pytest.mark.parametrize("solver_class", ["otfm", "genot"])
    def test_predict_batch(self, dataloader, solver_class):
        if solver_class == "otfm":
            vf_class = cellflow.networks.ConditionalVelocityField
        else:
            vf_class = cellflow.networks.GENOTConditionalVelocityField

        opt = optax.adam(1e-3)
        vf = vf_class(
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        if solver_class == "otfm":
            solver = _otfm.OTFlowMatching(
                vf=vf,
                match_fn=match_linear,
                probability_path=dynamics.ConstantNoiseFlow(0.0),
                optimizer=opt,
                conditions={"drug": np.random.rand(2, 1, 3)},
                rng=vf_rng,
            )
        else:
            solver = _genot.GENOT(
                vf=vf,
                data_match_fn=match_linear,
                probability_path=dynamics.ConstantNoiseFlow(0.0),
                optimizer=opt,
                source_dim=5,
                target_dim=5,
                conditions={"drug": np.random.rand(2, 1, 3)},
                rng=vf_rng,
            )

        predict_kwargs = {"max_steps": 3, "throw": False}
        trainer = cellflow.training.CellFlowTrainer(solver=solver, predict_kwargs=predict_kwargs)
        trainer.train(
            dataloader=dataloader,
            num_iterations=2,
            valid_freq=1,
        )
        start_batched = time.time()
        x_pred_batched = solver.predict(src, cond, batched=True)
        end_batched = time.time()
        diff_batched = end_batched - start_batched

        start_nonbatched = time.time()
        x_pred_nonbatched = jax.tree.map(
            functools.partial(solver.predict, batched=False),
            src,
            cond,  # type: ignore[attr-defined]
        )
        end_nonbatched = time.time()
        diff_nonbatched = end_nonbatched - start_nonbatched

        assert x_pred_batched[("drug_1",)].shape == x_pred_nonbatched[("drug_1",)].shape
        assert np.allclose(
            x_pred_batched[("drug_1",)],
            x_pred_nonbatched[("drug_1",)],
            atol=1e-1,
            rtol=1e-2,
        )
        assert diff_nonbatched - diff_batched > 2

    @pytest.mark.parametrize("ema", [0.5, 1.0])
    def test_EMA(self, dataloader, ema):
        vf_class = cellflow.networks.ConditionalVelocityField
        drug = np.random.rand(2, 1, 3)
        opt = optax.adam(1e-3)
        vf1 = vf_class(
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=(6, 6),
            decoder_dims=(5, 5),
        )

        solver1 = _otfm.OTFlowMatching(
            vf=vf1,
            match_fn=match_linear,
            probability_path=dynamics.ConstantNoiseFlow(0.0),
            optimizer=opt,
            conditions={"drug": drug},
            rng=vf_rng,
            ema=ema,
        )
        trainer1 = cellflow.training.CellFlowTrainer(solver=solver1)
        trainer1.train(
            dataloader=dataloader,
            num_iterations=5,
            valid_freq=10,
        )

        if ema == 1.0:
            assert jax.tree.all(
                jax.tree.map(
                    lambda x, y: np.allclose(x, y, atol=1e-5, rtol=1e-2),
                    solver1.vf_state_inference.params,
                    solver1.vf_state.params,
                )
            )
        else:
            assert not jax.tree.all(
                jax.tree.map(
                    lambda x, y: np.allclose(x, y, atol=1e-5),
                    solver1.vf_state_inference.params,
                    solver1.vf_state.params,
                )
            )
