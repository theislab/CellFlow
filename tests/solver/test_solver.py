import functools
import time

import jax
import numpy as np
import optax
from ott.neural.methods.flows import dynamics

import cellflow
from cellflow.solvers import _otfm
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
    def test_predict_batch(self, dataloader, valid_loader):
        opt = optax.adam(1e-3)
        vf = cellflow.networks.ConditionalVelocityField(
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=(32, 32),
            decoder_dims=(32, 32),
        )
        solver = _otfm.OTFlowMatching(
            vf=vf,
            match_fn=match_linear,
            probability_path=dynamics.ConstantNoiseFlow(0.0),
            optimizer=opt,
            conditions={"drug": np.random.rand(2, 1, 3)},
            rng=vf_rng,
        )

        trainer = cellflow.training.CellFlowTrainer(solver=solver)
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
