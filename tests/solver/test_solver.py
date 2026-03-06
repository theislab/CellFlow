import jax
import numpy as np
import optax
import pytest

import cellflow
from cellflow._compat import ConstantNoiseFlow
from cellflow.solvers import _otfm
from cellflow.utils import match_linear

vf_rng = jax.random.PRNGKey(111)


class TestSolver:
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
            probability_path=ConstantNoiseFlow(0.0),
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
