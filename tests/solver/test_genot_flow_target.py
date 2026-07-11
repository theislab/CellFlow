"""Regression test for the GENOT flow-matching regression target.

GENOT (Klein et al., 2023) learns a flow from a latent (noise) sample to the target,
conditioned on the source. The flow-matching loss regresses the velocity field onto
the velocity of that path, ``target - latent`` -- see the author's implementation
(https://github.com/MUCDK/genot, ``genot/models/model.py``)::

    phi_t(x_0, x_1, t) = (1 - t) * x_0 + t * x_1
    u_t(x_0, x_1)      = x_1 - x_0
    d_psi = u_t(batch["noise"], batch["target"])   # target - noise
    loss  = mean(l2_loss(v_t(phi_t(noise, target) | source), d_psi))

i.e. both the interpolation and ``u_t`` use ``(noise/latent, target)``; the source is
only the velocity field's condition. Regressing onto ``target - source`` instead is a
silent bug: it biases every generated sample by ``latent - source`` (no crash, since
source and latent share the target dimensionality).

This test traces the actual values passed to ``compute_xt``/``compute_ut`` during a
real training step and asserts ``compute_ut`` receives the latent, not the source.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

import cellflow
from cellflow._compat import ConstantNoiseFlow
from cellflow.solvers import _genot
from cellflow.utils import match_linear


def test_genot_flow_matching_target_uses_latent_not_source():
    vf = cellflow.networks.GENOTConditionalVelocityField(
        output_dim=5,
        max_combination_length=2,
        condition_embedding_dim=12,
        hidden_dims=(16, 16),
        decoder_dims=(16, 16),
        genot_source_dims=(16, 16),
    )
    solver = _genot.GENOT(
        vf=vf,
        data_match_fn=match_linear,
        probability_path=ConstantNoiseFlow(0.0),
        optimizer=optax.adam(1e-3),
        source_dim=5,
        target_dim=5,
        conditions={"pert1": np.random.rand(2, 1, 3)},
        rng=jax.random.PRNGKey(0),
    )

    # Record the x0 argument each primitive receives. Source cells are ~+100 and the
    # latent is ~N(0, 1), so the x0 that reaches compute_ut is unambiguous by scale.
    rec: dict[str, np.ndarray] = {}
    flow = solver.probability_path
    orig_xt, orig_ut = flow.compute_xt, flow.compute_ut
    flow.compute_xt = lambda rng, t, x0, x1: (rec.__setitem__("xt_x0", x0), orig_xt(rng, t, x0, x1))[1]
    flow.compute_ut = lambda t, x, x0, x1: (rec.__setitem__("ut_x0", x0), orig_ut(t, x, x0, x1))[1]

    batch = {
        "src_cell_data": jnp.ones((10, 5)) * 100.0,  # source cells ~ +100
        "tgt_cell_data": jnp.ones((10, 5)) * -100.0,  # target ~ -100
        "condition": {"pert1": jnp.ones((1, 2, 3))},
    }

    with jax.disable_jit():
        solver.step_fn(jax.random.PRNGKey(1), batch)

    xt_x0 = np.asarray(rec["xt_x0"])  # the latent the ODE starts from
    ut_x0 = np.asarray(rec["ut_x0"])  # the regression-target path start

    # compute_ut must use the same path start as compute_xt (the latent), matching the
    # author's `u_t(batch["noise"], batch["target"])`. On the bug it is the source (~+100).
    assert np.allclose(ut_x0, xt_x0), (
        f"compute_ut regresses onto the wrong path start: got mean {ut_x0.mean():+.1f} "
        f"(source) but expected the latent {xt_x0.mean():+.1f}."
    )
