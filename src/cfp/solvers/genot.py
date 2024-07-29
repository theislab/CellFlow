import functools
from collections.abc import Callable
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
from ott import utils
from ott.neural.methods.flows import dynamics
from ott.neural.networks import velocity_field
from ott.solvers import utils as solver_utils

from cfp._constants import GENOT_CELL_KEY
from cfp._types import ArrayLike
from cfp.model._utils import _multivariate_normal

__all__ = ["GENOT"]

LinTerm = tuple[jnp.ndarray, jnp.ndarray]
QuadTerm = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None]
DataMatchFn = Callable[[LinTerm], jnp.ndarray] | Callable[[QuadTerm], jnp.ndarray]


class GENOT:
    """Generative Entropic Neural Optimal Transport :cite:`klein_uscidda:23`.

    GENOT is a framework for learning neural optimal transport plans between
    two distributions. It allows for learning linear and quadratic
    (Fused) Gromov-Wasserstein couplings, in both the balanced and
    the unbalanced setting.

    Args:
      vf: Vector field parameterized by a neural network.
      flow: Flow between the latent and the target distributions.
      data_match_fn: Function to match samples from the source and the target
        distributions. Depending on the data passed in :meth:`__call__`, it has
        the following signature:

        - ``(src_lin, tgt_lin) -> matching`` - linear matching.
        - ``(src_quad, tgt_quad, src_lin, tgt_lin) -> matching`` -
          quadratic (fused) GW matching. In the pure GW setting, both ``src_lin``
          and ``tgt_lin`` will be set to :obj:`None`.

      source_dim: Dimensionality of the source distribution.
      target_dim: Dimensionality of the target distribution.
      condition_dim: Dimension of the conditions. If :obj:`None`, the underlying
        velocity field has no conditions.
      time_sampler: Time sampler with a ``(rng, n_samples) -> time`` signature.
      latent_noise_fn: Function to sample from the latent distribution in the
        target space with a ``(rng, shape) -> noise`` signature.
        If :obj:`None`, multivariate normal distribution is used.
      n_samples_per_src: Number of samples drawn from the conditional distribution
        per one source sample.
      kwargs: Keyword arguments for
        :meth:`~ott.neural.networks.velocity_field.VelocityField.create_train_state`.
    """

    def __init__(
        self,
        vf: velocity_field.VelocityField,
        flow: dynamics.BaseFlow,
        data_match_fn: DataMatchFn,
        *,
        source_dim: int,
        target_dim: int,
        condition_dim: int | None = None,
        time_sampler: Callable[
            [jax.Array, int], jnp.ndarray
        ] = solver_utils.uniform_sampler,
        latent_noise_fn: (
            Callable[[jax.Array, tuple[int, ...]], jnp.ndarray] | None
        ) = None,
        **kwargs: Any,
    ):
        self.vf = vf
        self.flow = flow
        self.data_match_fn = jax.jit(data_match_fn)
        self.time_sampler = time_sampler
        if latent_noise_fn is None:
            latent_noise_fn = functools.partial(_multivariate_normal, dim=target_dim)
        self.latent_noise_fn = latent_noise_fn

        self.vf_state = self.vf.create_train_state(
            input_dim=target_dim,
            additional_cond_dim=source_dim,
            **kwargs,
        )
        self.step_fn = self._get_step_fn()

    def _get_step_fn(self) -> Callable:

        @jax.jit
        def step_fn(
            rng: jax.Array,
            vf_state: train_state.TrainState,
            time: jnp.ndarray,
            source: jnp.ndarray,
            target: jnp.ndarray,
            latent: jnp.ndarray,
            conditions: dict[str, jnp.ndarray] | None,
        ):

            def loss_fn(
                params: jnp.ndarray,
                time: jnp.ndarray,
                source: jnp.ndarray,
                target: jnp.ndarray,
                latent: jnp.ndarray,
                condition: dict[str, jnp.ndarray] | None,
                rng: jax.Array,
            ) -> jnp.ndarray:
                rng_flow, rng_dropout = jax.random.split(rng, 2)
                x_t = self.flow.compute_xt(rng_flow, time, latent, target)
                if condition is not None:
                    condition[GENOT_CELL_KEY] = source
                else:
                    condition = {GENOT_CELL_KEY: source}

                v_t = vf_state.apply_fn(
                    {"params": params},
                    time,
                    x_t,
                    condition,
                    rngs={"dropout": rng_dropout},
                )
                u_t = self.flow.compute_ut(time, latent, target)

                return jnp.mean((v_t - u_t) ** 2)

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(
                vf_state.params, time, source, target, latent, conditions, rng
            )

            return loss, vf_state.apply_gradients(grads=grads)

        return step_fn

    @staticmethod
    def _prepare_data(batch: dict[str, jnp.ndarray]) -> tuple[
        tuple[jnp.ndarray, jnp.ndarray | None, jnp.ndarray],
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None],
    ]:
        src_lin, src_quad = batch.get("src_cell_data"), batch.get("src_cell_data_quad")
        tgt_lin, tgt_quad = batch.get("tgt_cell_data"), batch.get("tgt_cell_data_quad")

        if src_quad is None and tgt_quad is None:  # lin
            src, tgt = src_lin, tgt_lin
            arrs = src_lin, tgt_lin
        elif src_lin is None and tgt_lin is None:  # quad
            src, tgt = src_quad, tgt_quad
            arrs = src_quad, tgt_quad
        elif all(
            arr is not None for arr in (src_lin, tgt_lin, src_quad, tgt_quad)
        ):  # fused quad
            src = jnp.concatenate([src_lin, src_quad], axis=1)
            tgt = jnp.concatenate([tgt_lin, tgt_quad], axis=1)
            arrs = src_quad, tgt_quad, src_lin, tgt_lin
        else:
            raise RuntimeError("Cannot infer OT problem type from data.")

        return (src, tgt), arrs

    def outer_step_fn(
        self,
        rng: jnp.ndarray,
        batch: dict[str, ArrayLike],
    ):
        """Step function for GENOT solver."""
        rng = jax.random.split(rng, 5)
        rng, rng_resample, rng_noise, rng_time, rng_step_fn = rng

        src, tgt = batch["src_cell_data"], batch["tgt_cell_data"]
        condition = batch.get("condition")

        matching_data = (src, tgt)

        (src, tgt), matching_data = self._prepare_data(batch)
        n = src.shape[0]
        time = self.time_sampler(rng_time, n)
        latent = self.latent_noise_fn(rng_noise, (n,))

        tmat = self.data_match_fn(*matching_data)
        src_ixs, tgt_ixs = solver_utils.sample_joint(
            rng_resample,
            tmat,
        )

        src, tgt = src[src_ixs], tgt[tgt_ixs]
        loss, self.vf_state = self.step_fn(
            rng_step_fn, self.vf_state, time, src, tgt, latent, condition
        )
        return loss

    def predict(
        self,
        source: ArrayLike,
        condition: dict[str, ArrayLike] | None = None,
        rng: ArrayLike | None = None,
        **kwargs: Any,
    ) -> ArrayLike:
        """Transport data with the learned plan.

        This function pushes forward the source distribution to its conditional
        distribution by solving the neural ODE.

        Args:
          source: Data to transport.
          condition: Condition of the input data.
          t0: Starting time of integration of neural ODE.
          t1: End time of integration of neural ODE.
          rng: Random generate used to sample from the latent distribution.
          kwargs: Keyword arguments for :func:`~diffrax.odesolve`.

        Returns
        -------
          The push-forward defined by the learned transport plan.
        """
        kwargs.setdefault("dt0", None)
        kwargs.setdefault("solver", diffrax.Tsit5())
        kwargs.setdefault(
            "stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5)
        )

        def vf(
            t: jnp.ndarray, x: jnp.ndarray, cond: dict[str, jnp.ndarray] | None
        ) -> jnp.ndarray:
            params = self.vf_state.params
            return self.vf_state.apply_fn({"params": params}, t, x, cond, train=False)

        def solve_ode(
            x: jnp.ndarray, condition: jnp.ndarray | None, cell_data: jnp.ndarray
        ) -> jnp.ndarray:
            ode_term = diffrax.ODETerm(vf)
            condition[GENOT_CELL_KEY] = cell_data
            result = diffrax.diffeqsolve(
                ode_term,
                t0=0.0,
                t1=1.0,
                y0=x,
                args=condition,
                dt0=None,
                solver=diffrax.Tsit5(),
                stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5),
            )
            return result.ys[0]

        rng = utils.default_prng_key(rng)
        latent = self.latent_noise_fn(rng, (len(source),))

        x_pred = jax.jit(jax.vmap(solve_ode, in_axes=[0, None, 0]))(
            latent, condition, source
        )
        return np.array(x_pred)
