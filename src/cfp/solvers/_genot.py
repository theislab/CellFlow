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
    """GENOT :cite:`klein:23` extended to the conditional setting.

    Parameters
    ----------
    vf
        Vector field parameterized by a neural network.
    flow
        Flow between the latent and the target distributions.
    data_match_fn
        Function to match samples from the source and the target
        distributions. Depending on the data passed :meth:`step_fn`, it has
        the following signature:

        - ``(src_lin, tgt_lin) -> matching`` - linear matching.
        - ``(src_quad, tgt_quad, src_lin, tgt_lin) -> matching`` - quadratic (fused) GW matching.
        In the pure GW setting, both ``src_lin`` and ``tgt_lin`` will be set to :obj:`None`.

    source_dim
        Dimensionality of the source distribution.
    target_dim
        Dimensionality of the target distribution.
    time_sampler
        Time sampler with a ``(rng, n_samples) -> time`` signature, see e.g.
        :func:`ott.solvers.utils.uniform_sampler`.
    latent_noise_fn
        Function to sample from the latent distribution in the
        target space with a ``(rng, shape) -> noise`` signature.
        If :obj:`None`, multivariate normal distribution is used.
    kwargs
        Keyword arguments for TODO
    """

    def __init__(
        self,
        vf: velocity_field.VelocityField,
        flow: dynamics.BaseFlow,
        data_match_fn: DataMatchFn,
        *,
        source_dim: int,
        target_dim: int,
        time_sampler: Callable[
            [jax.Array, int], jnp.ndarray
        ] = solver_utils.uniform_sampler,
        latent_noise_fn: (
            Callable[[jax.Array, tuple[int, ...]], jnp.ndarray] | None
        ) = None,
        **kwargs: Any,
    ):
        self._is_trained: bool = False
        self.vf = vf
        self.flow = flow
        self.data_match_fn = jax.jit(data_match_fn)
        self.time_sampler = time_sampler
        self.source_dim = source_dim
        if latent_noise_fn is None:
            latent_noise_fn = functools.partial(_multivariate_normal, dim=target_dim)
        self.latent_noise_fn = latent_noise_fn

        self.vf_state = self.vf.create_train_state(
            input_dim=target_dim,
            additional_cond_dim=source_dim,
            **kwargs,
        )
        self.vf_step_fn = self._get_vf_step_fn()

    def _get_vf_step_fn(self) -> Callable:  #  type: ignore[type-arg]

        @jax.jit
        def vf_step_fn(
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
                u_t = self.flow.compute_ut(time, x_t, latent, target)

                return jnp.mean((v_t - u_t) ** 2)

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(
                vf_state.params, time, source, target, latent, conditions, rng
            )

            return loss, vf_state.apply_gradients(grads=grads)

        return vf_step_fn

    @staticmethod
    def _prepare_data(batch: dict[str, jnp.ndarray]) -> tuple[
        tuple[ArrayLike, ArrayLike],
        tuple[ArrayLike | None, ...],
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

        return (src, tgt), arrs  # type: ignore[return-value]

    def step_fn(
        self,
        rng: jnp.ndarray,
        batch: dict[str, ArrayLike],
    ):
        """Single step function of the solver.

        Parameters
        ----------
        rng
            Random number generator.
        batch
            Data batch with keys ``src_cell_data``, ``tgt_cell_data``, and
            optionally ``condition``.

        Returns
        -------
        Loss value.
        """
        rng = jax.random.split(rng, 5)
        rng, rng_resample, rng_noise, rng_time, rng_step_fn = rng

        condition = batch.get("condition")
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
        loss, self.vf_state = self.vf_step_fn(
            rng_step_fn, self.vf_state, time, src, tgt, latent, condition
        )
        return loss

    def get_condition_embedding(self, condition: dict[str, ArrayLike]) -> ArrayLike:
        """Get learnt embeddings of the conditions.

        Parameters
        ----------
        condition
            Conditions to encode

        Returns
        -------
        Encoded conditions.
        """
        dummy_source = jnp.zeros((1, self.source_dim))
        condition.update({GENOT_CELL_KEY: dummy_source})
        cond_embed = self.vf.apply(
            {"params": self.vf_state.params},
            condition,
            method="get_condition_embedding",
        )
        return np.asarray(cond_embed)

    def predict(
        self,
        source: ArrayLike,
        condition: dict[str, ArrayLike] | None = None,
        n_samples: int = 1,
        rng: ArrayLike | None = None,
        **kwargs: Any,
    ) -> ArrayLike | tuple[ArrayLike, diffrax.Solution]:
        """Generate the push-forward of ``'source'`` under condition ``'condition'``.

        This function solves the ODE learnt with
        the :class:`~cfp.networks.ConditionalVelocityField`.

        Parameters
        ----------
        source
            Input data of shape [batch_size, ...].
        condition
            Condition of the input data of shape [batch_size, ...].
        n_samples
            Number of conditional samples to generate.
        rng
            Random generate used to sample from the latent distribution.
        kwargs
            Keyword arguments for :func:`diffrax.diffeqsolve`.

        Returns
        -------
        The push-forward distribution of ``'x'`` under condition ``'condition'``.
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
            x: jnp.ndarray, condition: dict[str, jnp.ndarray], cell_data: jnp.ndarray
        ) -> jnp.ndarray:
            ode_term = diffrax.ODETerm(vf)
            condition[GENOT_CELL_KEY] = cell_data
            result = diffrax.diffeqsolve(
                ode_term,
                t0=0.0,
                t1=1.0,
                y0=x,
                args=condition,
                **kwargs,
            )
            return result.ys[0]

        rng = utils.default_prng_key(rng)
        latent = self.latent_noise_fn(rng, (len(source), n_samples))

        x_pred = jax.jit(
            jax.vmap(jax.vmap(solve_ode, in_axes=[0, None, 0]), in_axes=[1, None, None])
        )(latent, condition, source)
        return np.transpose(np.array(x_pred), (1, 2, 0))

    @property
    def is_trained(self) -> bool:
        """If the model is trained."""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value) -> None:
        self._is_trained = value
