import functools
from collections.abc import Callable
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
from ott.neural.methods.flows import dynamics
from ott.neural.networks import velocity_field
from ott.solvers import utils as solver_utils

from cellflow import utils
from cellflow._types import ArrayLike
from cellflow.model._utils import _multivariate_normal

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
    probability_path
        Probability path between the latent and the target distributions.
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
        Keyword arguments.
    """

    def __init__(
        self,
        vf: velocity_field.VelocityField,
        probability_path: dynamics.BaseFlow,
        data_match_fn: DataMatchFn,
        *,
        source_dim: int,
        target_dim: int,
        time_sampler: Callable[[jax.Array, int], jnp.ndarray] = solver_utils.uniform_sampler,
        latent_noise_fn: (Callable[[jax.Array, tuple[int, ...]], jnp.ndarray] | None) = None,
        **kwargs: Any,
    ):
        self._is_trained: bool = False
        self.vf = vf
        self.condition_encoder_mode = self.vf.condition_mode
        self.condition_encoder_regularization = self.vf.regularization
        self.probability_path = probability_path
        self.data_match_fn = jax.jit(data_match_fn)
        self.time_sampler = time_sampler
        self.source_dim = source_dim
        if latent_noise_fn is None:
            latent_noise_fn = functools.partial(_multivariate_normal, dim=target_dim)
        self.latent_noise_fn = latent_noise_fn

        self.vf_state = self.vf.create_train_state(
            input_dim=target_dim,
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
            encoder_noise: jnp.ndarray,
        ):
            def loss_fn(
                params: jnp.ndarray,
                t: jnp.ndarray,
                source: jnp.ndarray,
                target: jnp.ndarray,
                latent: jnp.ndarray,
                condition: dict[str, jnp.ndarray] | None,
                encoder_noise: jnp.ndarray,
                rng: jax.Array,
            ) -> jnp.ndarray:
                rng_flow, rng_encoder, rng_dropout = jax.random.split(rng, 3)
                x_t = self.probability_path.compute_xt(rng_flow, t, latent, target)
                v_t, mean_cond, logvar_cond = vf_state.apply_fn(
                    {"params": params},
                    t,
                    x_t,
                    source,
                    condition,
                    encoder_noise=encoder_noise,
                    rngs={"dropout": rng_dropout, "condition_encoder": rng_encoder},
                )
                u_t = self.probability_path.compute_ut(t, x_t, source, target)
                flow_matching_loss = jnp.mean((v_t - u_t) ** 2)
                condition_mean_regularization = 0.5 * jnp.mean(mean_cond**2)
                condition_var_regularization = -0.5 * jnp.mean(1 + logvar_cond - jnp.exp(logvar_cond))
                if self.condition_encoder_mode == "stochastic":
                    encoder_loss = condition_mean_regularization + condition_var_regularization
                elif (self.condition_encoder_mode == "deterministic") and (self.condition_encoder_regularization > 0):
                    encoder_loss = condition_mean_regularization
                else:
                    encoder_loss = 0.0
                return flow_matching_loss + encoder_loss

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(vf_state.params, time, source, target, latent, conditions, encoder_noise, rng)

            return loss, vf_state.apply_gradients(grads=grads)

        return vf_step_fn

    @staticmethod
    def _prepare_data(
        batch: dict[str, jnp.ndarray],
    ) -> tuple[
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
        elif all(arr is not None for arr in (src_lin, tgt_lin, src_quad, tgt_quad)):  # fused quad
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
        rng = jax.random.split(rng, 6)
        rng, rng_resample, rng_noise, rng_time, rng_step_fn, rng_encoder_noise = rng

        condition = batch.get("condition")
        (src, tgt), matching_data = self._prepare_data(batch)
        n = src.shape[0]
        time = self.time_sampler(rng_time, n)
        latent = self.latent_noise_fn(rng_noise, (n,))
        encoder_noise = jax.random.normal(rng_encoder_noise, (n, self.vf.condition_embedding_dim))
        # TODO: test whether it's better to sample the same noise for all samples or different ones

        tmat = self.data_match_fn(*matching_data)
        src_ixs, tgt_ixs = solver_utils.sample_joint(
            rng_resample,
            tmat,
        )

        src, tgt = src[src_ixs], tgt[tgt_ixs]
        loss, self.vf_state = self.vf_step_fn(
            rng_step_fn, self.vf_state, time, src, tgt, latent, condition, encoder_noise
        )
        return loss

    def get_condition_embedding(self, condition: dict[str, ArrayLike], return_as_numpy=True) -> ArrayLike:
        """Get learnt embeddings of the conditions.

        Parameters
        ----------
        condition
            Conditions to encode
        return_as_numpy
            Whether to return the embeddings as numpy arrays.


        Returns
        -------
        Mean and log-variance of encoded conditions.
        """
        cond_mean, cond_logvar = self.vf.apply(
            {"params": self.vf_state.params},
            condition,
            method="get_condition_embedding",
        )
        if return_as_numpy:
            return np.asarray(cond_mean), np.asarray(cond_logvar)
        return cond_mean, cond_logvar

    def predict(
        self,
        x: ArrayLike,
        condition: dict[str, ArrayLike] | None = None,
        rng: ArrayLike | None = None,
        rng_genot: ArrayLike | None = None,
        **kwargs: Any,
    ) -> ArrayLike | tuple[ArrayLike, diffrax.Solution]:
        """Generate the push-forward of ``x`` under condition ``condition``.

        This function solves the ODE learnt with
        the :class:`~cellflow.networks.ConditionalVelocityField`.

        Parameters
        ----------
        x
            Input data of shape [batch_size, ...].
        condition
            Condition of the input data of shape [batch_size, ...].
        rng
            Random number generator to sample from the latent distribution,
            only used if ``condition_mode='stochastic'``. If :obj:`None`, the
            mean embedding is used.
        rng_genot
            Random generate used to sample from the latent distribution in cell space.
        kwargs
            Keyword arguments for :func:`diffrax.diffeqsolve`.

        Returns
        -------
        The push-forward distribution of ``x`` under condition ``condition``.
        """
        kwargs.setdefault("dt0", None)
        kwargs.setdefault("solver", diffrax.Tsit5())
        kwargs.setdefault("stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5))

        noise_dim = (1, self.vf.condition_embedding_dim)
        use_mean = rng is None or self.condition_encoder_mode == "deterministic"
        rng = utils.default_prng_key(rng)
        encoder_noise = jnp.zeros(noise_dim) if use_mean else jax.random.normal(rng, noise_dim)
        rng_genot = utils.default_prng_key(rng_genot)
        latent = self.latent_noise_fn(rng_genot, (x.shape[0],))

        def vf(t: float, x: jnp.ndarray, args: tuple[dict[str, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
            params = self.vf_state.params
            x_0, condition, encoder_noise = args
            return self.vf_state.apply_fn({"params": params}, t, x, x_0, condition, encoder_noise, train=False)[0]

        def solve_ode(
            latent: jnp.ndarray, x: jnp.ndarray, condition: dict[str, jnp.ndarray], encoder_noise: jnp.ndarray
        ) -> jnp.ndarray:
            term = diffrax.ODETerm(vf)
            sol = diffrax.diffeqsolve(
                term,
                t0=0.0,
                t1=1.0,
                y0=latent,
                args=(x, condition, encoder_noise),
                **kwargs,
            )
            return sol.ys[0]

        x_pred = jax.jit(jax.vmap(solve_ode, in_axes=[0, 0, None, None]))(latent, x, condition, encoder_noise)
        return np.array(x_pred)

    @property
    def is_trained(self) -> bool:
        """If the model is trained."""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value) -> None:
        self._is_trained = value
