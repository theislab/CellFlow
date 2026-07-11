import functools
from collections.abc import Callable
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from ott.solvers import utils as solver_utils

from cellflow import utils
from cellflow._compat import BaseFlow
from cellflow._types import ArrayLike
from cellflow.model._utils import _multivariate_normal
from cellflow.solvers._base import BaseSolver, Guidance, VelocityFn

__all__ = ["GENOT"]

LinTerm = tuple[jnp.ndarray, jnp.ndarray]
QuadTerm = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None]
DataMatchFn = Callable[[LinTerm], jnp.ndarray] | Callable[[QuadTerm], jnp.ndarray]


class GENOT(BaseSolver):
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
    guidance
        Optional guidance strategy applied to the velocity field on the predict
        path, see e.g. :class:`~cellflow.solvers.ClassifierFreeGuidance`. If
        :obj:`None` (the default), the plain conditional velocity field is used.
        Guidance requires a velocity field trained with ``condition_dropout_prob > 0``.
    kwargs
        Keyword arguments.
    """

    @staticmethod
    def _match_kwargs(*, match_fn: Callable, data_dim: int) -> dict[str, Any]:
        """Solver-specific constructor kwargs derived from the model's match function and data dim.

        Called by :meth:`cellflow.model.CellFlow.prepare_model` so each solver owns how it names its
        matching function and whether it needs source/target dimensions, keeping the model code free of
        per-solver branches. ``GENOT`` matches on ``data_match_fn`` and needs explicit
        ``source_dim``/``target_dim``.
        """
        return {"data_match_fn": match_fn, "source_dim": data_dim, "target_dim": data_dim}

    def __init__(
        self,
        vf: nn.Module,
        probability_path: BaseFlow,
        data_match_fn: DataMatchFn,
        *,
        source_dim: int,
        target_dim: int,
        time_sampler: Callable[[jax.Array, int], jnp.ndarray] = solver_utils.uniform_sampler,
        latent_noise_fn: (Callable[[jax.Array, tuple[int, ...]], jnp.ndarray] | None) = None,
        guidance: Guidance | None = None,
        **kwargs: Any,
    ):
        super().__init__(vf, probability_path, time_sampler, guidance)
        self.data_match_fn = jax.jit(data_match_fn)
        self.source_dim = source_dim
        self.latent_noise_fn = latent_noise_fn or functools.partial(_multivariate_normal, dim=target_dim)

        self.vf_state = self.vf.create_train_state(
            input_dim=target_dim,
            **kwargs,
        )
        self.vf_step_fn = self._get_vf_step_fn()

    def _apply_vf(
        self,
        vf_state: train_state.TrainState,
        params: Any,
        t: jnp.ndarray,
        x_t: jnp.ndarray,
        source: jnp.ndarray,
        condition: dict[str, jnp.ndarray] | None,
        encoder_noise: jnp.ndarray,
        rngs: dict[str, jax.Array],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Evaluate the velocity field, conditioning on the source cells ``source``."""
        return vf_state.apply_fn(
            {"params": params},
            t,
            x_t,
            source,
            condition,
            encoder_noise=encoder_noise,
            rngs=rngs,
        )

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
        self.vf_state, loss = self.vf_step_fn(
            rng_step_fn, self.vf_state, time, src, tgt, latent, condition, encoder_noise
        )
        return loss

    def predict(
        self,
        x: ArrayLike | dict[str, ArrayLike],
        condition: dict[str, ArrayLike] | None = None,
        rng: ArrayLike | None = None,
        rng_genot: ArrayLike | None = None,
        guidance_scale: float = 1.0,
        **kwargs: Any,
    ) -> ArrayLike | dict[str, ArrayLike]:
        """Generate the push-forward of ``x`` under condition ``condition``.

        Extends :meth:`~cellflow.solvers._base.BaseSolver.predict` with the extra
        ``rng_genot`` control and a per-call classifier-free ``guidance_scale``; all
        other behavior (array/dict dispatch, diffrax kwargs) is inherited.

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
            Random generator used to sample from the latent distribution in cell space.
        guidance_scale
            Per-call classifier-free guidance scale. When not ``1.0`` it applies
            :class:`~cellflow.solvers.ClassifierFreeGuidance` for this call (requires
            :attr:`cfg_enabled`), overriding the construction-time ``guidance``.
        kwargs
            Keyword arguments for :func:`diffrax.diffeqsolve`.

        Returns
        -------
        The push-forward distribution of ``x`` under condition ``condition``.
        """
        return super().predict(x, condition, rng, rng_genot=rng_genot, guidance_scale=guidance_scale, **kwargs)

    def _base_velocity(self) -> VelocityFn:
        """Return the base (conditional) velocity closure used on the predict path.

        The closure has the diffrax ``(t, x, args) -> velocity`` signature, with
        ``args`` being ``(params, x_0, condition, encoder_noise)``, and evaluates the
        inference velocity field conditionally (``force_uncond=False``).
        """

        def vf(
            t: float, x: jnp.ndarray, args: tuple[Any, jnp.ndarray, dict[str, jnp.ndarray], jnp.ndarray]
        ) -> jnp.ndarray:
            params, x_0, condition, encoder_noise = args
            return self._inference_state.apply_fn({"params": params}, t, x, x_0, condition, encoder_noise, train=False)[
                0
            ]

        return vf

    def _null_velocity(self) -> VelocityFn:
        """Return the unconditional velocity closure (``force_uncond=True``).

        Same ``args`` layout as :meth:`_base_velocity`; used by guidance to form the
        classifier-free combination.
        """

        def vf(
            t: float, x: jnp.ndarray, args: tuple[Any, jnp.ndarray, dict[str, jnp.ndarray], jnp.ndarray]
        ) -> jnp.ndarray:
            params, x_0, condition, encoder_noise = args
            return self._inference_state.apply_fn(
                {"params": params}, t, x, x_0, condition, encoder_noise, train=False, force_uncond=True
            )[0]

        return vf

    def _build_predict_fn(self, kwargs: dict[str, Any]) -> Callable:
        """Build the jit+vmap predict function for the given diffrax kwargs.

        The velocity closure (from :meth:`_guided_velocity`, applying a per-call
        ``guidance_scale`` or the construction-time ``guidance``) additionally
        conditions on the source cell ``x_0``, and the ODE is integrated from a
        sampled latent, so the ``vmap`` maps over both the latent and the source
        (``in_axes=[None, 0, 0, None, None]``).
        """
        vf, kwargs = self._guided_velocity(kwargs)

        def solve_ode(
            params: Any,
            latent: jnp.ndarray,
            x: jnp.ndarray,
            condition: dict[str, jnp.ndarray],
            encoder_noise: jnp.ndarray,
        ) -> jnp.ndarray:
            term = diffrax.ODETerm(vf)
            sol = diffrax.diffeqsolve(
                term,
                t0=0.0,
                t1=1.0,
                y0=latent,
                args=(params, x, condition, encoder_noise),
                **kwargs,
            )
            return sol.ys[0]

        return jax.jit(jax.vmap(solve_ode, in_axes=[None, 0, 0, None, None]))

    def _predict_jit(
        self,
        x: ArrayLike,
        condition: dict[str, ArrayLike] | None = None,
        rng: ArrayLike | None = None,
        rng_genot: ArrayLike | None = None,
        **kwargs: Any,
    ) -> ArrayLike:
        kwargs_frozen = self._resolve_diffrax_kwargs(kwargs)
        encoder_noise = self._sample_encoder_noise(rng)
        rng_genot = utils.default_prng_key(rng_genot)
        latent = self.latent_noise_fn(rng_genot, (x.shape[0],))

        predict_fn = self._get_predict_fn(kwargs_frozen)
        return predict_fn(self._inference_state.params, latent, x, condition, encoder_noise)
