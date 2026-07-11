from collections.abc import Callable
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
from flax.training import train_state
from ott.solvers import utils as solver_utils

from cellflow._compat import BaseFlow
from cellflow._types import ArrayLike
from cellflow.networks._velocity_field import ConditionalVelocityField
from cellflow.solvers._base import BaseSolver, ClassifierFreeGuidance, Guidance, VelocityFn
from cellflow.solvers.utils import ema_update

# Re-exported for backwards compatibility; the canonical home is ``cellflow.solvers._base``.
__all__ = ["OTFlowMatching", "ClassifierFreeGuidance", "Guidance"]


class OTFlowMatching(BaseSolver):
    """(OT) flow matching :cite:`lipman:22` extended to the conditional setting.

    With an extension to OT-CFM :cite:`tong:23,pooladian:23`, and its
    unbalanced version :cite:`eyring:24`.

    Parameters
    ----------
        vf
            Vector field parameterized by a neural network.
        probability_path
            Probability path between the source and the target distributions.
        match_fn
            Function to match samples from the source and the target
            distributions. It has a ``(src, tgt) -> matching`` signature,
            see e.g. :func:`cellflow.utils.match_linear`. If :obj:`None`, no
            matching is performed, and pure probability_path matching :cite:`lipman:22`
            is applied.
        time_sampler
            Time sampler with a ``(rng, n_samples) -> time`` signature, see e.g.
            :func:`ott.solvers.utils.uniform_sampler`.
        guidance
            Optional guidance strategy applied to the velocity field on the predict
            path, see e.g. :class:`ClassifierFreeGuidance`. If :obj:`None` (the
            default), the plain conditional velocity field is used and prediction is
            unchanged.
        kwargs
            Keyword arguments for :meth:`cellflow.networks.ConditionalVelocityField.create_train_state`.
    """

    @staticmethod
    def _match_kwargs(*, match_fn: Callable, data_dim: int) -> dict[str, Any]:
        """Solver-specific constructor kwargs derived from the model's match function and data dim.

        Called by :meth:`cellflow.model.CellFlow.prepare_model` so each solver owns how it names its
        matching function and whether it needs source/target dimensions, keeping the model code free of
        per-solver branches. ``OTFlowMatching`` matches on ``match_fn`` and takes no explicit dimensions.
        """
        return {"match_fn": match_fn}

    def __init__(
        self,
        vf: ConditionalVelocityField,
        probability_path: BaseFlow,
        match_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
        time_sampler: Callable[[jax.Array, int], jnp.ndarray] = solver_utils.uniform_sampler,
        guidance: Guidance | None = None,
        **kwargs: Any,
    ):
        super().__init__(vf, probability_path, time_sampler, guidance)
        self.match_fn = jax.jit(match_fn)
        self.ema = kwargs.pop("ema", 1.0)

        self.vf_state = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_state_inference = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_step_fn = self._get_vf_step_fn()

    @property
    def _inference_state(self) -> train_state.TrainState:
        """Inference train state (EMA-averaged when ``ema < 1``) driving prediction."""
        return self.vf_state_inference

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
        """Evaluate the velocity field on the condition only (``source`` is unused)."""
        return vf_state.apply_fn(
            {"params": params},
            t,
            x_t,
            condition,
            encoder_noise=encoder_noise,
            rngs=rngs,
        )

    def step_fn(
        self,
        rng: jnp.ndarray,
        batch: dict[str, ArrayLike],
    ) -> float:
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
        src, tgt = batch["src_cell_data"], batch["tgt_cell_data"]
        condition = batch.get("condition")
        rng_resample, rng_time, rng_step_fn, rng_encoder_noise = jax.random.split(rng, 4)
        n = src.shape[0]
        time = self.time_sampler(rng_time, n)
        encoder_noise = jax.random.normal(rng_encoder_noise, (n, self.vf.condition_embedding_dim))
        # TODO: test whether it's better to sample the same noise for all samples or different ones

        if self.match_fn is not None:
            tmat = self.match_fn(src, tgt)
            src_ixs, tgt_ixs = solver_utils.sample_joint(rng_resample, tmat)
            src, tgt = src[src_ixs], tgt[tgt_ixs]

        self.vf_state, loss = self.vf_step_fn(
            rng_step_fn,
            self.vf_state,
            time,
            src,
            tgt,
            src,  # x_0: the flow starts at the source cells
            condition,
            encoder_noise,
        )

        if self.ema == 1.0:
            self.vf_state_inference = self.vf_state
        else:
            self.vf_state_inference = self.vf_state_inference.replace(
                params=ema_update(self.vf_state_inference.params, self.vf_state.params, self.ema)
            )
        return loss

    def _base_velocity(self) -> VelocityFn:
        """Return the base (conditional) velocity closure used on the predict path.

        The closure has the diffrax ``(t, x, args) -> velocity`` signature, with
        ``args`` being ``(params, condition, encoder_noise)``, and evaluates the
        inference velocity field conditionally (``force_uncond=False``).
        """

        def vf(t: jnp.ndarray, x: jnp.ndarray, args: tuple[Any, dict[str, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
            params, condition, encoder_noise = args
            return self._inference_state.apply_fn({"params": params}, t, x, condition, encoder_noise, train=False)[0]

        return vf

    def _null_velocity(self) -> VelocityFn:
        """Return the unconditional velocity closure (``force_uncond=True``).

        Same ``args`` layout as :meth:`_base_velocity`; used by guidance to form the
        classifier-free combination.
        """

        def vf(t: jnp.ndarray, x: jnp.ndarray, args: tuple[Any, dict[str, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
            params, condition, encoder_noise = args
            return self._inference_state.apply_fn(
                {"params": params}, t, x, condition, encoder_noise, train=False, force_uncond=True
            )[0]

        return vf

    def _build_predict_fn(self, kwargs: dict[str, Any]) -> Callable:
        """Build the jit+vmap predict function for the given diffrax kwargs.

        The velocity closure comes from :meth:`_guided_velocity`, which applies a
        per-call ``guidance_scale`` (popped from ``kwargs``) or the construction-time
        ``guidance``. With no guidance the closure is exactly the base conditional
        velocity, so prediction is unchanged (no unconditional velocity is computed).
        """
        vf, kwargs = self._guided_velocity(kwargs)

        def solve_ode(
            params: Any, x: jnp.ndarray, condition: dict[str, jnp.ndarray], encoder_noise: jnp.ndarray
        ) -> jnp.ndarray:
            ode_term = diffrax.ODETerm(vf)
            result = diffrax.diffeqsolve(
                ode_term,
                t0=0.0,
                t1=1.0,
                y0=x,
                args=(params, condition, encoder_noise),
                **kwargs,
            )
            return result.ys[0]

        return jax.jit(jax.vmap(solve_ode, in_axes=[None, 0, None, None]))

    def predict(
        self,
        x: ArrayLike | dict[str, ArrayLike],
        condition: dict[str, ArrayLike] | dict[str, dict[str, ArrayLike]] | None = None,
        rng: jax.Array | None = None,
        guidance_scale: float = 1.0,
        **kwargs: Any,
    ) -> ArrayLike | dict[str, ArrayLike]:
        """Predict the translated source ``x`` under condition ``condition``.

        Extends :meth:`~cellflow.solvers._base.BaseSolver.predict` with a per-call
        classifier-free ``guidance_scale``; all other behavior (array/dict dispatch,
        diffrax kwargs) is inherited.

        Parameters
        ----------
        x
            Input data, or a dictionary mapping condition names to input arrays.
        condition
            Condition of the input data, matching the structure of ``x``.
        rng
            Random number generator to sample from the latent distribution, only
            used if ``condition_mode='stochastic'``. If :obj:`None`, the mean
            embedding is used.
        guidance_scale
            Per-call classifier-free guidance scale. When not ``1.0`` it applies
            :class:`ClassifierFreeGuidance` for this call (requires
            :attr:`cfg_enabled`), overriding the construction-time ``guidance``.
            Handy for sweeping ``w`` without rebuilding the solver.
        kwargs
            Keyword arguments for :func:`diffrax.diffeqsolve`.

        Returns
        -------
        The push-forward distribution of ``x`` under condition ``condition``.
        """
        return super().predict(x, condition, rng, guidance_scale=guidance_scale, **kwargs)

    def _predict_jit(
        self,
        x: ArrayLike,
        condition: dict[str, ArrayLike],
        rng: jax.Array | None = None,
        **kwargs: Any,
    ) -> ArrayLike:
        """See :meth:`predict`."""
        kwargs_frozen = self._resolve_diffrax_kwargs(kwargs)
        encoder_noise = self._sample_encoder_noise(rng)
        predict_fn = self._get_predict_fn(kwargs_frozen)
        return predict_fn(self._inference_state.params, x, condition, encoder_noise)
