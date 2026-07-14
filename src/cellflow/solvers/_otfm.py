import warnings
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import frozen_dict
from flax.training import train_state
from ott.solvers import utils as solver_utils

from cellflow import utils
from cellflow._compat import BaseFlow
from cellflow._types import ArrayLike
from cellflow.networks._velocity_field import ConditionalVelocityField
from cellflow.solvers._base import BaseSolver
from cellflow.solvers.utils import ema_update

__all__ = ["OTFlowMatching", "ClassifierFreeGuidance", "Guidance"]

# A velocity closure with the diffrax signature ``(t, x, args) -> velocity``,
# where ``args`` is ``(params, condition, encoder_noise)``.
VelocityFn = Callable[[jnp.ndarray, jnp.ndarray, tuple[Any, ...]], jnp.ndarray]


@runtime_checkable
class Guidance(Protocol):
    """Pluggable transform applied to the base velocity field on the predict path.

    A guidance strategy receives the base velocity closure ``vf(t, x, args,
    force_uncond=False)`` — which owns the field's call signature and returns the
    conditional (``force_uncond=False``) or unconditional (``True``) velocity — and
    returns a plain ``(t, x, args) -> velocity`` closure. Taking the closure (rather
    than the train state) keeps guidance agnostic to solver-specific ``args`` such as
    GENOT's source ``x_0``.
    """

    def wrap(self, vf: Callable) -> VelocityFn:
        """Wrap the base velocity ``vf`` and return the guided velocity."""
        ...


class ClassifierFreeGuidance:
    """Classifier-free guidance on the predict path.

    Combines the conditional velocity ``v_cond`` with the unconditional velocity
    ``v_null`` (obtained by forcing the velocity field to drop its condition via
    ``force_uncond=True``) according to ``v = v_null + scale * (v_cond - v_null)``.

    A ``scale`` of ``1.0`` recovers the purely conditional velocity (and thus the
    behavior without guidance), while larger values amplify the influence of the
    condition. Using this strategy only makes sense for a velocity field trained
    with ``condition_dropout_prob > 0`` so that ``v_null`` is meaningful.

    How the unconditional ``v_null`` is defined (zeroed embedding vs a
    :attr:`~cellflow.networks.ConditionalVelocityField.mask_value`-filled condition)
    is controlled by the velocity field's ``condition_null`` mode; this strategy is
    agnostic to it.

    The equivalent "guidance weight" convention ``(1 + w) * v_cond - w * v_null``
    (with ``w = 0`` meaning no guidance) is available via :meth:`from_ode_weight`,
    since ``scale = 1 + w``.

    Parameters
    ----------
    scale
        Guidance strength.
    """

    def __init__(self, scale: float):
        self.scale = scale

    @classmethod
    def from_ode_weight(cls, cfg_ode_weight: float) -> "ClassifierFreeGuidance":
        """Build from the ``cfg_ode_weight`` convention: ``(1 + w) * v_cond - w * v_null``.

        This is the parameterization used elsewhere in the ecosystem, where
        ``cfg_ode_weight = 0`` means no guidance and larger values increase it. It
        maps to ``scale = 1 + cfg_ode_weight``.
        """
        if cfg_ode_weight < 0:
            raise ValueError("cfg_ode_weight must be non-negative.")
        return cls(scale=1.0 + cfg_ode_weight)

    def wrap(self, vf: Callable) -> VelocityFn:
        """Return a velocity closure computing ``v_null + scale * (v_cond - v_null)``.

        ``vf`` is the base velocity ``vf(t, x, args, force_uncond=False)``; it owns the
        field's call signature, so this blend is agnostic to solver-specific ``args``
        (e.g. GENOT threads its source ``x_0`` through ``args``).
        """
        scale = self.scale

        def guided_vf(t: jnp.ndarray, x: jnp.ndarray, args: tuple[Any, ...]) -> jnp.ndarray:
            v_cond = vf(t, x, args, force_uncond=False)
            v_null = vf(t, x, args, force_uncond=True)
            return v_null + scale * (v_cond - v_null)

        return guided_vf


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
        super().__init__(vf, probability_path, time_sampler)
        self.match_fn = jax.jit(match_fn)
        self.guidance = guidance
        self.ema = kwargs.pop("ema", 1.0)

        self.vf_state = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_state_inference = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_step_fn = self._get_vf_step_fn()

    def _get_vf_step_fn(self) -> Callable:  # type: ignore[type-arg]
        @jax.jit
        def vf_step_fn(
            rng: jax.Array,
            vf_state: train_state.TrainState,
            time: jnp.ndarray,
            source: jnp.ndarray,
            target: jnp.ndarray,
            conditions: dict[str, jnp.ndarray],
            encoder_noise: jnp.ndarray,
        ):
            def loss_fn(
                params: jnp.ndarray,
                t: jnp.ndarray,
                source: jnp.ndarray,
                target: jnp.ndarray,
                conditions: dict[str, jnp.ndarray],
                encoder_noise: jnp.ndarray,
                rng: jax.Array,
            ) -> jnp.ndarray:
                rng_flow, rng_encoder, rng_dropout = jax.random.split(rng, 3)
                x_t = self.probability_path.compute_xt(rng_flow, t, source, target)
                v_t, mean_cond, logvar_cond = vf_state.apply_fn(
                    {"params": params},
                    t,
                    x_t,
                    conditions,
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
            loss, grads = grad_fn(vf_state.params, time, source, target, conditions, encoder_noise, rng)
            return vf_state.apply_gradients(grads=grads), loss

        return vf_step_fn

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

    @property
    def _inference_state(self) -> train_state.TrainState:
        """OTFM predicts and reads condition embeddings from the EMA inference state."""
        return self.vf_state_inference

    @property
    def cfg_enabled(self) -> bool:
        """Whether classifier-free guidance is available at predict time.

        Guidance needs a meaningful unconditional velocity ``v_null``, which only
        exists when the velocity field was trained with condition dropout
        (``condition_dropout_prob > 0``). When ``False``, a per-call
        ``guidance_scale`` is ignored (with a warning) and the plain conditional
        velocity is used.
        """
        return float(getattr(self.vf, "condition_dropout_prob", 0.0)) > 0.0

    def _base_velocity(self) -> VelocityFn:
        """Return the base (conditional) velocity closure used on the predict path.

        The closure has the diffrax ``(t, x, args) -> velocity`` signature, with
        ``args`` being ``(params, condition, encoder_noise)``, and evaluates the
        inference velocity field conditionally (``force_uncond=False``).
        """

        def vf(
            t: jnp.ndarray,
            x: jnp.ndarray,
            args: tuple[Any, dict[str, jnp.ndarray], jnp.ndarray],
            force_uncond: bool = False,
        ) -> jnp.ndarray:
            params, condition, encoder_noise = args
            return self.vf_state_inference.apply_fn(
                {"params": params}, t, x, condition, encoder_noise, train=False, force_uncond=force_uncond
            )[0]

        return vf

    def _base_velocity_from_embedding(self) -> Callable:
        """Velocity closure taking a *precomputed* condition embedding in ``args``.

        ``args`` is ``(params, cond_embedding)``. Used on the predict path when no guidance is active, so
        the condition encoder runs once (outside the ODE, in :meth:`_get_predict_fn`) rather than on
        every integration step; the embedding is constant along the trajectory, so the result is
        identical to :meth:`_base_velocity`.
        """

        def vf(t: jnp.ndarray, x: jnp.ndarray, args: tuple[Any, jnp.ndarray]) -> jnp.ndarray:
            params, cond_embedding = args
            return self.vf_state_inference.apply_fn(
                {"params": params}, t, x, cond_embedding, train=False, method="velocity_from_embedding"
            )

        return vf

    def _effective_guidance(self, guidance_scale: float, *, warn: bool) -> "Guidance | None":
        """Resolve the guidance strategy for a predict call.

        A per-call ``guidance_scale != 1.0`` builds a :class:`ClassifierFreeGuidance` (requires
        :attr:`cfg_enabled`), overriding the construction-time :attr:`guidance`; ``1.0`` uses
        :attr:`guidance` as-is. ``warn`` emits the "not trained for CFG" notice — set only where the fn is
        actually built (on cache miss) so routing checks stay silent and it fires once, as before.
        """
        guidance_scale = float(guidance_scale)
        if guidance_scale == 1.0:
            return self.guidance
        if self.cfg_enabled:
            # v = v_null + scale·(v_cond − v_null); overrides construction-time guidance.
            return ClassifierFreeGuidance(scale=guidance_scale)
        if warn:
            warnings.warn(
                f"guidance_scale={guidance_scale} ignored: the velocity field was not trained "
                "with classifier-free guidance (condition_dropout_prob == 0), so the "
                "unconditional velocity is undefined. Using the plain conditional velocity.",
                stacklevel=2,
            )
        return None

    @staticmethod
    def _frozen_predict_kwargs(kwargs: dict[str, Any]) -> frozen_dict.FrozenDict:
        """Apply the diffrax defaults, then freeze — the frozen dict is the predict-fn cache key."""
        kwargs.setdefault("dt0", None)
        kwargs.setdefault("solver", diffrax.Tsit5())
        kwargs.setdefault("stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5))
        return frozen_dict.freeze(kwargs)

    def _encoder_noise(self, rng: jax.Array | None, n: int = 1) -> jnp.ndarray:
        """Latent noise for the condition encoder: zeros for the mean embedding, else one Gaussian draw."""
        use_mean = rng is None or self.condition_encoder_mode == "deterministic"
        dim = (n, self.vf.condition_embedding_dim)
        return jnp.zeros(dim) if use_mean else jax.random.normal(utils.default_prng_key(rng), dim)

    def _encode_conditions(self, condition: dict[str, jnp.ndarray], encoder_noise: jnp.ndarray) -> jnp.ndarray:
        """Run the condition encoder once → the (normed) per-condition embedding fed to the predict rhs."""
        return self.vf_state_inference.apply_fn(
            {"params": self.vf_state_inference.params},
            condition,
            encoder_noise,
            train=False,
            method="encode_condition",
        )[0]

    def _get_predict_fn(self, kwargs_frozen: frozen_dict.FrozenDict) -> Callable:
        """Build and cache the ``jit(vmap)`` predict fn for the given diffrax kwargs.

        One cache (:attr:`_predict_fn_cache`) with two possible fn shapes. A solver's guidance state is
        fixed and ``guidance_scale`` is part of the key, so each key maps to exactly one shape:

        - **no guidance**: ``(params, x, cond_embedding)`` vmapped over cells with a precomputed *per-cell*
          embedding (``in_axes=[None, 0, 0]``). The condition encoder runs once, outside the ODE; the same
          fn serves both single-array and condition-batched prediction — the callers build the embedding
          (broadcast for one condition, concatenated across conditions for a batch).
        - **guidance**: ``(params, x, condition, encoder_noise)`` vmapped over cells; guidance wraps the
          full velocity, which needs the encoder inside the ODE (conditional + null per step).
        """
        if kwargs_frozen in self._predict_fn_cache:
            return self._predict_fn_cache[kwargs_frozen]

        kwargs = dict(kwargs_frozen)
        guidance = self._effective_guidance(kwargs.pop("guidance_scale", 1.0), warn=True)

        if guidance is None:
            vf = self._base_velocity_from_embedding()

            def solve_ode(params: Any, x: jnp.ndarray, cond_embedding: jnp.ndarray) -> jnp.ndarray:
                return diffrax.diffeqsolve(
                    diffrax.ODETerm(vf), t0=0.0, t1=1.0, y0=x, args=(params, cond_embedding), **kwargs
                ).ys[0]

            fn = jax.jit(jax.vmap(solve_ode, in_axes=[None, 0, 0]))
        else:
            vf = guidance.wrap(self._base_velocity())

            def solve_ode(
                params: Any, x: jnp.ndarray, condition: dict[str, jnp.ndarray], encoder_noise: jnp.ndarray
            ) -> jnp.ndarray:
                return diffrax.diffeqsolve(
                    diffrax.ODETerm(vf), t0=0.0, t1=1.0, y0=x, args=(params, condition, encoder_noise), **kwargs
                ).ys[0]

            fn = jax.jit(jax.vmap(solve_ode, in_axes=[None, 0, None, None]))

        self._predict_fn_cache[kwargs_frozen] = fn
        return fn

    def _predict_batched(
        self,
        x: dict[str, ArrayLike],
        condition: dict[str, dict[str, ArrayLike]],
        rng: jax.Array | None = None,
        **kwargs: Any,
    ) -> dict[str, ArrayLike]:
        """Predict every condition in one condition-batched ODE solve (no-guidance path).

        Encodes each condition once, concatenates all conditions' source cells into a single batch (each
        cell tagged with its condition's embedding), integrates them in one vmapped solve, then splits the
        result back per condition. Differing per-condition cell counts are handled by concatenation (no
        padding). Numerically identical to the per-condition loop: each cell's ODE is independent.
        """
        fn = self._get_predict_fn(self._frozen_predict_kwargs(dict(kwargs)))
        keys = list(x)
        # One encoder-noise draw shared across conditions (matches the per-condition loop under one rng).
        encoder_noise = self._encoder_noise(rng)
        groups = list(condition[keys[0]])
        cond_stacked = {g: jnp.concatenate([jnp.asarray(condition[k][g]) for k in keys], axis=0) for g in groups}
        cond_embedding = self._encode_conditions(cond_stacked, encoder_noise)  # (n_conditions, embedding_dim)

        sizes = [int(jnp.asarray(x[k]).shape[0]) for k in keys]
        x_flat = jnp.concatenate([jnp.asarray(x[k]) for k in keys], axis=0)
        emb_flat = jnp.concatenate(
            [
                jnp.broadcast_to(cond_embedding[i : i + 1], (sizes[i], cond_embedding.shape[-1]))
                for i in range(len(keys))
            ],
            axis=0,
        )
        out_flat = fn(self.vf_state_inference.params, x_flat, emb_flat)

        out: dict[str, ArrayLike] = {}
        offset = 0
        for k, n in zip(keys, sizes, strict=True):
            out[k] = out_flat[offset : offset + n]
            offset += n
        return out

    def _predict_jit(
        self,
        x: ArrayLike,
        condition: dict[str, ArrayLike],
        rng: jax.Array | None = None,
        **kwargs: Any,
    ) -> ArrayLike:
        """Predict a single array of source cells ``x`` under one ``condition``. See :meth:`predict`."""
        fn = self._get_predict_fn(self._frozen_predict_kwargs(dict(kwargs)))
        params = self.vf_state_inference.params
        encoder_noise = self._encoder_noise(rng)
        if self._effective_guidance(kwargs.get("guidance_scale", 1.0), warn=False) is None:
            # No guidance: encode once and broadcast the single embedding across the source cells, then
            # run the same per-cell fn the condition-batched path uses.
            emb = self._encode_conditions(condition, encoder_noise)  # (1, embedding_dim)
            emb = jnp.broadcast_to(emb, (jnp.asarray(x).shape[0], emb.shape[-1]))
            return fn(params, x, emb)
        return fn(params, x, condition, encoder_noise)

    def predict(
        self,
        x: ArrayLike | dict[str, ArrayLike],
        condition: dict[str, ArrayLike] | dict[str, dict[str, ArrayLike]],
        rng: jax.Array | None = None,
        **kwargs: Any,
    ) -> ArrayLike | dict[str, ArrayLike]:
        """Predict the translated source ``x`` under condition ``condition``.

        This function solves the ODE learnt with
        the :class:`~cellflow.networks.ConditionalVelocityField`.

        Parameters
        ----------
        x
            A dictionary with keys indicating the name of the condition and values containing
            the input data as arrays.
        condition
            A dictionary with keys indicating the name of the condition and values containing
            the condition of input data as arrays.
        rng
            Random number generator to sample from the latent distribution,
            only used if ``condition_mode='stochastic'``. If :obj:`None`, the
            mean embedding is used.
        kwargs
            Keyword arguments for :func:`diffrax.diffeqsolve`. May also include
            ``guidance_scale`` (a float): a per-call classifier-free guidance scale
            that, when not ``1.0``, applies :class:`ClassifierFreeGuidance` for this
            call (requires :attr:`cfg_enabled`), overriding the construction-time
            ``guidance``. Handy for sweeping ``w`` without rebuilding the solver.

        Returns
        -------
        The push-forward distribution of ``x`` under condition ``condition``.
        """
        if "batched" in kwargs:
            warnings.warn(
                "The `batched` argument is deprecated and will be removed in a future version. "
                "Batched prediction is now the default behavior when passing a dictionary.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs.pop("batched")

        if isinstance(x, dict) and not x:
            return {}

        if isinstance(x, dict):
            # No guidance: batch every condition into one condition-batched solve. Guidance stays on the
            # per-condition loop since it needs the conditional + null embeddings inside the ODE. Same
            # result either way. (Routing resolution is silent; the warning fires once when the fn builds.)
            if self._effective_guidance(kwargs.get("guidance_scale", 1.0), warn=False) is None:
                jax_results = self._predict_batched(x, condition, rng, **kwargs)
            else:
                jax_results = {k: self._predict_jit(x[k], condition[k], rng, **kwargs) for k in x}
            return {k: np.array(v) for k, v in jax_results.items()}
        else:
            x_pred = self._predict_jit(x, condition, rng, **kwargs)
            return np.array(x_pred)
