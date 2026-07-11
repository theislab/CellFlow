import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import frozen_dict
from flax.training import train_state

from cellflow import utils
from cellflow._types import ArrayLike

__all__ = ["BaseSolver", "ClassifierFreeGuidance", "Guidance"]

# A velocity closure with the diffrax signature ``(t, x, args) -> velocity``, where
# ``args`` is the solver-specific tuple of ``(params, ..., condition, encoder_noise)``.
VelocityFn = Callable[[jnp.ndarray, jnp.ndarray, tuple[Any, ...]], jnp.ndarray]


@runtime_checkable
class Guidance(Protocol):
    """Pluggable transform combining a conditional and unconditional velocity field.

    A guidance strategy receives the base (conditional) velocity closure ``v_cond``
    and the unconditional velocity closure ``v_null`` (both with the same
    ``(t, x, args) -> velocity`` signature) and returns a new velocity closure. This
    keeps guidance agnostic to each solver's ``args`` layout: the solver supplies the
    two closures via :meth:`BaseSolver._base_velocity` and
    :meth:`BaseSolver._null_velocity`.
    """

    def wrap(self, v_cond: VelocityFn, v_null: VelocityFn) -> VelocityFn:
        """Combine the conditional ``v_cond`` and unconditional ``v_null`` velocities."""
        ...


class ClassifierFreeGuidance:
    """Classifier-free guidance combining conditional and unconditional velocities.

    Combines the conditional velocity ``v_cond`` with the unconditional velocity
    ``v_null`` according to ``v = v_null + scale * (v_cond - v_null)``.

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

    def wrap(self, v_cond: VelocityFn, v_null: VelocityFn) -> VelocityFn:
        """Return a velocity closure computing ``v_null + scale * (v_cond - v_null)``."""
        scale = self.scale

        def guided_vf(t: jnp.ndarray, x: jnp.ndarray, args: tuple[Any, ...]) -> jnp.ndarray:
            vc = v_cond(t, x, args)
            vn = v_null(t, x, args)
            return vn + scale * (vc - vn)

        return guided_vf


class BaseSolver(ABC):
    """Shared behavior for conditional flow-matching solvers.

    Holds the pieces that :class:`~cellflow.solvers.OTFlowMatching` and
    :class:`~cellflow.solvers.GENOT` implement identically: the training flag, the
    condition-encoder regularization loss, condition-embedding extraction, and the
    prediction plumbing (diffrax defaults, encoder-noise sampling, per-kwargs jit
    cache, and the array/dict prediction dispatch).

    Subclasses call ``super().__init__(vf, probability_path, time_sampler)`` and
    then wire their solver-specific state; they must implement :meth:`_predict_jit`
    and :meth:`_build_predict_fn`. The attributes set here are:

    - ``vf`` - the conditional velocity field module.
    - ``vf_state`` - the training :class:`~flax.training.train_state.TrainState`.
    - ``condition_encoder_mode`` / ``condition_encoder_regularization`` - copied
      from the velocity field and used by :meth:`_encoder_loss`.
    - ``probability_path`` / ``time_sampler`` - the flow objects.
    - ``_predict_fn_cache`` - the per-kwargs prediction cache.
    """

    def __init__(
        self,
        vf: Any,
        probability_path: Any,
        time_sampler: Callable[[jax.Array, int], jnp.ndarray],
        guidance: Guidance | None = None,
    ) -> None:
        """Set the attributes shared by every solver.

        Called via ``super().__init__`` by each subclass before it wires its
        solver-specific state (matching function, dimensions, train state, step
        function). Keeps the common attribute names and defaults in one place.
        """
        self._is_trained: bool = False
        self.vf = vf
        self.condition_encoder_mode = vf.condition_mode
        self.condition_encoder_regularization = vf.regularization
        self.probability_path = probability_path
        self.time_sampler = time_sampler
        self.guidance = guidance
        self._predict_fn_cache: dict[frozen_dict.FrozenDict, Any] = {}

    @property
    def cfg_enabled(self) -> bool:
        """Whether classifier-free guidance is available at predict time.

        Guidance needs a meaningful unconditional velocity ``v_null``, which only
        exists when the velocity field was trained with condition dropout
        (``condition_dropout_prob > 0``). When :obj:`False`, a per-call
        ``guidance_scale`` is ignored (with a warning) and the plain conditional
        velocity is used.
        """
        return float(getattr(self.vf, "condition_dropout_prob", 0.0)) > 0.0

    @abstractmethod
    def _base_velocity(self) -> VelocityFn:
        """Return the base (conditional) velocity closure used on the predict path.

        The closure has the diffrax ``(t, x, args) -> velocity`` signature and
        evaluates the inference velocity field conditionally (``force_uncond=False``).
        Solvers differ in their ``args`` layout.
        """

    @abstractmethod
    def _null_velocity(self) -> VelocityFn:
        """Return the unconditional velocity closure (``force_uncond=True``).

        Same signature as :meth:`_base_velocity`; used by guidance to build the
        classifier-free combination.
        """

    def _guided_velocity(self, kwargs: dict[str, Any]) -> tuple[VelocityFn, dict[str, Any]]:
        """Select the (possibly guided) velocity closure and strip guidance kwargs.

        Pops ``guidance_scale`` from ``kwargs`` (default ``1.0``; not a diffrax arg).
        When it is not ``1.0`` it builds a :class:`ClassifierFreeGuidance` for this
        call, overriding the construction-time ``guidance`` — the convenient scalar
        entry point (e.g. sweeping ``w`` at validation). It requires
        :attr:`cfg_enabled`; otherwise it is ignored (with a warning) and the plain
        conditional velocity is used. With ``guidance_scale == 1.0`` the
        construction-time ``guidance`` applies (``None`` means no guidance).

        Returns the velocity closure and the remaining ``kwargs`` (diffrax args).
        """
        guidance_scale = float(kwargs.pop("guidance_scale", 1.0))
        guidance = self.guidance
        if guidance_scale != 1.0:
            if self.cfg_enabled:
                # v = v_null + scale·(v_cond − v_null); overrides construction-time guidance.
                guidance = ClassifierFreeGuidance(scale=guidance_scale)
            else:
                warnings.warn(
                    f"guidance_scale={guidance_scale} ignored: the velocity field was not trained "
                    "with classifier-free guidance (condition_dropout_prob == 0), so the "
                    "unconditional velocity is undefined. Using the plain conditional velocity.",
                    stacklevel=2,
                )
                guidance = None

        v_cond = self._base_velocity()
        if guidance is None:
            return v_cond, kwargs
        return guidance.wrap(v_cond, self._null_velocity()), kwargs

    @property
    def _inference_state(self):
        """Train state whose params drive prediction and condition embedding.

        Defaults to the training state ``vf_state``. :class:`OTFlowMatching`
        overrides this to return its (optionally EMA-averaged) inference state.
        """
        return self.vf_state

    def _encoder_loss(self, mean_cond: jnp.ndarray, logvar_cond: jnp.ndarray) -> jnp.ndarray | float:
        """Condition-encoder regularization added to the flow-matching loss.

        Mirrors the VAE-style objective: a mean penalty plus, in ``stochastic``
        mode, the KL variance term. Returns ``0.0`` when the encoder is
        deterministic and unregularized. ``condition_encoder_mode`` and
        ``condition_encoder_regularization`` are Python values baked in at trace
        time, so this is safe to call inside a jitted loss closure.
        """
        mean_regularization = 0.5 * jnp.mean(mean_cond**2)
        var_regularization = -0.5 * jnp.mean(1 + logvar_cond - jnp.exp(logvar_cond))
        if self.condition_encoder_mode == "stochastic":
            return mean_regularization + var_regularization
        if (self.condition_encoder_mode == "deterministic") and (self.condition_encoder_regularization > 0):
            return mean_regularization
        return 0.0

    @abstractmethod
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
        """Evaluate the velocity field, returning ``(v_t, mean_cond, logvar_cond)``.

        The solvers differ only in whether the source ``x_0`` is passed to the
        network: :class:`OTFlowMatching` conditions on ``condition`` alone, while
        :class:`GENOT` additionally threads ``source`` through the velocity field.
        """

    def _flow_matching_loss(
        self,
        params: Any,
        vf_state: train_state.TrainState,
        rng: jax.Array,
        t: jnp.ndarray,
        x_0: jnp.ndarray,
        source: jnp.ndarray,
        target: jnp.ndarray,
        condition: dict[str, jnp.ndarray] | None,
        encoder_noise: jnp.ndarray,
    ) -> jnp.ndarray:
        """Flow-matching loss plus condition-encoder regularization.

        Integrates the probability path from ``x_0`` (the source cells for OTFM,
        sampled latent noise for GENOT) to ``target`` and regresses the velocity
        field onto the path's conditional vector field. ``params`` is the first
        argument so this can be wrapped directly with :func:`jax.value_and_grad`.
        """
        rng_flow, rng_encoder, rng_dropout = jax.random.split(rng, 3)
        x_t = self.probability_path.compute_xt(rng_flow, t, x_0, target)
        v_t, mean_cond, logvar_cond = self._apply_vf(
            vf_state,
            params,
            t,
            x_t,
            source,
            condition,
            encoder_noise,
            rngs={"dropout": rng_dropout, "condition_encoder": rng_encoder},
        )
        # The regression target is the derivative of the path from ``x_0`` to ``target``,
        # so it must use ``x_0`` (source cells for OTFM, latent for GENOT) — the same point
        # the inference ODE integrates from — NOT the conditioning ``source``. For OTFM the
        # two coincide; for GENOT they differ, and using ``source`` here biases every sample.
        u_t = self.probability_path.compute_ut(t, x_t, x_0, target)
        flow_matching_loss = jnp.mean((v_t - u_t) ** 2)
        return flow_matching_loss + self._encoder_loss(mean_cond, logvar_cond)

    def _get_vf_step_fn(self) -> Callable:
        """Build the jitted train step: one gradient update of the velocity field.

        The path start ``x_0`` is passed explicitly by the caller (source cells for
        OTFM, sampled latent for GENOT). Returns ``(new_state, loss)``.
        """

        @jax.jit
        def vf_step_fn(
            rng: jax.Array,
            vf_state: train_state.TrainState,
            time: jnp.ndarray,
            source: jnp.ndarray,
            target: jnp.ndarray,
            x_0: jnp.ndarray,
            conditions: dict[str, jnp.ndarray] | None,
            encoder_noise: jnp.ndarray,
        ):
            grad_fn = jax.value_and_grad(self._flow_matching_loss)
            loss, grads = grad_fn(vf_state.params, vf_state, rng, time, x_0, source, target, conditions, encoder_noise)
            return vf_state.apply_gradients(grads=grads), loss

        return vf_step_fn

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
            {"params": self._inference_state.params},
            condition,
            method="get_condition_embedding",
        )
        if return_as_numpy:
            return np.asarray(cond_mean), np.asarray(cond_logvar)
        return cond_mean, cond_logvar

    @staticmethod
    def _resolve_diffrax_kwargs(kwargs: dict[str, Any]) -> frozen_dict.FrozenDict:
        """Fill in the default diffrax solver options and freeze for cache keying.

        Mutates ``kwargs`` in place with the shared defaults (``dt0``, ``solver``,
        ``stepsize_controller``) and returns a frozen copy usable as a
        :attr:`_predict_fn_cache` key.
        """
        kwargs.setdefault("dt0", None)
        kwargs.setdefault("solver", diffrax.Tsit5())
        kwargs.setdefault("stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5))
        return frozen_dict.freeze(kwargs)

    def _sample_encoder_noise(self, rng: jax.Array | None) -> jnp.ndarray:
        """Sample (or zero) the condition-encoder noise for a single prediction.

        Returns zeros when ``rng`` is :obj:`None` or the encoder is deterministic
        (use the mean embedding), otherwise a standard-normal sample of shape
        ``(1, condition_embedding_dim)``.
        """
        noise_dim = (1, self.vf.condition_embedding_dim)
        use_mean = rng is None or self.condition_encoder_mode == "deterministic"
        rng = utils.default_prng_key(rng)
        return jnp.zeros(noise_dim) if use_mean else jax.random.normal(rng, noise_dim)

    def _get_predict_fn(self, kwargs_frozen: frozen_dict.FrozenDict) -> Callable:
        """Return a jit+vmap predict function for the given diffrax kwargs, cached.

        The function is built once per unique set of diffrax kwargs by the
        subclass hook :meth:`_build_predict_fn`, then reused on later calls.
        """
        if kwargs_frozen in self._predict_fn_cache:
            return self._predict_fn_cache[kwargs_frozen]
        fn = self._build_predict_fn(dict(kwargs_frozen))
        self._predict_fn_cache[kwargs_frozen] = fn
        return fn

    @abstractmethod
    def _build_predict_fn(self, kwargs: dict[str, Any]) -> Callable:
        """Build the (uncached) jit+vmap predict function for ``kwargs``.

        Implemented by each solver, which owns its velocity closure, ODE solve,
        and ``vmap`` axes. Called by :meth:`_get_predict_fn`.
        """

    @abstractmethod
    def _predict_jit(self, x: ArrayLike, condition: dict[str, ArrayLike], rng: jax.Array | None, **kwargs: Any):
        """Run the jitted prediction for a single ``(x, condition)`` pair.

        Implemented by each solver on top of the shared helpers
        :meth:`_resolve_diffrax_kwargs`, :meth:`_sample_encoder_noise`, and
        :meth:`_get_predict_fn`.
        """

    def predict(
        self,
        x: ArrayLike | dict[str, ArrayLike],
        condition: dict[str, ArrayLike] | dict[str, dict[str, ArrayLike]] | None = None,
        rng: jax.Array | None = None,
        *,
        batched: bool | None = None,
        **kwargs: Any,
    ) -> ArrayLike | dict[str, ArrayLike]:
        """Predict the push-forward of ``x`` under ``condition``.

        Dispatches over a single array or a dict of arrays keyed by condition,
        delegating each to :meth:`_predict_jit`. Extra keyword arguments are
        forwarded to :func:`diffrax.diffeqsolve` (and to the solver's
        ``_predict_jit``, e.g. GENOT's ``rng_genot``).

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
        batched
            Deprecated and ignored. Batched prediction is now the default behavior
            when passing a dictionary; passing this argument emits a
            :class:`DeprecationWarning`.
        kwargs
            Keyword arguments for :func:`diffrax.diffeqsolve`.

        Returns
        -------
        The push-forward distribution of ``x`` under condition ``condition``.
        """
        if batched is not None:
            warnings.warn(
                "The `batched` argument is deprecated and will be removed in a future version. "
                "Batched prediction is now the default behavior when passing a dictionary.",
                DeprecationWarning,
                stacklevel=2,
            )

        if isinstance(x, dict) and not x:
            return {}

        if isinstance(x, dict):
            jax_results = {k: self._predict_jit(x[k], condition[k], rng, **kwargs) for k in x}
            return {k: np.array(v) for k, v in jax_results.items()}
        else:
            x_pred = self._predict_jit(x, condition, rng, **kwargs)
            return np.array(x_pred)

    @property
    def is_trained(self) -> bool:
        """Whether the model is trained."""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        self._is_trained = value
