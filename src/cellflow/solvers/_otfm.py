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
from cellflow.solvers.utils import ema_update

__all__ = ["OTFlowMatching", "ClassifierFreeGuidance", "Guidance"]

# A velocity closure with the diffrax signature ``(t, x, args) -> velocity``,
# where ``args`` is ``(params, condition, encoder_noise)``.
VelocityFn = Callable[[jnp.ndarray, jnp.ndarray, tuple[Any, ...]], jnp.ndarray]


@runtime_checkable
class Guidance(Protocol):
    """Pluggable transform applied to the base velocity field on the predict path.

    A guidance strategy receives the base (conditional) velocity closure and the
    inference train state, and returns a new velocity closure with the same
    ``(t, x, args) -> velocity`` signature.
    """

    def wrap(self, vf: VelocityFn, inference_state: train_state.TrainState) -> VelocityFn:
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

    def wrap(self, vf: VelocityFn, inference_state: train_state.TrainState) -> VelocityFn:
        """Return a velocity closure computing ``v_null + scale * (v_cond - v_null)``."""
        scale = self.scale

        def guided_vf(t: jnp.ndarray, x: jnp.ndarray, args: tuple[Any, ...]) -> jnp.ndarray:
            params, condition, encoder_noise = args
            v_cond = vf(t, x, args)
            v_null = inference_state.apply_fn(
                {"params": params},
                t,
                x,
                condition,
                encoder_noise,
                train=False,
                force_uncond=True,
            )[0]
            return v_null + scale * (v_cond - v_null)

        return guided_vf


class OTFlowMatching:
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
        self._is_trained: bool = False
        self.vf = vf
        self.condition_encoder_mode = self.vf.condition_mode
        self.condition_encoder_regularization = self.vf.regularization
        self.probability_path = probability_path
        self.time_sampler = time_sampler
        self.match_fn = jax.jit(match_fn)
        self.guidance = guidance
        self.ema = kwargs.pop("ema", 1.0)

        self.vf_state = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_state_inference = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_step_fn = self._get_vf_step_fn()
        self._predict_fn_cache: dict[frozen_dict.FrozenDict, Any] = {}

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
            {"params": self.vf_state_inference.params},
            condition,
            method="get_condition_embedding",
        )
        if return_as_numpy:
            return np.asarray(cond_mean), np.asarray(cond_logvar)
        return cond_mean, cond_logvar

    def _base_velocity(self) -> VelocityFn:
        """Return the base (conditional) velocity closure used on the predict path.

        The closure has the diffrax ``(t, x, args) -> velocity`` signature, with
        ``args`` being ``(params, condition, encoder_noise)``, and evaluates the
        inference velocity field conditionally (``force_uncond=False``).
        """

        def vf(t: jnp.ndarray, x: jnp.ndarray, args: tuple[Any, dict[str, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
            params, condition, encoder_noise = args
            return self.vf_state_inference.apply_fn({"params": params}, t, x, condition, encoder_noise, train=False)[0]

        return vf

    def _get_predict_fn(self, kwargs_frozen: frozen_dict.FrozenDict) -> Callable:
        """Build and cache a jit+vmap predict function for the given diffrax kwargs.

        The base velocity closure is produced by :meth:`_base_velocity` and, when a
        guidance strategy is set via ``guidance``, wrapped by it. With
        ``guidance=None`` the closure is exactly the base conditional velocity, so
        prediction is unchanged (no unconditional velocity is computed).

        The returned function is created once per unique set of diffrax kwargs,
        then reused on subsequent calls.
        """
        if kwargs_frozen in self._predict_fn_cache:
            return self._predict_fn_cache[kwargs_frozen]

        kwargs = dict(kwargs_frozen)

        vf = self._base_velocity()
        if self.guidance is not None:
            vf = self.guidance.wrap(vf, self.vf_state_inference)

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

        fn = jax.jit(jax.vmap(solve_ode, in_axes=[None, 0, None, None]))
        self._predict_fn_cache[kwargs_frozen] = fn
        return fn

    def _predict_jit(
        self,
        x: ArrayLike,
        condition: dict[str, ArrayLike],
        rng: jax.Array | None = None,
        **kwargs: Any,
    ) -> ArrayLike:
        """See :meth:`OTFlowMatching.predict`."""
        kwargs.setdefault("dt0", None)
        kwargs.setdefault("solver", diffrax.Tsit5())
        kwargs.setdefault("stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5))
        kwargs_frozen = frozen_dict.freeze(kwargs)

        noise_dim = (1, self.vf.condition_embedding_dim)
        use_mean = rng is None or self.condition_encoder_mode == "deterministic"
        rng = utils.default_prng_key(rng)
        encoder_noise = jnp.zeros(noise_dim) if use_mean else jax.random.normal(rng, noise_dim)

        predict_fn = self._get_predict_fn(kwargs_frozen)
        return predict_fn(self.vf_state_inference.params, x, condition, encoder_noise)

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
            Keyword arguments for :func:`diffrax.diffeqsolve`.

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
