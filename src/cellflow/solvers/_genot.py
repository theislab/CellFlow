import functools
import warnings
from collections.abc import Callable
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import frozen_dict
from flax.training import train_state
from ott.solvers import utils as solver_utils

from cellflow import utils
from cellflow._compat import BaseFlow
from cellflow._types import ArrayLike
from cellflow.model._utils import _multivariate_normal
from cellflow.solvers._base import BaseSolver
from cellflow.solvers._otfm import ClassifierFreeGuidance, Guidance, VelocityFn

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
        super().__init__(vf, probability_path, time_sampler)
        self.data_match_fn = jax.jit(data_match_fn)
        self.source_dim = source_dim
        self.guidance = guidance
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
                # GENOT target is the latent->target path velocity (target - latent); source only conditions v.
                u_t = self.probability_path.compute_ut(t, x_t, latent, target)
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
            jax_results = {k: self._predict_jit(x[k], condition[k], rng, rng_genot, **kwargs) for k in x}
            return {k: np.array(v) for k, v in jax_results.items()}
        else:
            x_pred = self._predict_jit(x, condition, rng, rng_genot, **kwargs)
            return np.array(x_pred)

    @property
    def cfg_enabled(self) -> bool:
        """Whether classifier-free guidance is available at predict time.

        Guidance needs a meaningful unconditional velocity ``v_null``, which only exists
        when the velocity field was trained with condition dropout
        (``condition_dropout_prob > 0``). When ``False``, a per-call ``guidance_scale`` is
        ignored (with a warning) and the plain conditional velocity is used.
        """
        return float(getattr(self.vf, "condition_dropout_prob", 0.0)) > 0.0

    def _base_velocity(self) -> VelocityFn:
        """Base (conditional) velocity closure for the predict path.

        Signature ``(t, x, args, force_uncond=False) -> velocity`` with ``args`` being
        ``(params, x_0, condition, encoder_noise)``. Nulling only the condition (``x_0`` is
        kept) gives the unconditional source→target velocity used by guidance.
        """

        def vf(
            t: float,
            x: jnp.ndarray,
            args: tuple[Any, jnp.ndarray, dict[str, jnp.ndarray], jnp.ndarray],
            force_uncond: bool = False,
        ) -> jnp.ndarray:
            params, x_0, condition, encoder_noise = args
            return self.vf_state.apply_fn(
                {"params": params}, t, x, x_0, condition, encoder_noise, train=False, force_uncond=force_uncond
            )[0]

        return vf

    def _get_predict_fn(self, kwargs_frozen: frozen_dict.FrozenDict) -> Callable:
        """Build and cache a jit+vmap predict function for the given diffrax kwargs.

        The base velocity from :meth:`_base_velocity` is wrapped by guidance when it
        applies: a per-call ``guidance_scale != 1.0`` (popped here, requires
        :attr:`cfg_enabled`) builds a :class:`~cellflow.solvers.ClassifierFreeGuidance`
        for this call, overriding the construction-time ``guidance``; otherwise the
        construction-time ``guidance`` is used (``None`` = plain conditional velocity).
        The returned function is created once per unique set of kwargs, then reused.
        """
        if kwargs_frozen in self._predict_fn_cache:
            return self._predict_fn_cache[kwargs_frozen]

        kwargs = dict(kwargs_frozen)
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

        vf = self._base_velocity()
        if guidance is not None:
            vf = guidance.wrap(vf)

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

        fn = jax.jit(jax.vmap(solve_ode, in_axes=[None, 0, 0, None, None]))
        self._predict_fn_cache[kwargs_frozen] = fn
        return fn

    def _predict_jit(
        self,
        x: ArrayLike,
        condition: dict[str, ArrayLike] | None = None,
        rng: ArrayLike | None = None,
        rng_genot: ArrayLike | None = None,
        **kwargs: Any,
    ) -> ArrayLike | tuple[ArrayLike, diffrax.Solution]:
        kwargs.setdefault("dt0", None)
        kwargs.setdefault("solver", diffrax.Tsit5())
        kwargs.setdefault("stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5))
        kwargs_frozen = frozen_dict.freeze(kwargs)

        noise_dim = (1, self.vf.condition_embedding_dim)
        use_mean = rng is None or self.condition_encoder_mode == "deterministic"
        rng = utils.default_prng_key(rng)
        encoder_noise = jnp.zeros(noise_dim) if use_mean else jax.random.normal(rng, noise_dim)
        rng_genot = utils.default_prng_key(rng_genot)
        latent = self.latent_noise_fn(rng_genot, (x.shape[0],))

        predict_fn = self._get_predict_fn(kwargs_frozen)
        return predict_fn(self.vf_state.params, latent, x, condition, encoder_noise)
