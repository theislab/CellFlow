from collections.abc import Callable
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from ott.neural.methods.flows import dynamics
from ott.solvers import utils as solver_utils

from cellflow import utils
from cellflow._types import ArrayLike
from cellflow.networks._velocity_field import ConditionalVelocityField

__all__ = ["OTFlowMatching"]


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
        kwargs
            Keyword arguments for :meth:`cellflow.networks.ConditionalVelocityField.create_train_state`.
    """

    def __init__(
        self,
        vf: ConditionalVelocityField,
        probability_path: dynamics.BaseFlow,
        match_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
        time_sampler: Callable[[jax.Array, int], jnp.ndarray] = solver_utils.uniform_sampler,
        mlp_eta: Any | None = None,
        mlp_xi: Any | None = None,
        **kwargs: Any,
    ):
        self._is_trained: bool = False
        self.vf = vf
        self.condition_encoder_mode = self.vf.condition_mode
        self.condition_encoder_regularization = self.vf.regularization
        self.probability_path = probability_path
        self.time_sampler = time_sampler
        self.match_fn = jax.jit(match_fn)
        self.kwargs = kwargs
        self.metrics = {"loss": [], "loss_eta": [], "loss_xi": []}
        self.mlp_eta = mlp_eta
        self.mlp_xi = mlp_xi

        self.vf_state = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.eta_state: train_state.TrainState | None = None
        self.xi_state: train_state.TrainState | None = None

        self.vf_step_fn = self._get_vf_step_fn()
        self.rescaling_step_fn = self._get_rescaling_step_fn()

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
            def vf_loss_fn(
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

            grad_fn = jax.value_and_grad(vf_loss_fn)
            loss, grads = grad_fn(vf_state.params, time, source, target, conditions, encoder_noise, rng)
            return vf_state.apply_gradients(grads=grads), loss

        return vf_step_fn

    def _get_rescaling_step_fn(self) -> Callable:  # type: ignore[type-arg]
        @jax.jit
        def rescaling_step_fn(
            rng: jax.Array,
            eta_state: train_state.TrainState | None,
            xi_state: train_state.TrainState | None,
            x_eta: jnp.ndarray,
            y_xi: jnp.ndarray,
            a: jnp.ndarray,
            b: jnp.ndarray,
            expectation_reweighting_eta: float,
            expectation_reweighting_xi: float,
        ):
            def loss_a_fn(
                params_eta: jnp.ndarray,
                apply_fn_eta: Callable,
                x: jnp.ndarray,
                a: jnp.ndarray,
                expectation_reweighting: float,
            ) -> tuple[float, jnp.ndarray]:
                eta_predictions = apply_fn_eta({"params": params_eta}, x)
                loss = (
                    optax.l2_loss(eta_predictions[:, 0], a).mean()
                    + optax.l2_loss(jnp.mean(eta_predictions), expectation_reweighting).mean()
                )
                return loss, eta_predictions

            def loss_b_fn(
                params_xi: jnp.ndarray,
                apply_fn_xi: Callable,
                x: jnp.ndarray,
                b: jnp.ndarray,
                expectation_reweighting: float,
            ) -> tuple[float, jnp.ndarray]:
                xi_predictions = apply_fn_xi({"params": params_xi}, x)
                loss = (
                    optax.l2_loss(xi_predictions[:, 0], b).mean()
                    + optax.l2_loss(jnp.mean(xi_predictions), expectation_reweighting).mean()
                )
                return loss, xi_predictions

            if eta_state is not None:
                grad_a_fn = jax.value_and_grad(loss_a_fn, argnums=0, has_aux=True)
                (loss_a, eta_predictions), grads_eta = grad_a_fn(
                    eta_state.params,
                    eta_state.apply_fn,
                    x_eta,
                    a,
                    expectation_reweighting_eta,
                )
                new_eta_state = eta_state.apply_gradients(grads=grads_eta)
            else:
                loss_a = 0.0
                eta_predictions = jnp.zeros_like(x_eta)
                new_eta_state = None

            if xi_state is not None:
                grad_b_fn = jax.value_and_grad(loss_b_fn, argnums=0, has_aux=True)
                (loss_b, xi_predictions), grads_xi = grad_b_fn(
                    xi_state.params,
                    xi_state.apply_fn,
                    y_xi,
                    b,
                    expectation_reweighting_xi,
                )
                new_xi_state = xi_state.apply_gradients(grads=grads_xi)
            else:
                loss_b = 0.0
                xi_predictions = jnp.zeros_like(y_xi)
                new_xi_state = None

            return new_eta_state, new_xi_state, loss_a, loss_b, eta_predictions, xi_predictions

        return rescaling_step_fn

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
        a = src
        b = tgt
        integration_eta = jnp.sum(a)
        integration_xi = jnp.sum(b)
        condition = batch.get("condition")
        rng_resample, rng_time, rng_step_fn, rng_encoder_noise = jax.random.split(rng, 4)
        n = src.shape[0]
        time = self.time_sampler(rng_time, n)
        encoder_noise = jax.random.normal(rng_encoder_noise, (n, self.vf.condition_embedding_dim))
        # TODO: test whether it's better to sample the same noise for all samples or different ones

        if self.match_fn is not None:
            tmat = self.match_fn(src, tgt)
            a, b = tmat.sum(axis=0), tmat.sum(axis=1)
            src_ixs, tgt_ixs = solver_utils.sample_joint(rng_resample, tmat)
            src, tgt = src[src_ixs], tgt[tgt_ixs]

        if self.mlp_eta is not None and self.eta_state is None:
            opt_eta = self.kwargs.get("opt_eta", optax.adamw(learning_rate=1e-4, weight_decay=1e-10))
            self.eta_state = self.mlp_eta.create_train_state(
                rng,
                opt_eta,
                input_dim=src.shape[-1],
            )
        if self.mlp_xi is not None and self.xi_state is None:
            opt_xi = self.kwargs.get("opt_xi", optax.adamw(learning_rate=1e-4, weight_decay=1e-10))
            self.xi_state = self.mlp_xi.create_train_state(
                rng,
                opt_xi,
                input_dim=tgt.shape[-1],
            )

    
        self.eta_state, self.xi_state, loss_a, loss_b, eta_predictions, xi_predictions = self.rescaling_step_fn(
            rng,
            self.eta_state,
            self.xi_state,
            src[:],
            tgt[:],
            a * len(src),
            b * len(tgt),
            integration_eta,
            integration_xi,
        )

        self.metrics["loss_xi"].append(loss_b)
        self.metrics["loss_eta"].append(loss_a)

        self.vf_state, vf_loss = self.vf_step_fn(
            rng_step_fn,
            self.vf_state,
            time,
            src,
            tgt,
            condition,
            encoder_noise,
        )
        self.metrics["loss"].append(vf_loss)

        return vf_loss + loss_a + loss_b

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
        self, x: ArrayLike, condition: dict[str, ArrayLike], rng: jax.Array | None = None, **kwargs: Any
    ) -> ArrayLike:
        """Predict the translated source ``x`` under condition ``condition``.

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

        def vf(t: jnp.ndarray, x: jnp.ndarray, args: tuple[dict[str, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
            params = self.vf_state.params
            condition, encoder_noise = args
            return self.vf_state.apply_fn({"params": params}, t, x, condition, encoder_noise, train=False)[0]

        def solve_ode(x: jnp.ndarray, condition: dict[str, jnp.ndarray], encoder_noise: jnp.ndarray) -> jnp.ndarray:
            ode_term = diffrax.ODETerm(vf)
            result = diffrax.diffeqsolve(
                ode_term,
                t0=0.0,
                t1=1.0,
                y0=x,
                args=(condition, encoder_noise),
                **kwargs,
            )
            return result.ys[0]

        x_pred = jax.jit(jax.vmap(solve_ode, in_axes=[0, None, None]))(x, condition, encoder_noise)
        return np.array(x_pred)

    @property
    def is_trained(self) -> bool:
        """Whether the model is trained."""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        self._is_trained = value
