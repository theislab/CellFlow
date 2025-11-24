from collections.abc import Callable
from functools import partial
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import frozen_dict
from flax.training import train_state
from ott.neural.methods.flows import dynamics
from ott.solvers import utils as solver_utils

from scaleflow import utils
from scaleflow._types import ArrayLike
from scaleflow.networks._velocity_field import ConditionalVelocityField
from scaleflow.solvers.utils import ema_update

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
            see e.g. :func:`scaleflow.utils.match_linear`. If :obj:`None`, no
            matching is performed, and pure probability_path matching :cite:`lipman:22`
            is applied.
        time_sampler
            Time sampler with a ``(rng, n_samples) -> time`` signature, see e.g.
            :func:`ott.solvers.utils.uniform_sampler`.
        kwargs
            Keyword arguments for :meth:`scaleflow.networks.ConditionalVelocityField.create_train_state`.
    """

    def __init__(
        self,
        vf: ConditionalVelocityField,
        probability_path: dynamics.BaseFlow,
        match_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
        time_sampler: Callable[[jax.Array, int], jnp.ndarray] = solver_utils.uniform_sampler,
        phenotype_predictor: Any | None = None,
        loss_weight_gex: float = 1.0,
        loss_weight_functional: float = 1.0,
        **kwargs: Any,
    ):
        self._is_trained: bool = False
        self.vf = vf
        self.condition_encoder_mode = self.vf.condition_mode
        self.condition_encoder_regularization = self.vf.regularization
        self.probability_path = probability_path
        self.time_sampler = time_sampler
        self.match_fn = jax.jit(match_fn) if match_fn is not None else None
        self.ema = kwargs.pop("ema", 1.0)
        self.loss_weight_gex = loss_weight_gex
        self.loss_weight_functional = loss_weight_functional

        self.vf_state = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_state_inference = self.vf.create_train_state(input_dim=self.vf.output_dims[-1], **kwargs)
        self.vf_step_fn = self._get_vf_step_fn()

        self.phenotype_predictor = phenotype_predictor
        if self.phenotype_predictor is not None:
            phenotype_optimizer = kwargs.get("optimizer", optax.adamw(learning_rate=1e-3))
            self.phenotype_state = self.phenotype_predictor.create_train_state(
                rng=kwargs.get("rng", jax.random.PRNGKey(0)),
                optimizer=phenotype_optimizer,
                condition_embedding_dim=self.vf.condition_embedding_dim,
            )
            self.phenotype_step_fn = self._get_phenotype_step_fn()

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

    def _get_phenotype_step_fn(self) -> Callable:  # type: ignore[type-arg]
        @jax.jit
        def phenotype_step_fn(
            rng: jax.Array,
            vf_state: train_state.TrainState,
            phenotype_state: train_state.TrainState,
            conditions: dict[str, jnp.ndarray],
            phenotypes: jnp.ndarray,
            encoder_noise: jnp.ndarray,
        ):
            def loss_fn(
                vf_params: jnp.ndarray,
                phenotype_params: jnp.ndarray,
                conditions: dict[str, jnp.ndarray],
                phenotypes: jnp.ndarray,
                encoder_noise: jnp.ndarray,
                rng: jax.Array,
            ) -> jnp.ndarray:
                rng_encoder = rng
                mean_cond, logvar_cond = vf_state.apply_fn(
                    {"params": vf_params},
                    conditions,
                    rngs={"condition_encoder": rng_encoder},
                    method="get_condition_embedding",
                )
                if self.condition_encoder_mode == "deterministic":
                    cond_embedding = mean_cond
                else:
                    cond_embedding = mean_cond + encoder_noise * jnp.exp(logvar_cond / 2.0)

                phenotype_pred = phenotype_state.apply_fn(
                    {"params": phenotype_params},
                    cond_embedding,
                    training=True,
                )

                mse_loss = jnp.mean((phenotype_pred - phenotypes) ** 2)
                condition_mean_regularization = 0.5 * jnp.mean(mean_cond**2)
                condition_var_regularization = -0.5 * jnp.mean(1 + logvar_cond - jnp.exp(logvar_cond))
                if self.condition_encoder_mode == "stochastic":
                    encoder_loss = condition_mean_regularization + condition_var_regularization
                elif (self.condition_encoder_mode == "deterministic") and (self.condition_encoder_regularization > 0):
                    encoder_loss = condition_mean_regularization
                else:
                    encoder_loss = 0.0
                return mse_loss + encoder_loss

            grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))
            loss, (vf_grads, phenotype_grads) = grad_fn(
                vf_state.params, phenotype_state.params, conditions, phenotypes, encoder_noise, rng
            )
            vf_state_updated = vf_state.replace(
                params=jax.tree_util.tree_map(lambda p, g: p - 0.0 * g, vf_state.params, vf_grads)
            )
            vf_state_updated = vf_state_updated.apply_gradients(grads=vf_grads)
            phenotype_state_updated = phenotype_state.apply_gradients(grads=phenotype_grads)
            return vf_state_updated, phenotype_state_updated, loss

        return phenotype_step_fn

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
            Data batch with keys ``task``, and task-specific data.
            For 'gex' task: ``src_cell_data``, ``tgt_cell_data``, ``condition``.
            For 'functional' task: ``condition``, ``phenotype``.

        Returns
        -------
        Loss value.
        """
        task = batch.get("task", "gex")

        if task == "gex":
            return self._step_gex(rng, batch)
        elif task == "functional":
            return self._step_functional(rng, batch)
        else:
            raise ValueError(f"Unknown task type: {task}")

    def _step_gex(self, rng: jnp.ndarray, batch: dict[str, ArrayLike]) -> float:
        """Step function for gene expression (flow matching) task."""
        src, tgt = batch["src_cell_data"], batch["tgt_cell_data"]
        condition = batch.get("condition")
        rng_resample, rng_time, rng_step_fn, rng_encoder_noise = jax.random.split(rng, 4)
        n = src.shape[0]
        time = self.time_sampler(rng_time, n)
        encoder_noise = jax.random.normal(rng_encoder_noise, (n, self.vf.condition_embedding_dim))

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
        return loss * self.loss_weight_gex

    def _step_functional(self, rng: jnp.ndarray, batch: dict[str, ArrayLike]) -> float:
        """Step function for functional (phenotype prediction) task."""
        if self.phenotype_predictor is None:
            raise ValueError("Cannot train functional task: phenotype_predictor not provided")

        condition = batch["condition"]
        phenotype = batch["phenotype"]
        rng_step_fn, rng_encoder_noise = jax.random.split(rng, 2)
        n = phenotype.shape[0]
        encoder_noise = jax.random.normal(rng_encoder_noise, (n, self.vf.condition_embedding_dim))

        self.vf_state, self.phenotype_state, loss = self.phenotype_step_fn(
            rng_step_fn,
            self.vf_state,
            self.phenotype_state,
            condition,
            phenotype,
            encoder_noise,
        )

        if self.ema == 1.0:
            self.vf_state_inference = self.vf_state
        else:
            self.vf_state_inference = self.vf_state_inference.replace(
                params=ema_update(self.vf_state_inference.params, self.vf_state.params, self.ema)
            )
        return loss * self.loss_weight_functional

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

    def _predict_jit(
        self, x: ArrayLike, condition: dict[str, ArrayLike], rng: jax.Array | None = None, **kwargs: Any
    ) -> ArrayLike:
        """See :meth:`OTFlowMatching.predict`."""
        kwargs.setdefault("dt0", None)
        kwargs.setdefault("solver", diffrax.Tsit5())
        kwargs.setdefault("stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5))
        kwargs = frozen_dict.freeze(kwargs)

        noise_dim = (1, self.vf.condition_embedding_dim)
        use_mean = rng is None or self.condition_encoder_mode == "deterministic"
        rng = utils.default_prng_key(rng)
        encoder_noise = jnp.zeros(noise_dim) if use_mean else jax.random.normal(rng, noise_dim)

        def vf(t: jnp.ndarray, x: jnp.ndarray, args: tuple[dict[str, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
            params = self.vf_state_inference.params
            condition, encoder_noise = args
            return self.vf_state_inference.apply_fn({"params": params}, t, x, condition, encoder_noise, train=False)[0]

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
        return x_pred

    def predict(
        self,
        x: ArrayLike | dict[str, ArrayLike],
        condition: dict[str, ArrayLike] | dict[str, dict[str, ArrayLike]],
        rng: jax.Array | None = None,
        batched: bool = False,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> ArrayLike | dict[str, ArrayLike]:
        """Predict the translated source ``x`` under condition ``condition``.

        This function solves the ODE learnt with
        the :class:`~scaleflow.networks.ConditionalVelocityField`.

        Parameters
        ----------
        x
            A dictionary with keys indicating the name of the condition and values containing
            the input data as arrays. If ``batched=False`` provide an array of shape [batch_size, ...].
        condition
            A dictionary with keys indicating the name of the condition and values containing
            the condition of input data as arrays. If ``batched=False`` provide an array of shape
            [batch_size, ...].
        rng
            Random number generator to sample from the latent distribution,
            only used if ``condition_mode='stochastic'``. If :obj:`None`, the
            mean embedding is used.
        batched
            Whether to use batched prediction. This is only supported if the input has
            the same number of cells for each condition. For example, this works when using
            :class:`~scaleflow.data.ValidationSampler` to sample the validation data.
        show_progress
            Whether to show a progress bar when predicting over multiple conditions.
        kwargs
            Keyword arguments for :func:`diffrax.diffeqsolve`.

        Returns
        -------
        The push-forward distribution of ``x`` under condition ``condition``.
        """
        if batched and not x:
            return {}

        if batched:
            keys = sorted(x.keys())
            condition_keys = sorted(set().union(*(condition[k].keys() for k in keys)))
            _predict_jit = jax.jit(lambda x, condition: self._predict_jit(x, condition, rng, **kwargs))
            batched_predict = jax.vmap(_predict_jit, in_axes=(0, dict.fromkeys(condition_keys, 0)))
            # assert that the number of cells is the same for each condition
            n_cells = x[keys[0]].shape[0]
            for k in keys:
                assert x[k].shape[0] == n_cells, "The number of cells must be the same for each condition"
            src_inputs = jnp.stack([x[k] for k in keys], axis=0)
            batched_conditions = {}
            for cond_key in condition_keys:
                batched_conditions[cond_key] = jnp.stack([condition[k][cond_key] for k in keys])

            pred_targets = batched_predict(src_inputs, batched_conditions)
            return {k: pred_targets[i] for i, k in enumerate(keys)}
        elif isinstance(x, dict):
            if show_progress:
                from tqdm import tqdm

                predict_fn = partial(self._predict_jit, rng=rng, **kwargs)
                results = {}
                keys = sorted(x.keys())
                for key in tqdm(keys, desc="Predicting conditions", leave=False):
                    results[key] = predict_fn(x[key], condition[key])
                return results
            else:
                return jax.tree.map(
                    partial(self._predict_jit, rng=rng, **kwargs),
                    x,
                    condition,  # type: ignore[attr-defined]
                )
        else:
            x_pred = self._predict_jit(x, condition, rng, **kwargs)
            return np.array(x_pred)

    def predict_phenotype(
        self,
        condition: dict[str, ArrayLike],
        rng: jax.Array | None = None,
    ) -> ArrayLike:
        """Predict phenotype from condition.

        Parameters
        ----------
        condition
            A dictionary with keys indicating the name of the condition and values containing
            the condition embeddings as arrays of shape [batch_size, max_combination_length, emb_dim].
        rng
            Random number generator to sample from the latent distribution,
            only used if ``condition_mode='stochastic'``. If :obj:`None`, the
            mean embedding is used.

        Returns
        -------
        Predicted phenotype of shape [batch_size, output_dim].
        """
        if self.phenotype_predictor is None:
            raise ValueError("Cannot predict phenotype: phenotype_predictor not provided")

        use_mean = rng is None or self.condition_encoder_mode == "deterministic"
        rng = utils.default_prng_key(rng)
        n = next(iter(condition.values())).shape[0]
        encoder_noise = (
            jnp.zeros((n, self.vf.condition_embedding_dim))
            if use_mean
            else jax.random.normal(rng, (n, self.vf.condition_embedding_dim))
        )

        mean_cond, logvar_cond = self.vf.apply(
            {"params": self.vf_state_inference.params},
            condition,
            method="get_condition_embedding",
        )

        if self.condition_encoder_mode == "deterministic":
            cond_embedding = mean_cond
        else:
            cond_embedding = mean_cond + encoder_noise * jnp.exp(logvar_cond / 2.0)

        phenotype_pred = self.phenotype_state.apply_fn(
            {"params": self.phenotype_state.params},
            cond_embedding,
            training=False,
        )

        return np.array(phenotype_pred)

    @property
    def is_trained(self) -> bool:
        """Whether the model is trained."""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        self._is_trained = value
