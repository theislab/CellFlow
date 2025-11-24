from collections.abc import Callable
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
from ott.neural.methods.flows import dynamics
from ott.solvers import utils as solver_utils

from scaleflow import utils
from scaleflow._types import ArrayLike
from scaleflow.networks._velocity_field import MultiTaskConditionalVelocityField
from scaleflow.solvers.utils import ema_update

__all__ = ["MultiTaskOTFlowMatching"]


class MultiTaskOTFlowMatching:
    """Multi-task OT Flow Matching for both single-cell and phenotype prediction.

    This solver extends the standard OT Flow Matching to handle both flow matching
    for single-cell data and phenotype prediction tasks, enabling transfer learning
    between the two modalities through shared condition encodings.

    Parameters
    ----------
    vf
        Multi-task velocity field parameterized by a neural network.
    probability_path
        Probability path between the source and the target distributions.
    match_fn
        Function to match samples from the source and the target distributions.
    time_sampler
        Time sampler with a ``(rng, n_samples) -> time`` signature.
    phenotype_loss_weight
        Weight for the phenotype prediction loss relative to flow matching loss.
    ema
        Exponential moving average parameter for inference state.
    kwargs
        Keyword arguments for velocity field initialization.
    """

    def __init__(
        self,
        vf: MultiTaskConditionalVelocityField,
        probability_path: dynamics.BaseFlow,
        match_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
        time_sampler: Callable[[jax.Array, int], jnp.ndarray] = solver_utils.uniform_sampler,
        phenotype_loss_weight: float = 1.0,
        ema: float = 0.999,
        **kwargs: Any,
    ):
        self._is_trained: bool = False
        self.vf = vf
        self.condition_encoder_mode = self.vf.condition_mode
        self.condition_encoder_regularization = self.vf.regularization
        self.probability_path = probability_path
        self.match_fn = match_fn
        self.time_sampler = time_sampler
        self.phenotype_loss_weight = phenotype_loss_weight
        self.ema = ema

        self.vf_state = self.vf.create_train_state(**kwargs)
        self.vf_state_inference = self.vf_state
        self.vf_step_fn = self._get_vf_step_fn()

    def _get_vf_step_fn(self) -> Callable:
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
                v_t, mean_cond, logvar_cond, _ = vf_state.apply_fn(
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

    def _get_phenotype_step_fn(self) -> Callable:
        @jax.jit
        def phenotype_step_fn(
            rng: jax.Array,
            vf_state: train_state.TrainState,
            conditions: dict[str, jnp.ndarray],
            phenotype_targets: jnp.ndarray,
            encoder_noise: jnp.ndarray,
        ):
            def phenotype_loss_fn(
                params: jnp.ndarray,
                conditions: dict[str, jnp.ndarray],
                phenotype_targets: jnp.ndarray,
                encoder_noise: jnp.ndarray,
                rng: jax.Array,
            ) -> jnp.ndarray:
                rng_encoder, rng_dropout = jax.random.split(rng, 2)

                # Create dummy inputs for flow matching components
                n = phenotype_targets.shape[0]
                dummy_t = jnp.zeros(n)
                dummy_x = jnp.zeros((n, self.vf.output_dim))

                # Forward pass through multi-task velocity field
                _, mean_cond, logvar_cond, phenotype_pred = vf_state.apply_fn(
                    {"params": params},
                    dummy_t,
                    dummy_x,
                    conditions,
                    encoder_noise=encoder_noise,
                    rngs={"dropout": rng_dropout, "condition_encoder": rng_encoder},
                )

                # Phenotype prediction loss (MSE for regression)
                phenotype_loss = jnp.mean((phenotype_pred.squeeze() - phenotype_targets) ** 2)

                # Same condition regularization as flow matching
                condition_mean_regularization = 0.5 * jnp.mean(mean_cond**2)
                condition_var_regularization = -0.5 * jnp.mean(1 + logvar_cond - jnp.exp(logvar_cond))
                if self.condition_encoder_mode == "stochastic":
                    encoder_loss = condition_mean_regularization + condition_var_regularization
                elif (self.condition_encoder_mode == "deterministic") and (self.condition_encoder_regularization > 0):
                    encoder_loss = condition_mean_regularization
                else:
                    encoder_loss = 0.0

                return self.phenotype_loss_weight * phenotype_loss + encoder_loss

            grad_fn = jax.value_and_grad(phenotype_loss_fn)
            loss, grads = grad_fn(vf_state.params, conditions, phenotype_targets, encoder_noise, rng)
            return vf_state.apply_gradients(grads=grads), loss

        return phenotype_step_fn

    def step_fn(
        self,
        rng: jnp.ndarray,
        batch: dict[str, ArrayLike],
    ) -> float:
        """Single step function handling both flow matching and phenotype tasks.

        Parameters
        ----------
        rng
            Random number generator.
        batch
            Data batch. For flow matching: ``src_cell_data``, ``tgt_cell_data``, ``condition``.
            For phenotype: ``condition``, ``phenotype_target``, ``task``.

        Returns
        -------
        Loss value.
        """
        task = batch.get("task", "flow_matching")

        if task == "phenotype":
            return self._phenotype_step(rng, batch)
        else:
            return self._flow_matching_step(rng, batch)

    def _flow_matching_step(self, rng: jnp.ndarray, batch: dict[str, ArrayLike]) -> float:
        """Handle flow matching step."""
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
        return loss

    def _phenotype_step(self, rng: jnp.ndarray, batch: dict[str, ArrayLike]) -> float:
        """Handle phenotype prediction step."""
        condition = batch["condition"]
        phenotype_target = batch["phenotype_target"]
        rng_step_fn, rng_encoder_noise = jax.random.split(rng, 2)
        n = phenotype_target.shape[0]
        encoder_noise = jax.random.normal(rng_encoder_noise, (n, self.vf.condition_embedding_dim))

        phenotype_step_fn = self._get_phenotype_step_fn()
        self.vf_state, loss = phenotype_step_fn(
            rng_step_fn,
            self.vf_state,
            condition,
            phenotype_target,
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
        """Get learnt embeddings of the conditions."""
        cond_mean, cond_logvar = self.vf.apply(
            {"params": self.vf_state_inference.params},
            condition,
            method="get_condition_embedding",
        )
        if return_as_numpy:
            return np.asarray(cond_mean), np.asarray(cond_logvar)
        return cond_mean, cond_logvar

    def predict(
        self,
        x: ArrayLike | dict[str, ArrayLike],
        condition: dict[str, ArrayLike] | dict[str, dict[str, ArrayLike]],
        rng: jax.Array | None = None,
        batched: bool = False,
        task: str = "flow_matching",
        show_progress: bool = False,
        **kwargs: Any,
    ) -> ArrayLike | dict[str, ArrayLike]:
        """Predict either flow matching or phenotype outcomes.

        Parameters
        ----------
        x
            Input data (ignored for phenotype prediction).
        condition
            Condition dictionary.
        rng
            Random number generator.
        batched
            Whether to use batched prediction.
        task
            Either "flow_matching" or "phenotype".
        show_progress
            Whether to show a progress bar when predicting over multiple conditions.
        kwargs
            Additional arguments for ODE solver.

        Returns
        -------
        Predictions based on the specified task.
        """
        if task == "phenotype":
            return self._predict_phenotype(condition, rng)
        else:
            return self._predict_flow_matching(x, condition, rng, batched, show_progress, **kwargs)

    def _predict_phenotype(self, condition: dict[str, ArrayLike], rng: jax.Array | None = None) -> ArrayLike:
        """Predict phenotype values."""
        use_mean = rng is None or self.condition_encoder_mode == "deterministic"
        rng = utils.default_prng_key(rng)

        # Get condition shape
        first_cond = next(iter(condition.values()))
        n_samples = first_cond.shape[0]

        encoder_noise = (
            jnp.zeros((n_samples, self.vf.condition_embedding_dim))
            if use_mean
            else jax.random.normal(rng, (n_samples, self.vf.condition_embedding_dim))
        )

        phenotype_pred = self.vf_state_inference.apply_fn(
            {"params": self.vf_state_inference.params},
            method="predict_phenotype",
            cond=condition,
            encoder_noise=encoder_noise,
            train=False,
        )
        return np.array(phenotype_pred)

    def _predict_flow_matching(
        self,
        x: ArrayLike | dict[str, ArrayLike],
        condition: dict[str, ArrayLike] | dict[str, dict[str, ArrayLike]],
        rng: jax.Array | None = None,
        batched: bool = False,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> ArrayLike | dict[str, ArrayLike]:
        """Predict flow matching outcomes (same as original OTFM)."""
        if batched and not x:
            return {}

        if batched:
            keys = sorted(x.keys())
            condition_keys = sorted(set().union(*(condition[k].keys() for k in keys)))
            _predict_jit = jax.jit(lambda x, condition: self._predict_jit(x, condition, rng, **kwargs))
            batched_predict = jax.vmap(_predict_jit, in_axes=(0, dict.fromkeys(condition_keys, 0)))
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

                results = {}
                keys = sorted(x.keys())
                for key in tqdm(keys, desc="Predicting conditions", leave=False):
                    results[key] = self._predict_jit(x[key], condition[key], rng, **kwargs)
                return results
            else:
                return jax.tree.map(
                    lambda xi, ci: self._predict_jit(xi, ci, rng, **kwargs),
                    x,
                    condition,
                )
        else:
            x_pred = self._predict_jit(x, condition, rng, **kwargs)
            return np.array(x_pred)

    def _predict_jit(
        self, x: ArrayLike, condition: dict[str, ArrayLike], rng: jax.Array | None = None, **kwargs: Any
    ) -> ArrayLike:
        """JIT-compiled prediction for flow matching."""
        kwargs.setdefault("dt0", None)
        kwargs.setdefault("solver", diffrax.Tsit5())
        kwargs.setdefault("stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5))

        noise_dim = (1, self.vf.condition_embedding_dim)
        use_mean = rng is None or self.condition_encoder_mode == "deterministic"
        rng = utils.default_prng_key(rng)
        encoder_noise = jnp.zeros(noise_dim) if use_mean else jax.random.normal(rng, noise_dim)

        def vf(t: jnp.ndarray, x: jnp.ndarray, args: tuple[dict[str, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
            params = self.vf_state_inference.params
            condition, encoder_noise = args
            # Only use flow matching output (first element)
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

    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained."""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        """Set the trained status."""
        self._is_trained = value
