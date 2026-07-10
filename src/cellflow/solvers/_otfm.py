import warnings
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import frozen_dict
from flax.training import train_state
from ott.solvers import utils as solver_utils

from cellflow import utils
from cellflow._compat import BaseFlow
from cellflow._types import ArrayLike
from cellflow.networks._velocity_field import ConditionalVelocityField
from cellflow.solvers.utils import ema_update

__all__ = ["AuxiliaryTask", "ClassifierFreeGuidance", "Guidance", "OTFlowMatching"]

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


@runtime_checkable
class AuxiliaryTask(Protocol):
    """Protocol for an auxiliary task trained jointly with the flow.

    An auxiliary task (e.g. a phenotype predictor) owns its own
    :class:`~flax.training.train_state.TrainState` — typically the parameters of
    an extra prediction head — and contributes a differentiable loss term that is
    optimized *together* with the flow-matching objective in a single step. Tasks
    are registered with :class:`OTFlowMatching` through the ``auxiliary_tasks``
    constructor argument (which may be supplied via
    :meth:`cellflow.model.CellFlow.prepare_model`'s ``solver_kwargs``), so extra
    heads can be trained without forking the solver.

    A task decides for itself whether it participates in a given batch through
    :meth:`is_active`, which inspects the batch for the data it needs — the solver
    never couples the data loader to a fixed task schedule. On each step the
    weighted losses of all active tasks, together with the flow-matching term, are
    summed into a single objective and differentiated jointly. As a result
    :attr:`loss_weight` genuinely shifts the shared-trunk gradient direction; a
    per-task step, by contrast, is scale-invariant under Adam and would leave
    training essentially unchanged.
    """

    name: str
    loss_weight: float

    def init_state(
        self,
        vf: ConditionalVelocityField,
        *,
        rng: jax.Array,
        optimizer: optax.GradientTransformation,
    ) -> train_state.TrainState | None:
        """Create the initial training state for the auxiliary task.

        Parameters
        ----------
        vf
            The conditional velocity field shared with the flow-matching task.
        rng
            Random number generator used to initialize the task parameters.
        optimizer
            Optimizer for the task parameters.

        Returns
        -------
        The initial training state of the auxiliary task, or :obj:`None` for a
        stateless task that owns no parameters of its own and only trains the
        shared velocity field.
        """
        ...

    def is_active(self, batch: dict[str, ArrayLike]) -> bool:
        """Return whether ``batch`` carries the data this task needs.

        Called on every batch; the task contributes its loss term only when this
        returns :obj:`True`, so a batch that lacks the task's data simply skips it.
        This keeps the data loader decoupled from the set of tasks.
        """
        ...

    def loss(
        self,
        vf_params: Any,
        aux_params: Any,
        batch: dict[str, ArrayLike],
        rng: jax.Array,
    ) -> jnp.ndarray:
        """Differentiable scalar loss contributed by this task.

        Parameters
        ----------
        vf_params
            Parameters of the shared velocity field. Use them (e.g. through
            ``vf.apply``) to co-train the shared trunk, or ignore them to train
            only the task's own head. Only the parameters your loss actually
            reads receive a gradient and are updated; wrap a read in
            :func:`jax.lax.stop_gradient` to use the shared representation
            without training the trunk.
        aux_params
            Parameters of this task's own state (``init_state(...).params``).
        batch
            The current data batch.
        rng
            Random number generator for this task.

        Returns
        -------
        The unweighted scalar loss. The solver multiplies it by
        :attr:`loss_weight` before summing it into the joint objective and takes a
        single optimization step over the velocity field and every active head.
        """
        ...


class _FlowMatchingTask:
    """Built-in :class:`AuxiliaryTask` wrapping the flow-matching objective.

    Makes the default gene-expression objective a first-class task, so
    :meth:`OTFlowMatching.step_fn` treats it uniformly with user-supplied tasks
    and its weight lives on the task (set via ``loss_weight_gex``). It is
    stateless — it trains the shared velocity field rather than a head of its own,
    so :meth:`init_state` returns :obj:`None` — and it participates whenever the
    batch carries ``src_cell_data``.
    """

    name = "gex"

    def __init__(self, solver: "OTFlowMatching", loss_weight: float):
        self._solver = solver
        self.loss_weight = loss_weight

    def init_state(self, vf: ConditionalVelocityField, *, rng: jax.Array, optimizer: Any) -> None:
        """Return :obj:`None`; the flow-matching task trains the shared field only."""
        return None

    def is_active(self, batch: dict[str, ArrayLike]) -> bool:
        """Active whenever the batch carries flow-matching (gene-expression) data."""
        return "src_cell_data" in batch

    def loss(self, vf_params: Any, aux_params: Any, batch: dict[str, ArrayLike], rng: jax.Array) -> jnp.ndarray:
        """Flow-matching loss: sample time/noise, OT-match the batch, then score."""
        solver = self._solver
        src, tgt = batch["src_cell_data"], batch["tgt_cell_data"]
        condition = batch.get("condition")
        rng_resample, rng_time, rng_flow, rng_encoder_noise = jax.random.split(rng, 4)
        n = src.shape[0]
        time = solver.time_sampler(rng_time, n)
        encoder_noise = jax.random.normal(rng_encoder_noise, (n, solver.vf.condition_embedding_dim))
        # TODO: test whether it's better to sample the same noise for all samples or different ones
        if solver.match_fn is not None:
            tmat = solver.match_fn(src, tgt)
            src_ixs, tgt_ixs = solver_utils.sample_joint(rng_resample, tmat)
            src, tgt = src[src_ixs], tgt[tgt_ixs]
        return solver._flow_matching_loss(vf_params, time, src, tgt, condition, encoder_noise, rng_flow)


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
        loss_weight_gex
            Weight of the flow-matching (gene-expression) loss term in the joint
            objective. The default objective is itself a built-in, stateless task
            (dispatched like any :class:`AuxiliaryTask`); this sets its weight.
        auxiliary_tasks
            Auxiliary tasks (see :class:`AuxiliaryTask`) trained jointly with the
            flow. Each task owns its own training state, self-reports participation
            per batch via :meth:`AuxiliaryTask.is_active`, and contributes a
            weighted loss term that :meth:`step_fn` optimizes together with the
            flow-matching loss in a single step.
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
        loss_weight_gex: float = 1.0,
        auxiliary_tasks: Sequence[AuxiliaryTask] = (),
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
        self._predict_fn_cache: dict[frozen_dict.FrozenDict, Any] = {}

        # Auxiliary tasks contribute weighted loss terms that step_fn optimizes
        # jointly with the flow-matching objective. Each task owns its own training
        # state (stateless tasks return None) and self-reports participation per
        # batch via ``is_active``.
        self.auxiliary_tasks: tuple[AuxiliaryTask, ...] = tuple(auxiliary_tasks)
        self._aux_states: dict[str, train_state.TrainState] = {}
        for task in self.auxiliary_tasks:
            state = task.init_state(self.vf, rng=kwargs.get("rng"), optimizer=kwargs.get("optimizer"))
            if state is not None:
                self._aux_states[task.name] = state
        # The default flow-matching objective is itself an (internal, stateless)
        # task, so step_fn iterates all tasks uniformly. Its weight lives on the
        # task and is configured by ``loss_weight_gex``.
        self._gex_task = _FlowMatchingTask(self, loss_weight_gex)
        self._tasks: tuple[AuxiliaryTask, ...] = (self._gex_task, *self.auxiliary_tasks)
        # Jitted joint-update fns, cached per active-task signature.
        self._joint_step_cache: dict[tuple[str, ...], Callable] = {}  # type: ignore[type-arg]

    @property
    def loss_weight_gex(self) -> float:
        """Weight of the built-in flow-matching task (its single source of truth)."""
        return self._gex_task.loss_weight

    @loss_weight_gex.setter
    def loss_weight_gex(self, value: float) -> None:
        self._gex_task.loss_weight = value

    def _flow_matching_loss(
        self,
        params: Any,
        t: jnp.ndarray,
        source: jnp.ndarray,
        target: jnp.ndarray,
        conditions: dict[str, jnp.ndarray] | None,
        encoder_noise: jnp.ndarray,
        rng: jax.Array,
    ) -> jnp.ndarray:
        """Flow-matching (gene-expression) loss of the built-in gex task.

        Kept as a solver method so the core objective lives in one place; it is
        differentiated inside the joint step via :meth:`_FlowMatchingTask.loss`.
        """
        rng_flow, rng_encoder, rng_dropout = jax.random.split(rng, 3)
        x_t = self.probability_path.compute_xt(rng_flow, t, source, target)
        v_t, mean_cond, logvar_cond = self.vf.apply(
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

    def step_fn(
        self,
        rng: jnp.ndarray,
        batch: dict[str, ArrayLike],
    ) -> float:
        """Single training step of the solver.

        The flow-matching (gene-expression) objective is itself a task, so every
        task — the built-in one and each :class:`AuxiliaryTask` — is treated
        uniformly: each inspects the batch via :meth:`AuxiliaryTask.is_active`,
        and the active tasks' weighted losses are summed into a single objective
        and optimized jointly in one step over the velocity field and every
        active head, so the loss weights genuinely shift the shared gradient.

        Parameters
        ----------
        rng
            Random number generator.
        batch
            Data batch. The flow-matching term uses ``src_cell_data``,
            ``tgt_cell_data`` and optionally ``condition``; auxiliary tasks read
            whatever additional keys they declare through ``is_active``.

        Returns
        -------
        The (weighted) loss value of the step.

        Notes
        -----
        A single step updates:

        - ``vf_state`` — the shared velocity field — from the *combined* gradient
          of every active task. A task feeds the trunk only through the
          ``vf_params`` its :meth:`AuxiliaryTask.loss` reads: the flow-matching
          term touches the whole field, whereas a task that only consumes the
          condition embedding updates just the condition encoder and leaves the
          flow/decoder weights untouched.
        - ``_aux_states[name]`` — each active task's own state (e.g. a prediction
          head) — from that task's own weighted loss only; one task's parameters
          never enter another task's loss term.
        - ``vf_state_inference`` — resynced from ``vf_state`` after the step (an
          exact copy when ``ema == 1.0``, otherwise an EMA update).
        """
        active = tuple(task for task in self._tasks if task.is_active(batch))
        return self._step_joint(rng, batch, active)

    def _update_inference_state(self) -> None:
        """Synchronize the inference velocity-field state with the training state."""
        if self.ema == 1.0:
            self.vf_state_inference = self.vf_state
        else:
            self.vf_state_inference = self.vf_state_inference.replace(
                params=ema_update(self.vf_state_inference.params, self.vf_state.params, self.ema)
            )

    def _step_joint(self, rng: jnp.ndarray, batch: dict[str, ArrayLike], active: tuple[AuxiliaryTask, ...]) -> float:
        """Joint step over every active task (the flow-matching task included).

        Sums each active task's weighted loss term into one objective,
        differentiates it once with respect to the velocity field and all active
        stateful heads, and applies the resulting gradients in a single
        optimization step. Stateless tasks (e.g. the flow-matching task) still
        feed the shared-field gradient but advance no state of their own.
        """
        active_names = tuple(task.name for task in active)
        rngs = jax.random.split(rng, len(active_names))
        task_rngs = {name: rngs[i] for i, name in enumerate(active_names)}

        joint_step = self._get_joint_step_fn(active_names)
        aux_states = {name: self._aux_states[name] for name in active_names if name in self._aux_states}
        self.vf_state, new_aux_states, loss = joint_step(self.vf_state, aux_states, batch, task_rngs)
        self._aux_states.update(new_aux_states)
        self._update_inference_state()
        return loss

    def _get_joint_step_fn(self, active_names: tuple[str, ...]) -> Callable:  # type: ignore[type-arg]
        """Build (and cache) the jitted joint update for a given active-task signature."""
        if active_names in self._joint_step_cache:
            return self._joint_step_cache[active_names]

        tasks_by_name = {task.name: task for task in self._tasks}
        active_tasks = tuple(tasks_by_name[name] for name in active_names)
        stateful_names = tuple(name for name in active_names if name in self._aux_states)

        @jax.jit
        def joint_step(vf_state, aux_states, batch, task_rngs):
            def total_loss(vf_params, aux_params):
                total = jnp.zeros(())
                for task in active_tasks:
                    total = total + task.loss_weight * task.loss(
                        vf_params, aux_params.get(task.name), batch, task_rngs[task.name]
                    )
                return total

            aux_params = {name: aux_states[name].params for name in stateful_names}
            loss, (grad_vf, grad_aux) = jax.value_and_grad(total_loss, argnums=(0, 1))(vf_state.params, aux_params)
            new_vf_state = vf_state.apply_gradients(grads=grad_vf)
            new_aux_states = {name: aux_states[name].apply_gradients(grads=grad_aux[name]) for name in stateful_names}
            return new_vf_state, new_aux_states, loss

        self._joint_step_cache[active_names] = joint_step
        return joint_step

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
