import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from ott.solvers import utils as solver_utils

import cellflow
from cellflow._compat import ConstantNoiseFlow
from cellflow.solvers import _otfm
from cellflow.solvers._otfm import AuxiliaryTask
from cellflow.utils import match_linear

vf_rng = jax.random.PRNGKey(111)


def _build_solver(**solver_kwargs):
    """Build an ``OTFlowMatching`` solver mirroring the existing solver tests."""
    opt = optax.adam(1e-3)
    vf = cellflow.networks.ConditionalVelocityField(
        output_dim=5,
        max_combination_length=2,
        condition_embedding_dim=12,
        hidden_dims=(32, 32),
        decoder_dims=(32, 32),
    )
    return _otfm.OTFlowMatching(
        vf=vf,
        match_fn=match_linear,
        probability_path=ConstantNoiseFlow(0.0),
        optimizer=opt,
        conditions={"drug": np.random.rand(2, 1, 3)},
        rng=vf_rng,
        **solver_kwargs,
    )


def _gex_batch():
    return {
        "src_cell_data": jnp.ones((10, 5)) * 10.0,
        "tgt_cell_data": jnp.ones((10, 5)),
        "condition": {"pert1": jnp.ones((1, 2, 3))},
    }


def _joint_batch():
    """A gex batch that additionally carries the phenotype task's data."""
    batch = _gex_batch()
    batch["phenotype"] = jnp.ones((1, 1)) * 3.0
    return batch


def _tree_all_equal(a, b):
    return jax.tree.all(jax.tree.map(lambda x, y: np.array_equal(np.asarray(x), np.asarray(y)), a, b))


class PhenotypeTask:
    """Toy auxiliary task: predict a scalar phenotype from the condition embedding.

    The loss depends on the shared velocity-field parameters (through
    ``get_condition_embedding``), so the task co-trains the trunk. This lets the
    tests check that loss weights shift the shared gradient rather than only the
    reported number. The task is active on any batch carrying a ``phenotype`` key.
    """

    def __init__(self, name="phenotype", loss_weight=1.0, embed_dim=12):
        self.name = name
        self.loss_weight = loss_weight
        self.embed_dim = embed_dim
        self._vf = None

    def init_state(self, vf, *, rng, optimizer):
        self._vf = vf
        params = {"w": jax.random.normal(rng, (self.embed_dim, 1)), "b": jnp.zeros((1,))}
        return train_state.TrainState.create(
            apply_fn=lambda p, x: x @ p["w"] + p["b"],
            params=params,
            tx=optimizer,
        )

    def is_active(self, batch):
        return "phenotype" in batch

    def loss(self, vf_params, aux_params, batch, rng):
        mean, _ = self._vf.apply(
            {"params": vf_params},
            batch["condition"],
            method="get_condition_embedding",
        )
        pred = mean @ aux_params["w"] + aux_params["b"]
        return jnp.mean((pred - batch["phenotype"]) ** 2)


class EmbeddingNormTask:
    """Stateless auxiliary task: L2-penalize the condition embedding.

    It owns no parameters (``init_state`` returns ``None``) but its loss reads
    ``vf_params``, so it trains the shared trunk only. Active on batches with a
    ``regularize`` flag.
    """

    name = "emb_norm"

    def __init__(self, loss_weight=1.0):
        self.loss_weight = loss_weight
        self._vf = None

    def init_state(self, vf, *, rng, optimizer):
        self._vf = vf
        return None

    def is_active(self, batch):
        return "regularize" in batch

    def loss(self, vf_params, aux_params, batch, rng):
        mean, _ = self._vf.apply({"params": vf_params}, batch["condition"], method="get_condition_embedding")
        return jnp.mean(mean**2)


class TestAuxiliaryTasks:
    def test_protocol_surface(self):
        task = PhenotypeTask()
        assert isinstance(task.name, str)
        assert hasattr(task, "loss_weight")
        assert callable(task.init_state)
        assert callable(task.is_active)
        assert callable(task.loss)
        assert isinstance(task, AuxiliaryTask)  # runtime_checkable structural check
        # a solver built without tasks holds no auxiliary state
        solver = _build_solver()
        assert solver.auxiliary_tasks == ()
        assert solver._aux_states == {}

    def test_gex_only_step_is_correct(self):
        """A gex-only batch reports the weighted flow-matching loss and trains the field.

        gex is dispatched uniformly through the joint path (no special fast path),
        so we replicate that path's rng derivation and objective and check the
        reported loss matches.
        """
        solver = _build_solver()
        assert solver.loss_weight_gex == 1.0
        batch = _gex_batch()
        rng = jax.random.PRNGKey(0)
        init_params = solver.vf_state.params

        # Replicate the joint path's objective for the single built-in gex task:
        # one active task gets rngs = split(rng, 1); the task splits that into 4.
        rng_gex = jax.random.split(rng, 1)[0]
        rng_resample, rng_time, rng_flow, rng_encoder_noise = jax.random.split(rng_gex, 4)
        src, tgt = batch["src_cell_data"], batch["tgt_cell_data"]
        n = src.shape[0]
        time = solver.time_sampler(rng_time, n)
        encoder_noise = jax.random.normal(rng_encoder_noise, (n, solver.vf.condition_embedding_dim))
        tmat = solver.match_fn(src, tgt)
        src_ixs, tgt_ixs = solver_utils.sample_joint(rng_resample, tmat)
        src_m, tgt_m = src[src_ixs], tgt[tgt_ixs]
        expected_loss = solver.loss_weight_gex * solver._flow_matching_loss(
            init_params, time, src_m, tgt_m, batch.get("condition"), encoder_noise, rng_flow
        )

        loss = solver.step_fn(rng, batch)

        assert np.allclose(np.asarray(loss), np.asarray(expected_loss))  # correct objective value
        assert not _tree_all_equal(solver.vf_state.params, init_params)  # field trained
        # ema defaults to 1.0, so the inference state tracks the training state
        assert _tree_all_equal(solver.vf_state_inference.params, solver.vf_state.params)

    def test_loss_weight_gex_scales_loss_and_affects_gex_update(self):
        """loss_weight_gex scales the reported gex loss and enters the gradient.

        Because gex is dispatched through the same joint objective as any task, its
        weight multiplies the differentiated loss, so it also changes the gex-only
        update (a single Adam step is not exactly scale-invariant — for small
        gradients the step is roughly linear in the gradient).
        """
        batch = _gex_batch()
        rng = jax.random.PRNGKey(0)

        solver1 = _build_solver(loss_weight_gex=1.0)
        solver3 = _build_solver(loss_weight_gex=3.0)
        assert _tree_all_equal(solver1.vf_state.params, solver3.vf_state.params)

        loss1 = solver1.step_fn(rng, batch)
        loss3 = solver3.step_fn(rng, batch)

        assert np.allclose(np.asarray(loss3), 3.0 * np.asarray(loss1))  # reported loss scales
        assert not _tree_all_equal(solver1.vf_state.params, solver3.vf_state.params)  # weight enters the gradient

    def test_task_registration_and_init(self):
        task = PhenotypeTask(name="phenotype", loss_weight=2.0)
        solver = _build_solver(auxiliary_tasks=[task])
        assert solver.auxiliary_tasks == (task,)
        assert isinstance(solver._aux_states["phenotype"], train_state.TrainState)

    def test_is_active_gates_participation(self):
        """A batch without the task's data leaves the task out of the step."""
        task = PhenotypeTask()
        solver = _build_solver(auxiliary_tasks=[task])
        aux_before = solver._aux_states["phenotype"].params
        vf_before = solver.vf_state.params

        # pure gex batch: no "phenotype" key -> task is inactive
        solver.step_fn(jax.random.PRNGKey(0), _gex_batch())

        assert _tree_all_equal(solver._aux_states["phenotype"].params, aux_before)  # head untouched
        assert not _tree_all_equal(solver.vf_state.params, vf_before)  # flow still trained

    def test_joint_step_updates_trunk_and_head(self):
        """An active task co-updates its head and the shared trunk in one step."""
        task = PhenotypeTask(loss_weight=2.0)
        solver = _build_solver(auxiliary_tasks=[task])
        aux_before = solver._aux_states["phenotype"].params
        vf_before = solver.vf_state.params

        loss = solver.step_fn(jax.random.PRNGKey(0), _joint_batch())

        assert np.isfinite(np.asarray(loss))
        assert not _tree_all_equal(solver._aux_states["phenotype"].params, aux_before)  # head updated
        assert not _tree_all_equal(solver.vf_state.params, vf_before)  # trunk co-trained
        assert _tree_all_equal(solver.vf_state_inference.params, solver.vf_state.params)

    def test_loss_weight_gex_shifts_joint_gradient(self):
        """In a joint step, loss_weight_gex changes the shared-trunk update direction."""
        s1 = _build_solver(loss_weight_gex=1.0, auxiliary_tasks=[PhenotypeTask(loss_weight=1.0)])
        s2 = _build_solver(loss_weight_gex=5.0, auxiliary_tasks=[PhenotypeTask(loss_weight=1.0)])
        assert _tree_all_equal(s1.vf_state.params, s2.vf_state.params)  # identical init
        assert _tree_all_equal(s1._aux_states["phenotype"].params, s2._aux_states["phenotype"].params)

        batch = _joint_batch()
        s1.step_fn(jax.random.PRNGKey(0), batch)
        s2.step_fn(jax.random.PRNGKey(0), batch)

        # different gex weight -> different combined gradient -> different trunk update
        assert not _tree_all_equal(s1.vf_state.params, s2.vf_state.params)
        # the head gradient does not depend on loss_weight_gex, so heads stay in sync
        assert _tree_all_equal(s1._aux_states["phenotype"].params, s2._aux_states["phenotype"].params)

    def test_aux_only_reported_loss_equals_weighted_sum(self):
        """On an aux-only batch the reported loss is loss_weight * the task loss."""
        task = PhenotypeTask(loss_weight=2.0)
        solver = _build_solver(auxiliary_tasks=[task])

        condition = {"pert1": jnp.ones((1, 2, 3))}
        y = jnp.ones((1, 1)) * 3.0
        batch = {"condition": condition, "phenotype": y}  # no gex data

        # expected loss evaluated at the pre-step parameters (get_condition_embedding
        # is deterministic, so no rng is involved)
        mean, _ = solver.vf.apply({"params": solver.vf_state.params}, condition, method="get_condition_embedding")
        p = solver._aux_states["phenotype"].params
        expected = task.loss_weight * jnp.mean((mean @ p["w"] + p["b"] - y) ** 2)

        loss = solver.step_fn(jax.random.PRNGKey(0), batch)

        assert np.allclose(np.asarray(loss), np.asarray(expected))
        # the head is updated; gex is inactive so its rng path is never touched
        assert not _tree_all_equal(solver._aux_states["phenotype"].params, p)

    def test_gex_is_a_task(self):
        """The default flow-matching objective is itself a task carrying its weight."""
        solver = _build_solver(loss_weight_gex=2.5)
        gex = solver._gex_task
        assert isinstance(gex, AuxiliaryTask)  # structural conformance to the protocol
        assert gex.name == "gex"
        assert gex.loss_weight == 2.5
        # it leads the unified task list and is stateless (owns no state)
        assert solver._tasks[0] is gex
        assert "gex" not in solver._aux_states
        assert gex.is_active(_gex_batch())
        assert not gex.is_active({"phenotype": jnp.zeros((1, 1))})

    def test_stateless_task_cotrains_trunk_without_state(self):
        """A stateless task (init_state -> None) trains the trunk but holds no state."""
        task = EmbeddingNormTask(loss_weight=1.0)
        solver = _build_solver(auxiliary_tasks=[task])
        # registered as a task, but no entry in the state registry
        assert task in solver._tasks
        assert "emb_norm" not in solver._aux_states

        vf_before = solver.vf_state.params
        batch = {"condition": {"pert1": jnp.ones((1, 2, 3))}, "regularize": jnp.ones(())}
        loss = solver.step_fn(jax.random.PRNGKey(0), batch)

        assert np.isfinite(np.asarray(loss))
        assert not _tree_all_equal(solver.vf_state.params, vf_before)  # trunk trained by the stateless task


def test_auxiliary_task_is_exported():
    # the protocol is importable from the public solvers package
    assert cellflow.solvers.AuxiliaryTask is AuxiliaryTask
