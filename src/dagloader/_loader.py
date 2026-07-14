"""``DAGLoader`` — streams matched ``{source, target, condition}`` batches from a :class:`Scheme`.

Each pass is a fresh epoch. The root (target) node draws a per-batch class schedule ∝ its weights via an
annbatch :class:`~annbatch.samplers.ClassSampler`; every bound child replays that schedule onto its own
cells via an annbatch :class:`~annbatch.samplers.BoundClassSampler` — matched by *label* on the bind's
``common`` columns (a ``matched=`` bind instead remaps the schedule to explicit child leaves). The loader
never wraps annbatch's RNG: every sampler that must agree within a pass (the schedule oracle, the target
reps, and each child's inner) is **reseeded from one per-pass seed** ``(node seed, pass index)``, so
target/condition/source stay aligned, a node's reps read the same rows, and a pickled loader resumes the
exact same stream. See ``README.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from importlib.util import find_spec

import numpy as np
import pandas as pd
from annbatch import Loader
from annbatch.samplers import BoundClassSampler, ClassSampler

from dagloader._io import key_backings, leaf_codes, obs_columns
from dagloader._schema import Bind, SamplerConfig, Scheme, _weight_vector

__all__ = ["DAGLoader"]

# annbatch's GPU path (cupy vstack/indexing → `to="jax"` yields a GPU-resident array via dlpack) needs
# cupy. When it's absent (CPU-only envs — Mac, CI), fall back so `to="jax"` still yields a CPU jax array.
_HAS_CUPY = find_spec("cupy") is not None


def _flat_categorical(codes: np.ndarray, leaves: list[tuple]) -> pd.Categorical:
    """A tuple-labelled categorical: per-cell leaf code over ``leaves`` (categories are the leaf tuples).

    Tuple labels (not opaque integer codes) are what let :class:`~annbatch.samplers.BoundClassSampler`
    match a child to its parent by the bind's ``common`` columns — it projects the label by position.
    """
    categories = pd.MultiIndex.from_tuples(leaves).to_flat_index()
    return pd.Categorical.from_codes(codes, categories=categories)


class _RemappedScheduleInner:
    """Schedule-only :class:`~annbatch.abc.BaseClassSampler` that remaps a root schedule to child leaves.

    For ``matched=`` binds (explicit parent-leaf → child-leaf pairing, not a shared-column match): it
    wraps the root schedule oracle and exposes, as its own ``batch_codes``, the child leaf of each batch
    (via the pairing). A :class:`~annbatch.samplers.BoundClassSampler` then reads matched child cells by
    label (``on=None``). Only the schedule surface (``vocab`` / ``emittable_codes`` / ``batch_codes`` /
    ``n_batches``) is used by the bound — never :meth:`_sample`.
    """

    def __init__(
        self,
        oracle: ClassSampler,
        root_leaves: list[tuple],
        child_leaves: list[tuple],
        matched: Mapping[tuple, tuple],
    ) -> None:
        self._oracle = oracle
        self._root_leaves = root_leaves
        self._child_vocab = pd.MultiIndex.from_tuples(child_leaves).to_flat_index()
        child_pos = {lf: i for i, lf in enumerate(child_leaves)}
        self._root_to_child: dict[int, int] = {}
        for ri, rlf in enumerate(root_leaves):
            paired = matched.get(tuple(rlf))
            if paired is not None:
                if tuple(paired) not in child_pos:
                    raise ValueError(f"matched child leaf {paired!r} is not a child leaf.")
                self._root_to_child[ri] = child_pos[tuple(paired)]

    @property
    def rng(self) -> np.random.Generator:
        return self._oracle.rng

    @rng.setter
    def rng(self, value: np.random.Generator) -> None:
        self._oracle.rng = value

    @property
    def vocab(self) -> pd.Index:
        return self._child_vocab

    def emittable_codes(self) -> np.ndarray:
        return np.array(sorted(set(self._root_to_child.values())), dtype=np.int64)

    def n_batches(self, n_obs: int = 0) -> int:
        return self._oracle.n_batches(n_obs)

    def batch_codes(self) -> np.ndarray:
        out = np.empty(self._oracle.n_batches(0), dtype=np.int64)
        for j, rc in enumerate(self._oracle.batch_codes()):
            rc = int(rc)
            if rc not in self._root_to_child:
                raise ValueError(f"matched bind has no child leaf for root leaf {self._root_leaves[rc]!r}.")
            out[j] = self._root_to_child[rc]
        return out


class DAGLoader:
    """Yields ``{"source", "target", "condition"}`` batches; every node streams through its own loader."""

    def __init__(
        self,
        scheme: Scheme,
        sampler_config: SamplerConfig | Mapping[str, SamplerConfig],
        condition_fn: Callable[[tuple], np.ndarray | Mapping[str, np.ndarray]] | None = None,
    ) -> None:
        self.s = scheme
        self._cond_fn = condition_fn
        self._cfg = self._resolve_configs(sampler_config)
        self._B = self._cfg[self.s.root].batch_size  # root/target batch size — drives the pass length

        # root's direct children are the bound sources (parity with the previous loader: depth-1 sources)
        self._child_binds: list[Bind] = [b for b in scheme.binds if b.parent == self.s.root]

        # per-node stable sub-seed from one SeedSequence, so nodes don't correlate; a pass's seed is
        # (sub-seed, pass index) → a pass is fully reproducible from its index.
        self._node_seeds: dict[str, int] = {
            name: int(seq.generate_state(1)[0])
            for name, seq in zip(
                sorted(scheme.nodes), np.random.SeedSequence(scheme.seed).spawn(len(scheme.nodes)), strict=True
            )
        }

        # per-node leaf partition + weights + tuple-labelled categorical (obs only — no cell matrices)
        self._st: dict[str, dict] = {}
        for name, node in scheme.nodes.items():
            obs = obs_columns(scheme.sources[node.source], node.cols)
            codes, leaves = leaf_codes(obs, node.cols)
            self._st[name] = {
                "node": node,
                "leaves": leaves,
                "w": _weight_vector(node.weights, leaves),
                "cats": _flat_categorical(codes, leaves),
            }

        # A natural epoch over the root (target) node: its cell count // the root's batch_size. The root
        # drives the zip, so every node draws the same number of batches; each node's num_samples ==
        # _n_batches * that node's batch_size (source rows need not equal target rows).
        n_root_obs = len(self._st[self.s.root]["cats"])
        self._n_batches = max(1, n_root_obs // self._B)

        self._build_samplers_and_loaders()

        self._iters: dict[str, dict[str, Iterator[dict]]] | None = None
        self._schedule: np.ndarray | None = None  # per-batch root leaf code for the current pass
        self._pos = 0

    def _resolve_configs(self, cfg: SamplerConfig | Mapping[str, SamplerConfig]) -> dict[str, SamplerConfig]:
        """Normalize to one config per node (nodes may use different batch sizes)."""
        if isinstance(cfg, SamplerConfig):
            return dict.fromkeys(self.s.nodes, cfg)
        missing = set(self.s.nodes) - set(cfg)
        if missing:
            raise ValueError(f"sampler_config mapping is missing node(s): {sorted(missing)}.")
        return {name: cfg[name] for name in self.s.nodes}

    # ── build ────────────────────────────────────────────────────────────
    def _new_class_sampler(self, name: str) -> ClassSampler:
        cfg = self._cfg[name]
        try:  # annbatch enforces its own run-length rule for chunk>1; forward with node context
            return ClassSampler(
                chunk_size=cfg.chunk_size,
                preload_nchunks=cfg.preload_nchunks,
                batch_size=cfg.batch_size,
                classes=self._st[name]["cats"],
                num_samples=self._n_batches * cfg.batch_size,
                class_weights=self._st[name]["w"],
                drop_last=True,
                rng=np.random.default_rng(self._node_seeds[name]),
            )
        except ValueError as e:
            raise ValueError(f"node {name!r}: {e}") from e

    def _bound_on(self, b: Bind) -> dict[int, int]:
        """Map root tuple positions → child tuple positions for the bind's ``common`` columns."""
        rcols, ccols = self._st[b.parent]["node"].cols, self._st[b.child]["node"].cols
        return {rcols.index(c): ccols.index(c) for c in b.common}

    def _new_bound_sampler(self, b: Bind) -> BoundClassSampler:
        if b.matched is not None:
            inner = _RemappedScheduleInner(
                self._new_class_sampler(self.s.root),
                self._st[self.s.root]["leaves"],
                self._st[b.child]["leaves"],
                b.matched,
            )
            return self._make_bound(b, inner, on=None)  # remapped inner emits exact child leaves
        inner = self._new_class_sampler(self.s.root)
        # Match on the bind's shared columns; the child's leaf weights (0 for excluded leaves, e.g.
        # perturbed cells in a control node) go in as the *secondary* class so only positive-weight
        # child leaves are drawn within each matched context — the exclusion `classes_to_bind_on` alone
        # can't express (it groups all cells sharing the context).
        return self._make_bound(
            b,
            inner,
            on=self._bound_on(b),
            classes=self._st[b.child]["cats"],
            class_weights=self._st[b.child]["w"],
        )

    def _make_bound(
        self,
        b: Bind,
        inner: ClassSampler | _RemappedScheduleInner,
        *,
        on: dict[int, int] | None,
        classes: pd.Categorical | None = None,
        class_weights: np.ndarray | None = None,
    ) -> BoundClassSampler:
        cfg = self._cfg[b.child]
        try:
            return BoundClassSampler(
                inner,  # type: ignore[arg-type]
                cfg.chunk_size,
                cfg.preload_nchunks,
                cfg.batch_size,
                classes_to_bind_on=self._st[b.child]["cats"],
                on=on,
                classes=classes,
                class_weights=class_weights,
                rng=np.random.default_rng(self._node_seeds[b.child]),
            )
        except ValueError as e:
            raise ValueError(f"node {b.child!r}: {e}") from e

    def _build_samplers_and_loaders(self) -> None:
        """Per node/key: a ClassSampler (root) or BoundClassSampler (child) + annbatch Loader.

        A per-child schedule *oracle* and each bound's inner are root-seeded so their class draws agree
        with the target's; all of a node's keys share the node seed, so the (identical) samplers select
        the same rows every batch — every rep of a node is the same cells. Reps need separate Loaders
        (annbatch can't mix feature dims in one loader), each with native chunked reads.
        """
        self._oracle = self._new_class_sampler(self.s.root)  # supplies the per-batch condition schedule

        self._samplers: dict[str, dict[str, ClassSampler | BoundClassSampler]] = {}
        self._loaders: dict[str, dict[str, Loader]] = {}
        self._add_node_loaders(self.s.root, lambda: self._new_class_sampler(self.s.root))
        for b in self._child_binds:
            self._add_node_loaders(b.child, lambda b=b: self._new_bound_sampler(b))

    def _add_node_loaders(self, name: str, make_sampler: Callable[[], ClassSampler | BoundClassSampler]) -> None:
        node = self._st[name]["node"]
        src = self.s.sources[node.source]
        self._samplers[name], self._loaders[name] = {}, {}
        for ki, key in enumerate(node.keys):
            sampler = make_sampler()
            return_index = name == self.s.root and ki == 0  # only for the schedule↔row alignment check
            # `to="jax"` yields native jax arrays (dense via `jnp.from_dlpack`, sparse as a jax CSR)
            # instead of numpy, so batches reach the jax solver without a host round-trip / numpy cast.
            # With cupy present, `preload_to_gpu` keeps the array on-GPU so the yielded jax array is
            # GPU-resident; without it, the array is a CPU jax array (device copy deferred to the step).
            loader = Loader(
                batch_sampler=sampler, return_index=return_index, to="jax", preload_to_gpu=_HAS_CUPY
            ).add_datasets(key_backings(src, key))
            self._samplers[name][key], self._loaders[name][key] = sampler, loader

    # ── per-pass scheduling ────────────────────────────────────────────────
    def _start_pass(self) -> None:
        """Draw the schedule and rebuild iterators for a fresh epoch (advancing every sampler's RNG once).

        The oracle's ``batch_codes()`` and each target/child ``iter()`` each consume one class draw, so —
        all root-referencing samplers having started from the root seed — the oracle, the target reps and
        every bound child's inner stay in lockstep and draw the *same* per-batch class each pass. The RNG
        advances across passes (a real epoch stream), so a pickled loader — whose sampler RNG state is
        kept — resumes the next pass rather than replaying.
        """
        self._schedule = self._oracle.batch_codes()
        self._iters = {name: {key: iter(ld) for key, ld in loaders.items()} for name, loaders in self._loaders.items()}
        self._pos = 0

    # ── pickling ─────────────────────────────────────────────────────────────
    def __getstate__(self) -> dict[str, object]:
        """Pickle without the live annbatch iterators (generators aren't picklable).

        Every sampler's RNG state is kept, so a reloaded loader resumes the same reproducible stream
        (the next pass) on the next ``__next__``; only ``_iters`` is dropped and rebuilt.
        """
        state = self.__dict__.copy()
        state["_iters"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._iters = None  # force `_start_pass` on the next `__next__`, using the restored sampler RNG

    # ── iteration ──────────────────────────────────────────────────────────
    def __iter__(self) -> DAGLoader:
        return self

    def _nodes_next(self, name: str) -> dict[str, np.ndarray]:
        """One batch per key of a node — identical samplers pick the same rows, so the reps are aligned."""
        node = self._st[name]["node"]
        return {key: next(self._iters[name][key])["X"] for key in node.keys}

    def __next__(self) -> dict[str, np.ndarray]:
        if self._iters is None or self._pos >= self._n_batches:
            self._start_pass()  # first pass, next epoch, or resume after unpickling
        j = self._pos
        self._pos += 1

        st = self._st[self.s.root]
        node = st["node"]
        leaf = st["leaves"][int(self._schedule[j])]  # per-batch category — from the schedule oracle
        reps = self._nodes_next(self.s.root)
        target = reps[node.keys[0]]  # primary streamed rep
        B = target.shape[0]

        out: dict = {"target": target}
        if len(node.keys) > 1:  # aligned reps of the target cells (state, per-cell condition, …)
            out["target_reps"] = reps
        if self._cond_fn is not None:
            cond = self._cond_fn(leaf)
            if isinstance(cond, Mapping):  # per-leaf structured condition (e.g. set-encoded) — emit as-is
                out["condition"] = {k: np.asarray(v, dtype=np.float32) for k, v in cond.items()}
            else:  # a per-leaf vector broadcast across the (class-coherent) batch
                cond = np.asarray(cond, dtype=np.float32)
                out["condition"] = np.broadcast_to(cond, (B, cond.shape[-1])).copy()

        for b in self._child_binds:  # bound child source, replayed via its BoundClassSampler
            cnode = self._st[b.child]["node"]
            creps = self._nodes_next(b.child)
            out["source"] = creps[cnode.keys[0]]
            if len(cnode.keys) > 1:
                out["source_reps"] = creps
        return out
