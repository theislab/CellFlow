"""``DAGLoader`` — streams matched ``{source, target, condition}`` batches from a :class:`Scheme`.

Each pass is a fresh epoch. The root (target) node draws a per-batch class schedule ∝ its weights via an
annbatch :class:`~annbatch.samplers.ClassSampler`; every bound child replays that schedule onto its own
cells via an annbatch :class:`~annbatch.samplers.BoundClassSampler` — matched by *label* on the bind's
``common`` columns (select via child weights + project via ``common``). The loader
never wraps annbatch's RNG: every sampler that must agree within a pass (the schedule oracle, the target
reps, and each child's inner) is **reseeded from one per-pass seed** ``(node seed, pass index)``, so
target/condition/source stay aligned, a node's reps read the same rows, and a pickled loader resumes the
exact same stream. See ``README.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from importlib.util import find_spec

import anndata as ad
import numpy as np
import pandas as pd
from annbatch import DatasetCollection, Loader
from annbatch.samplers import BoundClassSampler, ClassSampler

from dagloader._io import _readable, key_backings, leaf_codes, materialize_node, obs_columns
from dagloader._schema import Bind, Container, SamplerConfig, Scheme, _weight_vector

__all__ = ["DAGLoader"]

# annbatch's GPU path (cupy vstack/indexing → `to="jax"` yields a GPU-resident array via dlpack) needs
# cupy. When it's absent (CPU-only envs — Mac, CI), fall back so `to="jax"` still yields a CPU jax array.
_HAS_CUPY = find_spec("cupy") is not None


def _is_backed(arr) -> bool:
    """True if ``arr`` is an on-disk backing (dense zarr array / backed ``CSRDataset``), not in-memory."""
    import zarr
    from anndata.abc import CSRDataset

    return isinstance(arr, zarr.Array | CSRDataset)


def _group_rep(g, key: str):
    """Rep ``key`` of a collection's zarr group as a readable backing (dense array / wrapped CSR group)."""
    if key == "X":
        return _readable(g["X"])
    field, sub = key.split("/", 1)  # "obsm/X_pca" | "layers/log1p"
    return _readable(g[field][sub])


def _flat_categorical(codes: np.ndarray, leaves: list[tuple]) -> pd.Categorical:
    """A tuple-labelled categorical: per-cell leaf code over ``leaves`` (categories are the leaf tuples).

    Tuple labels (not opaque integer codes) are what let :class:`~annbatch.samplers.BoundClassSampler`
    match a child to its parent by the bind's ``common`` columns — it projects the label by position.
    """
    categories = pd.MultiIndex.from_tuples(leaves).to_flat_index()
    return pd.Categorical.from_codes(codes, categories=categories)


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

        # resolve each node's source; a `Node.in_memory` node is materialized into RAM once (see _io).
        self._nodes: dict[str, Container] = {
            name: (materialize_node(scheme.sources[node.source], node) if node.in_memory else scheme.sources[node.source])
            for name, node in scheme.nodes.items()
        }

        # per-node leaf partition + weights + tuple-labelled categorical (obs only — no cell matrices).
        # Nodes over the same source object + cols (e.g. the perturbed root and its matched-control child)
        # produce identical `(codes, leaves)` — only their weights differ — so read obs and factorize ONCE
        # per (source, cols) and reuse. This avoids re-reading/re-factorizing the (100M-row) obs per node.
        self._st: dict[str, dict] = {}
        leaf_cache: dict[tuple[int, tuple[str, ...]], tuple[np.ndarray, list[tuple]]] = {}
        for name, node in scheme.nodes.items():
            src = self._nodes[name]
            ck = (id(src), tuple(node.cols))
            if ck not in leaf_cache:
                leaf_cache[ck] = leaf_codes(obs_columns(src, node.cols), node.cols)
            codes, leaves = leaf_cache[ck]
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
        """Normalize to one config per node (nodes may use different batch sizes).

        An ``in_memory`` node is materialized into RAM (see :func:`~dagloader._io.materialize_node`), so
        chunked contiguous reads buy it nothing and the run-length rule is meaningless for it — force
        ``chunk_size=1`` there. Its in-RAM reads then carry no run-length constraint, so a matched-control
        child in memory never blocks a ``chunk_size > 1`` perturbed stream; only streamed nodes' on-disk
        layouts have to satisfy the rule. ``preload_nchunks`` must stay a positive multiple of
        ``batch_size // chunk_size`` (``= batch_size`` at chunk 1), so it is rescaled to preserve the read
        window's cell count.
        """
        if isinstance(cfg, SamplerConfig):
            resolved = dict.fromkeys(self.s.nodes, cfg)
        else:
            missing = set(self.s.nodes) - set(cfg)
            if missing:
                raise ValueError(f"sampler_config mapping is missing node(s): {sorted(missing)}.")
            resolved = {name: cfg[name] for name in self.s.nodes}
        for name, node in self.s.nodes.items():
            c = resolved[name]
            if node.in_memory and c.chunk_size != 1:
                windows = max(1, round(c.chunk_size * c.preload_nchunks / c.batch_size))
                resolved[name] = replace(c, chunk_size=1, preload_nchunks=windows * c.batch_size)
        return resolved

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
        inner: ClassSampler,
        *,
        on: dict[int, int] | None,
        classes: pd.Categorical | None = None,
        class_weights: np.ndarray | None = None,
    ) -> BoundClassSampler:
        cfg = self._cfg[b.child]
        try:
            return BoundClassSampler(
                inner,
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
        src = self._nodes[name]
        cfg = self._cfg[name]
        preload_to_gpu = cfg.preload_to_gpu if cfg.preload_to_gpu is not None else _HAS_CUPY  # None ⇒ auto
        self._samplers[name], self._loaders[name] = {}, {}
        for ki, key in enumerate(node.keys):
            sampler = make_sampler()
            return_index = name == self.s.root and ki == 0  # only for the schedule↔row alignment check
            # `to` (default "jax") + `preload_to_gpu` are user-set via SamplerConfig. `to="jax"` yields
            # native jax arrays (no host round-trip); `preload_to_gpu` keeps the read window on-GPU (needs
            # cupy), else it defers the device copy to the step. Auto-selects cupy when unset.
            base = Loader(batch_sampler=sampler, return_index=return_index, to=cfg.to, preload_to_gpu=preload_to_gpu)
            loader = self._attach(base, src, key)
            self._samplers[name][key], self._loaders[name][key] = sampler, loader

    def _attach(self, loader: Loader, src: Container, key: str) -> Loader:
        """Feed rep ``key`` of ``src`` to a fresh ``Loader`` via the source-appropriate annbatch entry point.

        Dispatch by source kind, streaming only rep ``key`` with **no obs** — dagloader owns the class
        labels through the sampler (``classes=``), so annbatch never needs the source's obs:

        * ``DatasetCollection`` → :meth:`~annbatch.Loader.use_collection` (annbatch's own collection API),
          each group's rep loaded as an obs-free ``X`` (the default ``load_adata`` would decode *all* obs);
        * **backed** ``AnnData`` / list of backed ``AnnData`` → the raw rep backings through ``add_datasets``;
        * **in-memory** ``AnnData`` (a user adata, or a materialized ``in_memory`` node such as the matched
          control) → ``add_adatas`` over the rep wrapped as an obs-free ``X``.
        """
        if isinstance(src, DatasetCollection):
            return loader.use_collection(src, load_adata=lambda g: ad.AnnData(X=_group_rep(g, key)))
        backings = key_backings(src, key)
        if isinstance(src, ad.AnnData) and not _is_backed(backings[0]):  # in-memory adata → add_adatas
            return loader.add_adatas([ad.AnnData(X=b) for b in backings])
        return loader.add_datasets(backings)  # backed adata or list of backed adata

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
