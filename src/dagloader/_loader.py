"""``DAGLoader`` — streams matched ``{source, target, condition}`` batches from a :class:`Scheme`.

Every node streams through its own :class:`~dagloader.ScheduledClassSampler` + annbatch
``Loader``, configured by a :class:`SamplerConfig` (one shared, or one per node). Each pass, the loader
draws the root's per-batch category schedule from the root's weights and *derives* each bound child's
schedule from the parent's (via the bind's shared columns), pushes the schedules onto the samplers, and
zips the loaders batch-for-batch. See ``README.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

import numpy as np
import pandas as pd
from annbatch import Loader

from dagloader._io import key_backings, leaf_codes, obs_columns
from dagloader._scheduled_sampler import ScheduledClassSampler
from dagloader._schema import Bind, SamplerConfig, Scheme, _weight_vector

__all__ = ["DAGLoader"]


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

        self._children: dict[str, list[Bind]] = {}
        for b in scheme.binds:
            self._children.setdefault(b.parent, []).append(b)

        # per-node independent RNG streams from one seed (by sorted node name). Each node splits into a
        # *decision* rng (root schedule draw / child tie-break) and a *row* seed shared identically by
        # that node's per-key samplers — identical samplers → the SAME rows, so a node's reps are aligned.
        self._rngs: dict[str, np.random.Generator] = {}
        self._row_seqs: dict[str, np.random.SeedSequence] = {}
        for name, seq in zip(
            sorted(scheme.nodes), np.random.SeedSequence(scheme.seed).spawn(len(scheme.nodes)), strict=True
        ):
            decision_seq, row_seq = seq.spawn(2)
            self._rngs[name] = np.random.default_rng(decision_seq)
            self._row_seqs[name] = row_seq

        # per-node leaf partition + weights (obs only — no cell matrices)
        self._st: dict[str, dict] = {}
        for name, node in scheme.nodes.items():
            obs = obs_columns(scheme.sources[node.source], node.cols)
            codes, leaves = leaf_codes(obs, node.cols)
            self._st[name] = {"node": node, "codes": codes, "leaves": leaves, "w": _weight_vector(node.weights, leaves)}

        # Batches per pass is derived, not configured: a natural epoch over the root (target) node —
        # its cell count // the root's batch_size. The root drives the zip, so every node draws the same
        # number of batches; each node's num_samples == _n_batches * that node's batch_size (see
        # `_build_loaders`), which lets nodes use different batch sizes (source rows need not equal target).
        n_root_obs = len(self._st[self.s.root]["codes"])
        self._n_batches = max(1, n_root_obs // self._B)

        self._build_loaders()
        self._build_bind_maps()
        self._iters: dict[str, Iterator[dict]] | None = None
        self._schedules: dict[str, np.ndarray] = {}
        self._pos = 0

    def _resolve_configs(self, cfg: SamplerConfig | Mapping[str, SamplerConfig]) -> dict[str, SamplerConfig]:
        """Normalize to one config per node.

        Nodes may use different batch sizes (each draws ``_n_batches`` batches of its own size; source
        and target row counts need not match).
        """
        if isinstance(cfg, SamplerConfig):
            resolved = dict.fromkeys(self.s.nodes, cfg)
        else:
            missing = set(self.s.nodes) - set(cfg)
            if missing:
                raise ValueError(f"sampler_config mapping is missing node(s): {sorted(missing)}.")
            resolved = {name: cfg[name] for name in self.s.nodes}
        return resolved

    # ── build ────────────────────────────────────────────────────────────
    def _build_loaders(self) -> None:
        """Per ``(node, key)``: a ScheduledClassSampler + Loader.

        All of a node's keys are seeded with the SAME row-draw seed, so their (otherwise identical)
        samplers select the same rows every batch — every rep of a node is the same cells. Reps need
        separate Loaders regardless (annbatch can't mix feature dims in one loader), and each rep gets
        native chunked reads.
        """
        self._samplers: dict[str, dict[str, ScheduledClassSampler]] = {}
        self._loaders: dict[str, dict[str, Loader]] = {}
        for name, st in self._st.items():
            node = st["node"]
            src = self.s.sources[node.source]
            cfg = self._cfg[name]
            # Per node: _n_batches batches of this node's own batch_size. Same batch count across nodes
            # (the root's _n_batches) → schedules align; sizes may differ (source rows ≠ target rows OK).
            num_samples = self._n_batches * cfg.batch_size
            K = len(st["leaves"])
            classes = pd.Categorical([str(c) for c in st["codes"]], categories=[str(i) for i in range(K)])
            self._samplers[name], self._loaders[name] = {}, {}
            for ki, key in enumerate(node.keys):
                try:  # annbatch enforces its own run-length rule for chunk>1; forward with node context
                    sampler = ScheduledClassSampler(
                        chunk_size=cfg.chunk_size,
                        preload_nchunks=cfg.preload_nchunks,
                        batch_size=cfg.batch_size,
                        classes=classes,
                        num_samples=num_samples,
                        class_weights=st["w"],
                        drop_last=True,
                        rng=np.random.default_rng(self._row_seqs[name]),  # identical across keys
                    )
                except ValueError as e:
                    raise ValueError(f"node {name!r}: {e}") from e
                return_index = name == self.s.root and ki == 0  # only for the schedule↔row alignment check
                loader = Loader(
                    batch_sampler=sampler, return_index=return_index, to_torch=False, preload_to_gpu=False
                ).add_datasets(key_backings(src, key))
                self._samplers[name][key] = sampler
                self._loaders[name][key] = loader

    def _nodes_next(self, name: str) -> dict[str, np.ndarray]:
        """One batch per key of a node — identical samplers pick the same rows, so the reps are aligned."""
        node = self._st[name]["node"]
        return {key: next(self._iters[name][key])["X"] for key in node.keys}

    def _build_bind_maps(self) -> None:
        """Precompute, per bound child, the maps to turn the parent's schedule into the child's."""
        self._bindmap: dict[str, dict] = {}
        for b in self._children.get(self.s.root, []):
            rst, cst = self._st[b.parent], self._st[b.child]
            rleaves, cleaves = rst["leaves"], cst["leaves"]
            if b.matched is not None:  # explicit parent-leaf → child-leaf pairing (sc-flow's matched_keys)
                root_idx = {lf: i for i, lf in enumerate(rleaves)}
                child_idx = {lf: i for i, lf in enumerate(cleaves)}
                code_to_child: dict[int, int] = {}
                for pk, ck in b.matched.items():
                    pk, ck = tuple(pk), tuple(ck)
                    if pk not in root_idx:
                        raise ValueError(
                            f"bind {b.parent!r}→{b.child!r}: matched key {pk!r} is not a {b.parent!r} leaf."
                        )
                    if ck not in child_idx or cst["w"][child_idx[ck]] <= 0:
                        raise ValueError(
                            f"bind {b.parent!r}→{b.child!r}: matched value {ck!r} is not a positive-weight {b.child!r} leaf."
                        )
                    code_to_child[root_idx[pk]] = child_idx[ck]
                self._bindmap[b.child] = {
                    "bind": b,
                    "mode": "matched",
                    "root_leaves": rleaves,
                    "code_to_child": code_to_child,
                }
                continue
            rcols, ccols = rst["node"].cols, cst["node"].cols
            # parent leaf code → shared-column value (empty tuple when common=() ⇒ unconditional bind)
            root_code_to_cv = {i: tuple(lf[rcols.index(c)] for c in b.common) for i, lf in enumerate(rleaves)}
            # shared-column value → positive-weight child leaf codes carrying it (key () holds all when common=())
            grouped: dict[tuple, list[int]] = {}
            for code, lf in enumerate(cleaves):
                if cst["w"][code] > 0:
                    grouped.setdefault(tuple(lf[ccols.index(c)] for c in b.common), []).append(code)
            self._bindmap[b.child] = {
                "bind": b,
                "mode": "common",
                "root_leaves": rleaves,
                "child_w": cst["w"],
                "root_code_to_cv": root_code_to_cv,
                "common_to_child": {cv: np.asarray(codes, dtype=np.int64) for cv, codes in grouped.items()},
            }

    # ── per-pass scheduling ────────────────────────────────────────────────
    def _draw_root_schedule(self) -> np.ndarray:
        """Draw one root leaf code per batch (``_n_batches``) ∝ the root's weights (its own RNG stream)."""
        w = self._st[self.s.root]["w"]
        pos = np.flatnonzero(w > 0)
        return self._rngs[self.s.root].choice(pos, size=self._n_batches, p=w[pos] / w[pos].sum()).astype(np.int64)

    def _derive_child_schedule(self, child: str, root_sched: np.ndarray) -> np.ndarray:
        """Map each root batch's category to a matching child leaf.

        ``matched`` binds map each parent leaf to exactly one child leaf. Otherwise the child is matched
        on ``common`` columns; when several child leaves share the bound value (the child partitions on
        columns beyond ``common``, e.g. child cols ``(a, x)`` bound on ``a``) one is drawn ∝ the child's
        leaf weights, so ``P(child extra cols | common)`` is weight-controlled. Conditioning is required:
        an unmatched value raises (no silent fallback); ``Bind(..., common=())`` opts into unconditional.
        """
        m = self._bindmap[child]
        b = m["bind"]
        out = np.empty(len(root_sched), dtype=np.int64)
        if m["mode"] == "matched":
            cmap = m["code_to_child"]
            for j, rc in enumerate(root_sched):
                rc = int(rc)
                if rc not in cmap:
                    raise ValueError(
                        f"bind {b.parent!r}→{b.child!r}: no matched child leaf for root leaf "
                        f"{m['root_leaves'][rc]!r}. Add it to `matched`."
                    )
                out[j] = cmap[rc]
            return out
        rng = self._rngs[child]
        cw = m["child_w"]
        for j, rc in enumerate(root_sched):
            cv = m["root_code_to_cv"][int(rc)]
            codes = m["common_to_child"].get(cv)
            if codes is None:  # no positive-weight child leaf carries this value — do NOT silently fall back
                raise ValueError(
                    f"bind {b.parent!r}→{b.child!r}: no positive-weight child leaf matches common "
                    f"{b.common}={cv!r} (root leaf {m['root_leaves'][int(rc)]!r}). Add a child leaf for "
                    f"it, or declare Bind(..., common=()) to sample {b.child!r} unconditionally."
                )
            out[j] = int(codes[0]) if len(codes) == 1 else int(rng.choice(codes, p=cw[codes] / cw[codes].sum()))
        return out

    def _start_pass(self) -> None:
        """Draw schedules (root + derived children), push them onto the samplers, rebuild iterators.

        Ordering matters: ``Loader.__iter__`` re-reads ``sampler.sample()`` and the scheduled draw is
        read up front, so ``set_schedule`` must land before ``iter(loader)``.
        """
        root_sched = self._draw_root_schedule()
        self._schedules = {self.s.root: root_sched}
        for smp in self._samplers[self.s.root].values():  # every key of the node gets the same schedule
            smp.set_schedule(root_sched)
        for b in self._children.get(self.s.root, []):
            child_sched = self._derive_child_schedule(b.child, root_sched)
            self._schedules[b.child] = child_sched
            for smp in self._samplers[b.child].values():
                smp.set_schedule(child_sched)
        self._iters = {name: {key: iter(ld) for key, ld in loaders.items()} for name, loaders in self._loaders.items()}
        self._pos = 0

    # ── pickling ─────────────────────────────────────────────────────────────
    def __getstate__(self) -> dict:
        """Pickle without the live annbatch iterators (generators aren't picklable).

        Everything that defines the stream is kept — the per-node RNG streams (``_rngs`` / ``_row_seqs``),
        the samplers (with their advanced RNG), the drawn ``_schedules`` and the configs — so a reloaded
        loader continues the *same* reproducible RNG sequence. Only the transient ``_iters`` are dropped;
        the next ``__next__`` rebuilds them (a fresh pass) from the restored RNG state.
        """
        state = self.__dict__.copy()
        state["_iters"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._iters = None  # force `_start_pass` on the next `__next__`, using the restored RNG state

    # ── iteration ──────────────────────────────────────────────────────────
    def __iter__(self) -> DAGLoader:
        return self

    def __next__(self) -> dict[str, np.ndarray]:
        if self._iters is None or self._pos >= self._n_batches:
            self._start_pass()
        j = self._pos
        self._pos += 1

        st = self._st[self.s.root]
        node = st["node"]
        leaf = st["leaves"][int(self._schedules[self.s.root][j])]  # per-batch category — from the schedule
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

        for b in self._children.get(self.s.root, []):  # bound child source, conditioned via its schedule
            cnode = self._st[b.child]["node"]
            creps = self._nodes_next(b.child)
            out["source"] = creps[cnode.keys[0]]
            if len(cnode.keys) > 1:
                out["source_reps"] = creps
        return out
