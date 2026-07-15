"""``DAGEvalLoader`` — control-rooted eval reader: a Sequential control *inner* + a bound perturbed target.

Anchors on the source (control) — "there is no perturbed without a source". Two loaders, both driven by
one deterministic :class:`~annbatch.samplers.SequentialClassSampler` over the control populations:

* the **source** loader *is* that inner — it reads each scheduled control population **in full** (all its
  controls), and
* the **target** loader is an :class:`~annbatch.samplers.BoundClassSampler` on the *same* inner, matched on
  the bind's ``common`` (context) columns, that **samples** a perturbed leaf (drug) within each matched
  context.

annbatch does all the class matching; nothing is derived or updated per pass here. The condition of each
batch is the perturbed leaf the bound drew (read from an identically-seeded oracle, so it lines up with
the target loader's own draw). Which control populations are visited — and how many batches — is set via
the schedule (``iter_conditions``). Works over an in-memory ``AnnData`` and a ``DatasetCollection`` alike.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

import numpy as np
from annbatch import Loader
from annbatch.samplers import BoundClassSampler, SequentialClassSampler

from dagloader._io import key_backings, leaf_codes, materialize_node, obs_columns
from dagloader._loader import _HAS_CUPY, _flat_categorical
from dagloader._schema import SamplerConfig, Scheme, _weight_vector

__all__ = ["DAGEvalLoader"]


class DAGEvalLoader:
    """Yield ``{"source", "target", "condition", "leaf"}`` per control population (source-anchored).

    Parameters
    ----------
    scheme
        A prepared :class:`~dagloader.Scheme` (root = perturbed/target node, one bound control child).
        Its bind's ``common`` columns are the matching context (e.g. ``cell_line``).
    sampler_config
        Read parameters for the bound perturbed-target sampler (``batch_size`` = target cells per
        condition; controls are read in full regardless).
    condition_fn
        Maps a perturbed leaf (over the target node's ``cols``) to its condition embedding.
    seed
        Seed for the target's drug sampling (reproducible across ``iter_conditions`` calls).
    """

    def __init__(
        self,
        scheme: Scheme,
        sampler_config: SamplerConfig,
        condition_fn: Callable[[tuple], np.ndarray | Mapping[str, np.ndarray]] | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.s = scheme
        self._cond_fn = condition_fn
        self._cfg = sampler_config
        self._seed = seed
        binds = [b for b in scheme.binds if b.parent == scheme.root]
        if len(binds) != 1:
            raise ValueError("DAGEvalLoader expects exactly one bound source of the root.")
        b = binds[0]
        self._pert = scheme.nodes[scheme.root]  # perturbed / target
        self._ctrl = scheme.nodes[b.child]  # control / source
        self._context = b.common
        # control source honors Node.in_memory (materialize the control cells into RAM once)
        ctrl_src = scheme.sources[self._ctrl.source]
        self._src = materialize_node(ctrl_src, self._ctrl) if self._ctrl.in_memory else ctrl_src
        self._src_p = scheme.sources[self._pert.source]

        # per-node tuple-labelled categorical + weight vector (obs only). Control/perturbed cells live in
        # the same source; the weights mark which leaves belong to each node.
        cc, cl = leaf_codes(obs_columns(self._src, self._ctrl.cols), self._ctrl.cols)
        pc, pl = leaf_codes(obs_columns(self._src_p, self._pert.cols), self._pert.cols)
        self._ctrl_cats = _flat_categorical(cc, cl)
        self._pert_cats = _flat_categorical(pc, pl)
        self._ctrl_w = _weight_vector(self._ctrl.weights, cl)
        self._pert_w = _weight_vector(self._pert.weights, pl)
        self._ctrl_leaves = cl
        # inner (control) tuple position -> target tuple position, for each shared context column
        self._on = {self._ctrl.cols.index(c): self._pert.cols.index(c) for c in self._context}
        # the control populations to visit (positive-weight control leaves)
        self._ctrl_codes = np.array([i for i in range(len(cl)) if self._ctrl_w[i] > 0], dtype=np.int64)
        if self._ctrl_codes.size == 0:
            raise ValueError("no control population (positive-weight control leaf) to evaluate.")

    @property
    def control_populations(self) -> list[tuple]:
        """The control leaves (contexts) this loader iterates over."""
        return [self._ctrl_leaves[i] for i in self._ctrl_codes]

    def _inner(self, schedule: np.ndarray) -> SequentialClassSampler:
        # deterministic: same schedule ⇒ same control-population order across the source loader and every
        # bound inner, so source[j] and target[j] refer to the same matched context.
        return SequentialClassSampler(self._ctrl_cats, schedule=schedule)

    def _bound(self, schedule: np.ndarray) -> BoundClassSampler:
        cfg = self._cfg
        return BoundClassSampler(
            self._inner(schedule),
            cfg.chunk_size,
            cfg.preload_nchunks,
            cfg.batch_size,
            classes_to_bind_on=self._pert_cats,
            on=self._on,
            classes=self._pert_cats,  # secondary = the perturbed leaf; weights pick positive-weight drugs
            class_weights=self._pert_w,
            rng=np.random.default_rng(self._seed),
        )


    # TODO(selmanozleyen): rename to _loaders_from_node
    def _node_loaders(self, src, node, make_sampler) -> dict:
        cfg = self._cfg
        preload_to_gpu = cfg.preload_to_gpu if cfg.preload_to_gpu is not None else _HAS_CUPY  # None ⇒ auto
        loaders = {}
        for key in node.keys:
            loader = Loader(
                batch_sampler=make_sampler(), return_index=False, to=cfg.to, preload_to_gpu=preload_to_gpu
            ).add_datasets(key_backings(src, key))
            loaders[key] = loader
        return loaders

    def iter_conditions(self, n_conditions: int | None = None) -> Iterator[dict]:
        """Yield one batch per scheduled control population.

        With ``n_conditions`` set, the control populations are cycled to that many batches (each re-reads
        the population's controls and the bound samples a fresh drug); otherwise every control population
        is visited once.
        """
        if n_conditions is None:
            schedule = self._ctrl_codes.copy()
        else:
            reps = int(np.ceil(n_conditions / self._ctrl_codes.size))
            schedule = np.tile(self._ctrl_codes, reps)[:n_conditions].astype(np.int64)

        # condition oracle: identically-seeded bound → its per-batch drawn perturbed leaf lines up with
        # the target loader's own draw (annbatch reproduces the same class sequence from the same seed).
        oracle = self._bound(schedule)
        vocab = oracle.vocab
        ctx = len(self._context)
        cond_leaves = [tuple(vocab[int(c)])[ctx:] for c in oracle.batch_codes()]  # strip the shared context prefix

        src_iters = {
            k: iter(ld) for k, ld in self._node_loaders(self._src, self._ctrl, lambda: self._inner(schedule)).items()
        }
        tgt_iters = {
            k: iter(ld) for k, ld in self._node_loaders(self._src_p, self._pert, lambda: self._bound(schedule)).items()
        }
        skeys, tkeys = list(src_iters), list(tgt_iters)

        for j in range(len(schedule)):
            src = {k: next(src_iters[k])["X"] for k in skeys}
            tgt = {k: next(tgt_iters[k])["X"] for k in tkeys}
            leaf = cond_leaves[j]
            out: dict = {"leaf": leaf, "source": src[skeys[0]], "target": tgt[tkeys[0]]}
            if len(skeys) > 1:
                out["source_reps"] = src
            if len(tkeys) > 1:
                out["target_reps"] = tgt
            if self._cond_fn is not None:
                out["condition"] = self._cond_fn(leaf)
            yield out
