"""Declarative schema for :class:`~dagloader.DAGLoader`.

A :class:`Scheme` is a rooted tree of :class:`Node`\\s over named cell *sources* — pure structure
(sources, grouping columns, weights, binds). How each node is *read* (chunk / preload / batch sizes)
lives in a separate :class:`SamplerConfig` passed to the loader, deliberately kept off the ``Node`` so
the same structure can be run with different sampler settings.

Each node partitions its source's cells into **leaves** (unique combinations of ``cols``) with a
per-combination :data:`Weights` mapping. A weight of 0 (or a combination absent from the mapping) is
*excluded* — that IS the selection, native to annbatch's ``ClassSampler``. :class:`Bind` links a
parent to a child on shared columns, so the child is sampled *conditioned* on the parent's values.
See ``README.md`` for the model and the cellflow / sc-flow-tools mapping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import anndata as ad
import numpy as np
from annbatch import DatasetCollection

# A cell source: an in-memory AnnData or an out-of-core annbatch DatasetCollection.
Container = ad.AnnData | DatasetCollection

# A sampling scheme is just a mapping {combination -> weight}. A combination absent from the mapping
# (or with weight 0) is excluded — that IS the selection. ``uniform`` / ``frequency`` /
# ``inverse_frequency`` are plain helpers that build such a dict; nothing about them is privileged.
Weights = Mapping[tuple, float]

__all__ = [
    "Bind",
    "Container",
    "Node",
    "SamplerConfig",
    "Scheme",
    "Weights",
    "frequency",
    "inverse_frequency",
    "uniform",
]


def uniform(combos: Sequence[tuple]) -> dict[tuple, float]:
    """Every combination equally likely."""
    return {tuple(c): 1.0 for c in combos}


def frequency(counts: Mapping[tuple, int]) -> dict[tuple, float]:
    """Sample each combination ∝ its cell count (favor abundant conditions)."""
    return {tuple(k): float(c) for k, c in counts.items()}


def inverse_frequency(counts: Mapping[tuple, int]) -> dict[tuple, float]:
    """Sample each combination ∝ 1 / cell count (balance rare vs abundant conditions)."""
    return {tuple(k): 1.0 / c for k, c in counts.items()}


def _weight_vector(weights: Weights, leaves: Sequence[tuple]) -> np.ndarray:
    """Resolve ``{combo: weight}`` to normalized per-leaf weights (→ ``ClassSampler.class_weights``)."""
    v = np.array([float(weights.get(tuple(lf), 0.0)) for lf in leaves], dtype=float)
    s = v.sum()
    if s <= 0:
        raise ValueError("weights resolve to all-zero over these leaves — nothing to sample.")
    return v / s


@dataclass(frozen=True)
class Node:
    """A partition of one source's cells into leaves, with a per-leaf sampling weight.

    Parameters
    ----------
    source
        Key into :attr:`Scheme.sources`.
    cols
        Tree levels → leaves are the unique combinations of these columns (over ALL the source's
        cells). These are the grouping/condition columns (cellflow's ``split_covariates`` +
        ``perturbation_covariates`` columns; sc-flow-tools' grouping keys).
    keys
        Representation location(s) to stream: ``"X"`` | ``"obsm/<k>"`` | ``"layers/<k>"``
        (cellflow's ``sample_rep``). A single string streams one rep; a tuple streams SEVERAL
        **aligned** reps of the *same* sampled cells — e.g. the state plus a per-cell continuous
        condition. The first key drives sampling (via annbatch's ``ClassSampler``); the rest are
        read back for the exact same rows, so every rep of a batch is the same cells.
    weights
        ``{combo: weight}``; a combination absent or with weight 0 is excluded (= the selection).
    """

    source: str
    cols: tuple[str, ...]
    keys: str | tuple[str, ...] = "X"  # one rep, or several aligned reps of the same cells
    weights: Weights = field(default_factory=dict)

    def __post_init__(self) -> None:  # structural checks (data-free)
        object.__setattr__(self, "keys", (self.keys,) if isinstance(self.keys, str) else tuple(self.keys))
        if not self.cols:
            raise ValueError("Node.cols must be non-empty.")
        if not self.keys or any(not k for k in self.keys):
            raise ValueError("Node.keys must be one or more non-empty representation locations.")
        for k in self.weights:
            if len(k) != len(self.cols):
                raise ValueError(f"weight key {k!r} arity != cols {self.cols}.")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("weights must be non-negative.")


@dataclass(frozen=True)
class Bind:
    """Condition ``child`` on ``parent``: match on the ``common`` columns (⊆ their shared cols).

    Each batch, the child's sampled leaf is derived from the parent's leaf via the ``common`` values
    (parent leaf → shared-column values → matching child leaf). This is the source↔target matching:
    with ``common`` = the context (e.g. cell line), the child (control) is drawn from the *same*
    context as the parent (perturbed) — cellflow's "control = same group", sc-flow-tools'
    ``control_values_dict`` + default same-context coupling.

    Conditioning is **required**: if a parent value has no matching positive-weight child leaf the
    loader raises (no silent fallback). When several child leaves share the bound value — the child
    partitions on columns beyond ``common`` (e.g. child cols ``(a, x)`` bound on ``a``) — one is drawn
    ∝ the child's leaf weights, so ``P(child extra cols | common)`` is weight-controlled. Pass
    ``common=()`` to opt into unconditional child sampling explicitly.

    For pairings that are NOT a shared-column match — sc-flow-tools' ``matched_keys`` (explicit
    source↔target pairs) — pass ``matched={parent_leaf: child_leaf, ...}`` instead of ``common``. Each
    parent leaf then maps to exactly one child leaf; a root leaf absent from the map raises.
    """

    parent: str
    child: str
    common: tuple[str, ...] = ()  # ⊆ parent.cols ∩ child.cols; () ⇒ unconditional (unless `matched`)
    matched: Mapping[tuple, tuple] | None = None  # explicit parent-leaf → child-leaf pairing; overrides `common`

    def __post_init__(self) -> None:
        if self.matched is not None and self.common:
            raise ValueError("Bind: give either `common` (column matching) or `matched` (explicit pairing), not both.")


@dataclass(frozen=True)
class SamplerConfig:
    """annbatch read parameters for a node's sampler — kept separate from the structural :class:`Node`.

    Passed to :class:`~dagloader.DAGLoader` as either one config (applied to every node)
    or a ``{node_name: SamplerConfig}`` mapping (per-node). Nodes may use **different** ``batch_size``\\s:
    every node draws the same number of batches (the root's, derived from its cell count), but a node's
    batch carries its own row count — so source and target row counts need not match.

    Parameters
    ----------
    batch_size
        Rows per emitted batch for this node (target rows for the root; source rows for a bound child).
    chunk_size
        annbatch read-slice size. ``1`` (default) ⇒ per-row reads (any on-disk layout). ``>1`` ⇒
        contiguous chunked reads (higher throughput on disk), assuming each sampled leaf sits in a
        contiguous run ≥ ``chunk_size``. Must divide ``batch_size`` (one category per batch).
    preload_nchunks
        Chunks per annbatch read window. ``None`` (default) ⇒ ``batch_size // chunk_size`` (one batch
        per window). If given, must be a positive multiple of ``batch_size // chunk_size``.
    """

    batch_size: int
    chunk_size: int = 1
    preload_nchunks: int

@dataclass(frozen=True)
class Scheme:
    """The structural sampling spec: sources, a rooted tree of nodes, and the reproducibility cadence.

    Read parameters (chunk / preload / batch sizes) are NOT here — they are a separate
    :class:`SamplerConfig` given to the loader.

    Parameters
    ----------
    sources
        ``{name: AnnData | DatasetCollection}`` — the cell sources the nodes reference.
    nodes
        ``{name: Node}``. Exactly one is the ``root`` (the streamed target); the rest are bound
        children (sources/controls) via ``binds``.
    root
        Name of the root node (must have no parent).
    seed
        Reproducibility seed. Per-node RNG streams are spawned from one ``SeedSequence(seed)`` so nodes
        do not correlate and the whole stream is reproducible.
    binds
        Parent→child links (see :class:`Bind`). Must form a rooted tree over ``nodes``.

    Notes
    -----
    Batches per with-replacement pass is *not* configured here: the loader derives it from the root
    (target) node — a natural epoch of ``root_n_obs // batch_size`` — and restarts each pass. The root
    drives the zip, so every node's sampler draws the same number of batches.
    """

    sources: Mapping[str, Container]
    nodes: Mapping[str, Node]
    root: str
    seed: int
    binds: tuple[Bind, ...] = ()

    def __post_init__(self) -> None:  # structural: rooted tree + references
        if self.root not in self.nodes:
            raise ValueError(f"root {self.root!r} not in nodes.")
        for name, n in self.nodes.items():
            if n.source not in self.sources:
                raise ValueError(f"node {name!r} references unknown source {n.source!r}.")
        parents: dict[str, str] = {}
        for b in self.binds:
            if b.parent not in self.nodes or b.child not in self.nodes:
                raise ValueError("bind references unknown node.")
            if b.child in parents:
                raise ValueError(f"node {b.child!r} has multiple parents — must be a rooted tree.")
            parents[b.child] = b.parent
            shared = set(self.nodes[b.parent].cols) & set(self.nodes[b.child].cols)
            if not set(b.common) <= shared:
                raise ValueError(f"bind.common {b.common} must be ⊆ shared cols of {b.parent}&{b.child} ({shared}).")
        if self.root in parents:
            raise ValueError("root must have no parent.")
        for name in self.nodes:
            if name != self.root and name not in parents:
                raise ValueError(f"non-root node {name!r} is not bound to the tree.")
