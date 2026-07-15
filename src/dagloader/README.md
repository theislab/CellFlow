# `dagloader` — declarative, index-free sampling over annbatch

A small, declarative layer that turns a **scheme of columns + weights** into a stream of matched
`{source, target, condition}` training batches, read (in- or out-of-core) through
[`annbatch`](https://github.com/laminlabs/annbatch). It is the streaming sampler that both
**cellflow** (this repo's model) and **sc-flow-tools** (`theislab/sc-flow-tools`) can share: one
condition per batch, source↔target matching, categorical + continuous condition payloads, and
reproducible weighted sampling — expressed as data, not as materialized row-index tables.

```
Scheme (sources + nodes + binds + weights)   +   SamplerConfig (batch/chunk/preload)
        │                                                 │
        └───────────────────► DAGLoader ◄────────────┘
                                    │
        {"target": X[perturbed], "source": X[matched control], "condition": emb}
                                    │
        ├─ every node streams through its own ScheduledClassSampler + annbatch.Loader
        └─ root category ~ weights; each bound child's category derived from the parent's
```

Structure (`Scheme`) and read parameters (`SamplerConfig`) are separate objects, both handed to the
loader — so the same scheme can run with different batch/chunk settings, and neither lives on the
`Node`.

---

## 1. Mental model

- **Source** — a named cell store: an in-memory `AnnData` **or** an out-of-core
  `annbatch.DatasetCollection`. The loader never branches on which; obs is read for grouping, cells
  are read only when a batch is materialized.
- **Node** — a partition of *one source's* cells into **leaves** = the unique combinations of its
  `cols`, plus a `{combo: weight}` mapping. `cols` are the grouping / condition columns; `key` is the
  single representation to stream (`"X"`, `"obsm/X_pca"`, `"layers/…"`) — add another node for another
  rep. The `Node` carries no read parameters; those live in `SamplerConfig` (below).
- **Weight = the selection.** A combination with weight `0` (or absent from the mapping) is simply
  not sampled. There is **no separate `select` mask** — inclusion *is* a positive weight. This is
  native to annbatch's `ClassSampler`.
- **Bind(parent, child, common)** — condition the child on the parent: each batch, the child's leaf
  is derived from the parent's leaf by matching the `common` column values. With `common` = the
  context (cell line, …), the child (control/source) is drawn from the **same context** as the parent
  (perturbed/target). This is the source↔target matching.
- **Scheme** — sources + nodes + a rooted tree of binds + the reproducibility `seed`. Pure structure.
  The **root** is the streamed target; bound children are the sources. Batches-per-pass is *not* set
  here — the loader derives it from the root (a natural epoch, `root_n_obs // batch_size`).
- **SamplerConfig** — the annbatch read parameters (`batch_size`, `chunk_size`, `preload_nchunks`),
  kept off the `Node` and the `Scheme`. Passed to the loader as one config (all nodes) or a
  `{node_name: SamplerConfig}` mapping (per-node). Nodes may use **different** `batch_size`\s (source
  and target row counts need not match); the root's drives the shared batch count.
- **DAGLoader** — iterates batches. Each node streams through its own `ScheduledClassSampler`.

### Batch contract

```python
{
  "target":    ndarray (B_t, d),   # root node's streamed rep (Node.key); B_t = root batch_size
  "source":    ndarray (B_s, d),   # bound child's streamed rep — matched to target's context
  "condition": ndarray (B_t, e),   # condition_fn(leaf) broadcast over the target batch (optional)
}
```
Each node's row count is its own `SamplerConfig.batch_size` (`B_t` for the target, `B_s` for the
source) — they need not be equal. Sparse `X` stays sparse.

---

## 2. Sampling schemes

The "sampling scheme" is entirely the **weights** plus the per-node sampler cadence — nothing is
hardcoded.

**Weight builders** (plain functions returning `{combo: weight}`; use them or write the dict yourself):

| helper | meaning |
|---|---|
| `uniform(combos)` | every combination equally likely |
| `frequency(counts)` | sample ∝ cell count (favor abundant conditions) |
| `inverse_frequency(counts)` | sample ∝ 1 / cell count (balance rare vs abundant) |

**`ScheduledClassSampler`** — the one genuinely new piece. It is an annbatch `ClassSampler` whose
per-batch category can be *supplied* (`set_schedule`) instead of drawn internally. It does **not copy**
annbatch's chunk math: `ClassSampler._iter_requests` makes exactly one *weighted* draw
(`rng.choice(n_classes, size=n_groups, p=…)`) to choose each group's class, so `ScheduledClassSampler`
runs that method unchanged but temporarily wraps `self._rng` so that one call returns the scheduled
positions. Every other draw (run selection, slice starts, shuffling) uses the real generator, so a
scheduled pass is bit-identical to what annbatch would produce for those group classes. That is what
makes the bind work:

- the **root** schedule is drawn from the root's weights (one category per batch, `_n_batches` total);
- each **bound child**'s schedule is *derived* from the root's (parent leaf → `common` value →
  matching child leaf; the child's RNG breaks ties / picks a fallback);
- the loaders then zip batch-for-batch with **no per-step reconfiguration**.

With `schedule=None`, `ScheduledClassSampler` is exactly `ClassSampler` (draws ∝ weights) — a strict
superset. (If annbatch exposed a `_group_positions(n_groups)` hook, even the rng-wrapping would go
away and this would be a one-method override — an upstream candidate.)

**Read parameters — `SamplerConfig` (batch / chunk / preload):**

- `batch_size` (`B`): rows per emitted batch for that node. Nodes may differ (target rows need not
  equal source rows) — every node still draws the same *number* of batches (the root's), so the
  schedules zip; only the per-batch row count differs.
- `chunk_size` (default `1`): annbatch read-slice size. `1` = per-row reads (any layout). `>1` =
  contiguous chunked reads for higher on-disk throughput; it assumes each sampled leaf is a contiguous
  run ≥ `chunk_size` (which a condition-sorted collection provides) and must divide `batch_size`.
- `preload_nchunks` (default `batch_size // chunk_size`): chunks per annbatch read window; a larger
  multiple packs more classes into one read.

Pass one `SamplerConfig` (applied to every node) or a `{node_name: SamplerConfig}` mapping (per-node —
e.g. a chunked root and a per-row control).

**Cadence / reproducibility (on the `Scheme`):**

- batches per pass: **derived**, not configured — the loader uses a natural epoch over the root
  (`root_n_obs // root batch_size`) and restarts each pass → effectively infinite with a fixed,
  reproducible restart cadence. All nodes restart together (the root drives the count).
- `seed`: per-node RNG streams are spawned from one `SeedSequence(seed)` (one independent stream per
  node, by sorted name) so nodes never correlate and the full `(source, target)` sequence is
  reproducible from the seed.

> **On layout / sortedness:** the default (`chunk_size=1`) reads per-row and is indifferent to on-disk
> order. `chunk_size>1` is purely a throughput opt-in for a condition-sorted collection; the loader
> does not police layout — it forwards annbatch's own behavior. Ordering never affects *correctness*
> (matching is recovered from the schedule / columns, not from row order).

---

## 3. The cellflow case

cellflow trains a conditional flow from a **control** population to a **perturbed** population,
matched within a context, one perturbation condition per batch. That maps directly onto a two-node
scheme (control child bound to the perturbed root on the context columns):

| cellflow concept | dagloader |
|---|---|
| `sample_rep` (`X` / an obsm key) | `Node.key` |
| `split_covariates` (context / grouping) | `context` → the `Bind.common` columns |
| `perturbation_covariates` columns | `perturbation` → the extra `cols` on both nodes |
| `control_key` / which cells are control | `control_values` → which combos land in the `ctrl` node |
| control = same group as target | `Bind("pert", "ctrl", common=context)` |
| one perturbation per batch, weighted | root `pert` node weights + `ScheduledClassSampler` |
| `batch_size` | `SamplerConfig.batch_size` |
| `perturbation_covariate_reps` (embeddings) | `condition_fn(leaf) -> embedding` |

```python
from dagloader import Bind, DAGLoader, Node, SamplerConfig, Scheme, uniform

cols = ("cell_line", "drug")  # context (split_covariates) + perturbation columns
combos = [tuple(r) for r in adata_or_collection.obs[list(cols)].drop_duplicates().to_numpy()]
pert = [c for c in combos if c[1] != "control"]  # control_values={"drug": "control"}
ctrl = [c for c in combos if c[1] == "control"]
scheme = Scheme(
    sources={"data": adata_or_collection},
    nodes={
        "pert": Node("data", cols, "X", uniform(pert)),  # root = perturbed; key="X" is sample_rep
        "ctrl": Node("data", cols, "X", uniform(ctrl)),  # matched control child
    },
    root="pert",
    binds=(Bind("pert", "ctrl", common=("cell_line",)),),  # match on context
    seed=0,
)
loader = DAGLoader(
    scheme,
    SamplerConfig(batch_size=256, chunk_size=1, preload_nchunks=256),  # all read params explicit (no hidden defaults)
    condition_fn=lambda leaf: DRUG_EMB[leaf[-1]],
)
batch = next(loader)  # {"target", "source", "condition"} — source is a matched-cell-line control
```

**How cellflow uses it.** `cellflow.data._annbatch.build_annbatch_training` assembles exactly this
scheme (plus the `condition_fn`) from a `prepare_data` covariate spec — the collection/streaming path is
**additive** to cellflow's in-memory `adata` API. The condition encoding (categorical embeddings,
combinations, pooling) stays in the model; the loader only needs a `condition_fn` that turns a leaf
tuple into the per-condition embedding.

---

## 4. The sc-flow-tools cases

sc-flow-tools already owns grouping (`HierarchicalIndexer`), matching (`control_values_dict`,
`matched_keys`), representations, and an in-memory multi-node `TrainSampler`. The one thing it lacks
is **out-of-core streaming**. `DAGLoader` is exactly that streaming sampler: its scheme
reproduces what a `DataManager` config already describes, reading cells from a `DatasetCollection`
instead of holding them in memory.

| sc-flow-tools concept | dagloader |
|---|---|
| `HierarchicalIndexer` grouping (obs Categorical codes) | `Node.cols` → leaves (obs-only, computed at build) |
| `control_values_dict` (which values are control) | the `ctrl` node's weighted combos (zero-weight = excluded) |
| default same-context coupling (one-to-many) | `Bind(..., common=context)` |
| `matched_keys` (explicit fixed source→target pairing) | a `Bind` whose `common` covers the pairing columns, so parent leaf → child leaf is forced |
| `sample_rep` in PCA space (an obsm key) | `Node.key = "obsm/X_pca"` |
| `*_reps` (uns embedding tables) | `condition_fn(leaf)` (concatenate per-covariate embeddings) |
| weighted / multi-node `TrainSampler` | root weights + a rooted tree of binds |

**Direct scheme construction** (the sc-flow-tools style — assemble nodes from a `DataManager` config):

```python
from dagloader import Bind, DAGLoader, Node, SamplerConfig, Scheme, uniform

# grouping columns: context (cell_line) + perturbation (drug); state read from a PCA obsm key
cols = ("cell_line", "drug")
scheme = Scheme(
    sources={"data": collection},
    nodes={
        "target":  Node("data", cols, key="obsm/X_pca", weights=uniform(perturbed_combos)),
        "control": Node("data", cols, key="obsm/X_pca", weights=uniform(control_combos)),
    },
    root="target",
    binds=(Bind("target", "control", common=("cell_line",)),),  # control_values_dict + same-context
    seed=0,
)
loader = DAGLoader(
    scheme,
    SamplerConfig(batch_size=256, chunk_size=1, preload_nchunks=256),  # per-node: {"target": SamplerConfig(batch_size=256, chunk_size=32, preload_nchunks=8), "control": SamplerConfig(batch_size=256, chunk_size=1, preload_nchunks=256)}
    condition_fn=lambda leaf: concat(cell_emb[leaf[0]], drug_emb[leaf[1]]),  # *_reps combination
)
batch = next(loader)  # {"target", "source", "condition"}
```

**How sc-flow-tools uses it.** Per the shared-data-layer plan (`docs/shared-data-layer-status.md`),
sc-flow-tools hosts the shared data layer and CellFlow2 depends on it. In that plan the tree nodes
carry **row indices** (obs-only), and the annbatch loader (this package's `DAGLoader`) streams
the sampled indices from a `DatasetCollection`. The `TrainSampler`'s node dispatch becomes a scheme:
each node → a `Node`, the matched-subpopulation tree → the binds, `control_values_dict` → weights,
`matched_keys` → binds with pairing columns in `common`.

### What is and isn't in this prototype

- **In:** single-level bind (root + one bound child) matched on shared columns; a single streamed rep
  per node (`X` / obsm / layer); categorical condition via `condition_fn`; in-memory and out-of-core
  sources; weighted per-condition sampling; per-node `SamplerConfig`; reproducible per-node RNG.
- **Documented mapping, not yet wired:** multi-child trees emit a single `source` today (the batch
  contract would grow a `sources[child_name]` map); **per-cell continuous covariates / Gromov
  `state_lin`+`state_quad`** (multiple aligned reps for the same streamed rows) would need annbatch to
  return several reps per batch, or a second aligned read — deliberately left out rather than
  reintroduce a bespoke row-gather sampler.

---

## 5. Files

| file | contents |
|---|---|
| `_schema.py` | `Node`, `Bind`, `Scheme`, `SamplerConfig`, `Weights`, `uniform` / `frequency` / `inverse_frequency` — structural, data-free |
| `_scheduled_sampler.py` | `ScheduledClassSampler` (reuses annbatch `_iter_requests` via an rng wrapper; the upstream candidate) |
| `_io.py` | container-agnostic obs / leaf-code / rep-backing helpers (AnnData or DatasetCollection; X / obsm / layers) |
| `_loader.py` | `DAGLoader` (config resolution, per-pass scheduling, bind derivation, batch assembly) |

Tests live in `tests/dagloader/`, organized by the cases above:

| test file | case |
|---|---|
| `test_scheduled_sampler.py` | `ScheduledClassSampler`: schedule adherence, chunk coherence, `schedule=None` ≡ `ClassSampler`, validation |
| `test_sampling_schemes.py` | weights as the selection; `uniform` / `frequency` / `inverse_frequency` empirical distributions; per-node independent RNG; reproducibility; `Node` / `Scheme` / `SamplerConfig` validation |
| `test_cellflow_case.py` | `perturbation_scheme`: matched control↔perturbed batches, condition, obsm rep, on-disk collection, `chunk_size>1` |
| `test_scflow_cases.py` | direct scheme construction: multi-covariate condition, explicit matched pairing via `common`, obsm rep as state, out-of-core |
| `test_train.py` | end-to-end: scheme → loader → a real flow-matching training step (loss decreases), in-memory + DatasetCollection + obsm + `chunk_size>1` |
