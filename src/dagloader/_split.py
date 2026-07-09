"""Split a :class:`~dagloader.Scheme`'s target combinations into named splits — weights-only.

A "split" needs no new data structure here. Because a leaf's **weight is the selection** (weight 0 /
absent ⇒ not sampled), holding out conditions is just restricting the root node's weights to a subset
of its combinations. :func:`split_scheme` partitions the root's positive-weight combinations — grouped
by ``split_by`` (a subset of the root's ``cols``) — into named splits and returns one :class:`Scheme`
per split, identical to the input except the root's ``weights`` are restricted. Bound children
(controls / sources) are **carried through unchanged**: a matched control must stay available in every
split, since each batch's control is drawn from whatever context the (split) target carries.

This mirrors CellFlow2's ``GroupedDistributionSplitter`` — split whole *combinations*, not cells (an
out-of-distribution generalization split) — but expressed over the ``Scheme``, so it touches no cells,
no ``DatasetCollection`` and no loader: it only rewrites a weights dict.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from dagloader._schema import SamplerConfig, Scheme

__all__ = ["resolve_split_configs", "split_assignment", "split_scheme"]

_DEFAULT_RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}


def _allocate(n: int, ratios: Mapping[str, float], names: Sequence[str]) -> dict[str, int]:
    """Largest-remainder apportionment of ``n`` groups to splits by ratio (the sizes sum to ``n``)."""
    exact = {name: ratios[name] * n for name in names}
    sizes = {name: int(exact[name]) for name in names}  # floor
    remainder = n - sum(sizes.values())
    order = sorted(names, key=lambda nm: exact[nm] - sizes[nm], reverse=True)  # largest fractional part first
    for nm in order[:remainder]:
        sizes[nm] += 1
    return sizes


def split_scheme(
    scheme: Scheme,
    *,
    split_by: Sequence[str],
    ratios: Mapping[str, float] | None = None,
    force_training_values: Mapping[str, object] | None = None,
    random_state: int = 42,
) -> dict[str, Scheme]:
    """Partition the root (target) combinations of ``scheme`` into named splits.

    Splits are over whole *combinations* of ``split_by`` (all combos sharing the same ``split_by``
    values land in the same split — CellFlow2's ``drop_duplicates(subset=split_by)`` + merge-back), so
    this is an out-of-distribution split that holds out entire conditions rather than random cells.

    Parameters
    ----------
    scheme
        The prepared scheme; its ``root`` node's positive-weight combinations are the split universe.
    split_by
        Columns whose unique combinations are partitioned across splits — a subset of the root's
        ``cols``.
    ratios
        ``{split_name: fraction}`` summing to 1.0 (all > 0). Defaults to
        ``{"train": 0.6, "val": 0.2, "test": 0.2}``. Insertion order defines the split order; the first
        split is the "training" split that :paramref:`force_training_values` forces into.
    force_training_values
        ``{column: value}`` (keys ⊆ ``split_by``): any group matching *any* of these (column, value)
        pairs is forced into the first split, regardless of the shuffle.
    random_state
        Seed for the group shuffle (reproducible).

    Returns
    -------
    ``{split_name: Scheme}`` — each a copy of ``scheme`` with the root node's ``weights`` restricted to
    that split's combinations; every other node (bound children / controls) is carried through
    unchanged.
    """
    ratios = dict(_DEFAULT_RATIOS if ratios is None else ratios)
    split_by = tuple(split_by)
    force = dict(force_training_values or {})

    root = scheme.nodes[scheme.root]
    cols = root.cols

    # --- validation -----------------------------------------------------------------------------
    if not split_by:
        raise ValueError("split_by must be non-empty.")
    missing = [c for c in split_by if c not in cols]
    if missing:
        raise ValueError(f"split_by columns {missing} are not in the root node's cols {cols}.")
    if not ratios:
        raise ValueError("ratios must be a non-empty {split_name: fraction} mapping.")
    if any(r <= 0 for r in ratios.values()):
        raise ValueError(f"ratios must all be > 0; got {ratios}.")
    if not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError(f"ratios must sum to 1.0; got {ratios} (sum={sum(ratios.values())}).")
    bad_force = [k for k in force if k not in split_by]
    if bad_force:
        raise ValueError(f"force_training_values keys {bad_force} must be a subset of split_by {split_by}.")

    names = list(ratios)  # deterministic split order (insertion order)
    train_name = names[0]  # forced groups land here

    # --- the split universe: the root's positive-weight combinations (weight = the selection) -----
    combos = [combo for combo, w in root.weights.items() if w > 0]
    if not combos:
        raise ValueError("root node has no positive-weight combinations to split.")

    proj_idx = [cols.index(c) for c in split_by]

    def project(combo: tuple) -> tuple:  # a combination → its split_by group
        return tuple(combo[i] for i in proj_idx)

    groups = sorted({project(c) for c in combos}, key=lambda g: tuple(map(str, g)))  # deterministic

    def is_forced(group: tuple) -> bool:  # OR across keys, like CellFlow2._contains_value
        gd = dict(zip(split_by, group, strict=True))
        return any(gd.get(k) == v for k, v in force.items())

    forced = [g for g in groups if is_forced(g)]
    free = [g for g in groups if not is_forced(g)]

    rng = np.random.default_rng(random_state)
    free = [free[i] for i in rng.permutation(len(free))]

    sizes = _allocate(len(free), ratios, names)
    # a split that must hold groups but got 0 is an error (mirrors CellFlow2). The train split is
    # exempt when it will still receive forced groups.
    empty = [n for n in names if sizes[n] == 0 and not (n == train_name and forced)]
    if empty:
        raise ValueError(
            f"splits {empty} received 0 of {len(groups)} `split_by` group(s); increase the number of "
            f"groups or adjust ratios {ratios}."
        )

    split_of: dict[tuple, str] = {}
    pos = 0
    for name in names:
        for g in free[pos : pos + sizes[name]]:
            split_of[g] = name
        pos += sizes[name]
    for g in forced:  # forced groups always go to the first (training) split
        split_of[g] = train_name

    # --- build one Scheme per split: restrict the root weights, carry everything else ------------
    out: dict[str, Scheme] = {}
    for name in names:
        kept = {combo: root.weights[combo] for combo in combos if split_of[project(combo)] == name}
        new_root = replace(root, weights=kept)
        out[name] = replace(scheme, nodes={**scheme.nodes, scheme.root: new_root})
    return out


def split_assignment(splits: Mapping[str, Scheme]) -> pd.DataFrame:
    """A tidy ``{*root cols, "split"}`` table of which target combination went to which split.

    Reconstructed from the split schemes' root weights (positive-weight combinations); useful for
    inspecting a :func:`split_scheme` result without touching any cells.
    """
    if not splits:
        raise ValueError("splits mapping is empty.")
    any_scheme = next(iter(splits.values()))
    cols = any_scheme.nodes[any_scheme.root].cols
    rows = [(*combo, name) for name, sch in splits.items() for combo, w in sch.nodes[sch.root].weights.items() if w > 0]
    df = pd.DataFrame(rows, columns=[*cols, "split"])
    return df.sort_values(["split", *cols]).reset_index(drop=True)


def resolve_split_configs(
    config: SamplerConfig | Mapping[str, SamplerConfig],
    split_names: Sequence[str],
) -> dict[str, SamplerConfig]:
    """Resolve a read-parameter spec into exactly one :class:`SamplerConfig` per split.

    ``config`` is either

    * **a single** :class:`SamplerConfig` — applied to every split; or
    * **a per-split mapping** ``{split_name: SamplerConfig}`` — in which case **every** split in
      ``split_names`` must be present (no partial specs, no unknown split names).

    Returns ``{split_name: SamplerConfig}`` for exactly ``split_names``.
    """
    names = list(split_names)
    if not names:
        raise ValueError("split_names must be non-empty.")

    if isinstance(config, SamplerConfig):  # one config → all splits
        return dict.fromkeys(names, config)

    if isinstance(config, Mapping):  # per-split mapping: every split specified, no extras
        missing = [n for n in names if n not in config]
        if missing:
            raise ValueError(
                f"sampler_config is per-split but is missing config(s) for split(s) {missing}; specify all of {names}."
            )
        extra = [k for k in config if k not in names]
        if extra:
            raise ValueError(f"sampler_config has config(s) for unknown split(s) {extra}; splits are {names}.")
        bad = [k for k, v in config.items() if not isinstance(v, SamplerConfig)]
        if bad:
            raise ValueError(
                f"sampler_config values must be SamplerConfig instances; got a non-SamplerConfig for {bad}."
            )
        return {name: config[name] for name in names}

    raise ValueError(
        f"sampler_config must be a SamplerConfig or a {{split: SamplerConfig}} mapping; got {type(config).__name__}."
    )
