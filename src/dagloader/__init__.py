r"""Declarative, index-free class-mapping sampler over annbatch.

A :class:`Scheme` is a rooted tree of :class:`Node`\\s over named cell sources; each node partitions
its source's cells into leaves (unique column-combinations) with a per-combination weight mapping.
:class:`DAGLoader` streams matched ``{source, target, condition}`` batches — one condition per
batch — where the root's per-batch category is drawn from its weights (annbatch ``ClassSampler``) and
each bound child replays that schedule via an annbatch ``BoundClassSampler`` (matched on the bind's
shared columns) so the loaders zip batch-for-batch. No row indices are exposed: the scheme is
columns / keys / weights.

See ``README.md`` (next to this file) for the model, the sampling schemes, and the mapping to
cellflow and sc-flow-tools use-cases.
"""

from dagloader._eval_loader import DAGEvalLoader
from dagloader._loader import DAGLoader
from dagloader._schema import (
    Bind,
    Container,
    Node,
    SamplerConfig,
    Scheme,
    Weights,
    frequency,
    inverse_frequency,
    uniform,
)
from dagloader._split import resolve_split_configs, split_assignment, split_scheme

__all__ = [
    "Bind",
    "Container",
    "DAGEvalLoader",
    "DAGLoader",
    "Node",
    "SamplerConfig",
    "Scheme",
    "Weights",
    "frequency",
    "inverse_frequency",
    "resolve_split_configs",
    "split_assignment",
    "split_scheme",
    "uniform",
]
