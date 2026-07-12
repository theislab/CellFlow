"""``ScheduledClassSampler`` — an annbatch ``ClassSampler`` whose per-batch category can be *supplied*.

``set_schedule(codes)`` fixes the category (leaf code) of each batch for the next pass; with
``schedule=None`` the sampler is exactly ``ClassSampler`` (draws ∝ ``class_weights``). That ownership
is what lets :class:`~dagloader.DAGLoader` zip a parent and its bound children
batch-for-batch: the root schedule is drawn from the root's weights, each child's schedule is
*derived* from the parent's.

Rather than copy annbatch's ``_iter_requests`` (all the chunk/RLE/slice math), this reuses it wholesale
and only redirects the one draw that picks each group's class. ``ClassSampler._iter_requests`` makes a
single *weighted* ``self._rng.choice(n_classes, size=n_groups, p=...)`` to choose group classes; we run
that method unchanged but with ``self._rng`` temporarily wrapped so that one call returns the scheduled
positions instead. Everything else (run selection, slice starts, shuffling) uses the real generator, so
a scheduled pass is bit-identical to what annbatch would produce for those same group classes.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from annbatch.samplers import ClassSampler

__all__ = ["ScheduledClassSampler"]


class _ScheduleRng:
    """Wraps a ``numpy.random.Generator``; the first *weighted* ``choice`` returns supplied positions.

    ``ClassSampler._iter_requests`` draws group classes with the only ``choice(..., p=<weights>)`` call
    in that method; we intercept it (identified by ``p is not None``) and return the pre-mapped
    schedule positions, validating the group count. Every other attribute/method — ``integers``,
    ``shuffle``, any unweighted ``choice`` — delegates to the real generator, so RNG state and
    reproducibility are unchanged.
    """

    def __init__(self, rng: np.random.Generator, positions: np.ndarray) -> None:
        self._rng = rng
        self._positions = positions
        self._fired = False

    def choice(self, a, size=None, replace=True, p=None):
        if p is not None and not self._fired:  # the per-group class draw in _iter_requests
            self._fired = True
            if size != len(self._positions):
                raise ValueError(
                    f"schedule length {len(self._positions)} != n_groups {size}; one category per batch "
                    "requires chunk_size to divide batch_size and len(schedule) == num_samples // batch_size."
                )
            return self._positions
        return self._rng.choice(a, size=size, replace=replace, p=p)

    def __getattr__(self, name):  # integers / shuffle / permutation / ... → real generator
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)  # don't delegate dunders — avoids recursion during (un)pickling / copy
        return getattr(self._rng, name)

    def __reduce__(self) -> tuple[object, ...]:
        # A transient per-pass wrapper: pickle it AS the underlying real Generator so a checkpointed
        # DAGLoader restores a clean, unwrapped RNG (the wrapper is rebuilt each pass in `_sample`).
        return self._rng.__reduce__()


class ScheduledClassSampler(ClassSampler):
    """``ClassSampler`` whose per-batch category sequence can be supplied via :meth:`set_schedule`.

    The schedule is one leaf code per batch (``len == num_samples // batch_size``); it is valid only
    when ``chunk_size`` divides ``batch_size`` (exactly one category per batch).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._schedule: np.ndarray | None = None

    def set_schedule(self, category_codes: np.ndarray | None) -> None:
        """Set (or clear) the per-batch category codes used for the next ``iter(loader)`` pass."""
        self._schedule = None if category_codes is None else np.asarray(category_codes, dtype=np.int64)

    def _schedule_positions(self) -> np.ndarray:
        """Map the scheduled leaf codes to positions into the sampleable-class table (as annbatch expects)."""
        info = self._per_class_sampling_info
        code_to_pos = {int(c): i for i, c in enumerate(info.index.to_numpy())}
        try:
            return np.array([code_to_pos[int(c)] for c in self._schedule], dtype=np.int64)
        except KeyError as e:
            raise ValueError(f"schedule contains a non-sampleable (zero-weight) category code {e}.") from e

    def _sample(self, n_obs: int) -> Iterator[dict]:
        if self._schedule is None:  # standalone → identical to ClassSampler
            yield from super()._sample(n_obs)
            return
        self._ensure_runs(n_obs)  # build _per_class_sampling_info before mapping codes → positions
        positions = self._schedule_positions()
        real_rng = self._rng
        self._rng = _ScheduleRng(real_rng, positions)  # redirect only the group-class draw
        try:
            yield from super()._sample(n_obs)
        finally:
            self._rng = real_rng
