import math
from typing import Any

import numpy as np
import os
import threading
from concurrent.futures import ThreadPoolExecutor, Future

from scaleflow.data._data import (
    GroupedDistribution,
)

from contextlib import nullcontext
from abc import ABC, abstractmethod

__all__ = [
    "ReservoirSampler",
    "SamplerABC",
]



class SamplerABC(ABC):
    @abstractmethod
    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        pass

    @abstractmethod
    def init_sampler(self, rng: np.random.Generator) -> None:
        pass

class ReservoirSampler:
    """Data sampler with gradual pool replacement using reservoir sampling.

    This approach replaces pool elements one by one rather than refreshing
    the entire pool, providing better cache locality while maintaining
    reasonable randomness.

    Parameters
    ----------
    data
        The training data.
    batch_size
        The batch size.
    pool_fraction
        Fraction of source distributions to cache (0 < pool_fraction <= 1).
        If 1.0, all sources are cached and no replacement is performed.
    replacement_prob
        Probability of replacing a pool element after each sample.
        Lower values = longer cache retention, less randomness.
        Higher values = faster cache turnover, more randomness.
    """

    def __init__(
        self,
        data: GroupedDistribution,
        batch_size: int = 1024,
        pool_fraction: float | None = None,
        replacement_prob: float | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.n_source_dists = len(data.data.src_data)
        self.n_target_dists = len(data.data.tgt_data)
        self._data = data
        self._cache_all = False

        if pool_fraction is None and replacement_prob is None:
            self._cache_all = True
        else:
            if pool_fraction is None:
                raise ValueError("pool_fraction must be provided if replacement_prob is provided.")
            if replacement_prob is None:
                raise ValueError("replacement_prob must be provided if pool_fraction is provided.")
        # Compute pool size from fraction
        if not (0 < pool_fraction <= 1):
            raise ValueError("pool_fraction must be in (0, 1].")
        self._pool_fraction = pool_fraction
        self._pool_size = math.ceil(pool_fraction * self.n_source_dists)
        self._replacement_prob = replacement_prob
        self._pool_usage_count = np.zeros(self.n_source_dists, dtype=int)
        self._initialized = False
        self._src_idx_pool = None

        if pool_fraction == 1.0:
            self._cache_all = True

        self._lock = nullcontext() if self._cache_all else threading.RLock()
        self._executor = None
        self._pending_replacements = {}
        if not self._cache_all:
            self._executor = ThreadPoolExecutor(max_workers=2) # TODO: avoid magic numbers
            self._pending_replacements: dict[int, dict[str, Any]] = {}

    

    def init_sampler(self, rng) -> None:
        if self._initialized:
            raise ValueError("Sampler already initialized. Call init_sampler() only once.")
        self._init_src_idx_pool(rng)
        self._init_cache_pool_elements()
        self._initialized = True
        return None

    def _init_src_idx_pool(self, rng) -> None:
        if self._cache_all:
            self._src_idx_pool = np.arange(self.n_source_dists)
        else:
            self._src_idx_pool = rng.choice(self.n_source_dists, size=self._pool_size, replace=False)
        return None


    def sample(self, rng) -> dict[str, Any]:
        """Sample a batch for gene expression (flow matching) task.

        Parameters
        ----------
        rng
            Random number generator

        Returns
        -------
        Dictionary with source cells, target cells, condition, and task type
        """
        source_dist_idx = self._sample_source_dist_idx(rng)
        target_dist_idx = self._sample_target_dist_idx(rng, source_dist_idx)
        print(f"sampled source dist idx: {source_dist_idx} and target dist idx: {target_dist_idx}")
        source_batch = self._sample_source_cells(rng, source_dist_idx)
        print(f"sampled source batch: {source_batch.shape}")
        target_batch = self._sample_target_cells(rng, source_dist_idx, target_dist_idx)
        print(f"sampled target batch: {target_batch.shape}")
        res = {
            "src_cell_data": source_batch,
            "tgt_cell_data": target_batch
        }
        res["condition"] = self._data.data.conditions[target_dist_idx]
        return res


    def _load_targets_parallel(self, tgt_indices):
        """Load multiple target distributions in parallel."""
        def _load_tgt(j: int):
            return j, self._data.data.tgt_data[j][...]
        
        max_workers = min(32, (os.cpu_count() or 4))  # TODO: avoid magic numbers
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_load_tgt, tgt_indices))
        return {j: arr for j, arr in results}

    def _init_cache_pool_elements(self) -> None:
        with self._lock:
            self._cached_srcs = {i: self._data.data.src_data[i][...] for i in self._src_idx_pool}
        
        tgt_indices = sorted({int(j) for i in self._src_idx_pool for j in self._data.data.src_to_tgt_dist_map[i]})
        
        with self._lock:
            self._cached_tgts = self._load_targets_parallel(tgt_indices)
        
        return None



    def _sample_target_dist_idx(self, rng, source_dist_idx: int) -> int:
        """Sample a target distribution index given the source distribution index."""
        return rng.choice(self._data.data.src_to_tgt_dist_map[source_dist_idx])





    def _sample_source_dist_idx(self, rng) -> int:
        """Sample a source distribution index with gradual pool replacement."""
        if not self._initialized:
            raise ValueError("Sampler not initialized. Call init_sampler() first.")

        return (self._sample_source_dist_idx_in_pool(rng) if not self._cache_all else \
            self._sample_source_dist_idx_in_memory(rng))


    def _sample_source_dist_idx_in_memory(self, rng) -> int:
        source_idx = rng.choice(sorted(self._cached_srcs.keys()))
        self._pool_usage_count[source_idx] += 1
        return source_idx

    def _sample_source_dist_idx_in_pool(self, rng) -> int:
        self._apply_ready_replacements()
        # Sample from current pool
        with self._lock:
            source_idx = rng.choice(sorted(self._cached_srcs.keys()))

        # Increment usage count for monitoring
        self._pool_usage_count[source_idx] += 1

        # Gradually replace elements based on replacement probability (schedule only)
        if rng.random() < self._replacement_prob:
            self._schedule_replacement(rng)
        
        return source_idx

    def _schedule_replacement(self, rng):
        if self._cache_all:
            return  # No replacement if everything is cached
        # weights same as previous logic
        most_used_weight = (self._pool_usage_count == self._pool_usage_count.max()).astype(float)
        if most_used_weight.sum() == 0:
            return
        most_used_weight /= most_used_weight.sum()
        replaced_pool_idx = rng.choice(self.n_source_dists, p=most_used_weight)

        with self._lock:
            pool_set = set(self._src_idx_pool.tolist())
            if replaced_pool_idx not in pool_set:
                return
            in_pool_idx = int(np.where(self._src_idx_pool == replaced_pool_idx)[0][0])

            # If there's already a pending replacement for this pool slot, skip
            if in_pool_idx in self._pending_replacements:
                return

            least_used_weight = (self._pool_usage_count == self._pool_usage_count.min()).astype(float)
            if least_used_weight.sum() == 0:
                return
            least_used_weight /= least_used_weight.sum()
            new_pool_idx = int(rng.choice(self.n_source_dists, p=least_used_weight))

            # Kick off background load for new indices
            fut: Future = self._executor.submit(self._load_new_cache, new_pool_idx)
            self._pending_replacements[in_pool_idx] = {
                "old": replaced_pool_idx,
                "new": new_pool_idx,
                "future": fut,
            }
            print(f"scheduled replacement of {replaced_pool_idx} with {new_pool_idx} (slot {in_pool_idx})")

    def _load_targets_parallel(self, tgt_indices):
        """Load multiple target distributions in parallel."""
        def _load_tgt(j: int):
            return j, self._data.data.tgt_data[j][...]
        
        max_workers = min(32, (os.cpu_count() or 4))  # TODO: avoid magic numbers
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_load_tgt, tgt_indices))
        return {j: arr for j, arr in results}


    def _apply_ready_replacements(self):
        if self._cache_all:
            return  # No replacement if everything is cached
        """Apply any finished background loads; non-blocking."""
        to_apply: list[int] = []
        with self._lock:
            for slot, info in self._pending_replacements.items():
                fut: Future = info["future"]
                if fut.done() and not fut.cancelled():
                    to_apply.append(slot)

        for slot in to_apply:
            with self._lock:
                info = self._pending_replacements.pop(slot, None)
                if info is None:
                    continue
                old_idx = int(info["old"])
                new_idx = int(info["new"])
                fut: Future = info["future"]
                try:
                    prepared = fut.result(timeout=0)  # already done
                except Exception as e:
                    print(f"background load failed for {new_idx}: {e}")
                    continue

                # Swap pool index
                self._src_idx_pool[slot] = new_idx

                # Add new entries first
                self._cached_srcs[new_idx] = prepared["src"]
                for k, arr in prepared["tgts"].items():
                    self._cached_tgts[k] = arr

                # Remove old entries
                if old_idx in self._cached_srcs:
                    del self._cached_srcs[old_idx]
                for k in self._data.data.src_to_tgt_dist_map[old_idx]:
                    if k in self._cached_tgts:
                        del self._cached_tgts[k]

                print(f"applied replacement: {old_idx} -> {new_idx} (slot {slot})")

    def _load_new_cache(self, src_idx: int) -> dict[str, Any]:
        """Load new src and corresponding tgt arrays in the background."""
        src_arr = self._data.data.src_data[src_idx][...]
        tgt_dict = {k: self._data.data.tgt_data[k][...] for k in self._data.data.src_to_tgt_dist_map[src_idx]}
        return {"src": src_arr, "tgts": tgt_dict}


    def _sample_source_cells(self, rng, source_dist_idx: int) -> np.ndarray:
        with self._lock:
            arr = self._cached_srcs[source_dist_idx]
        idxs = rng.choice(arr.shape[0], size=self.batch_size, replace=True)
        return arr[idxs]

    def _sample_target_cells(self, rng, source_dist_idx: int, target_dist_idx: int) -> np.ndarray:
        del source_dist_idx  # unused
        with self._lock:
            arr = self._cached_tgts[target_dist_idx]
        idxs = rng.choice(arr.shape[0], size=self.batch_size, replace=True)
        return arr[idxs]

