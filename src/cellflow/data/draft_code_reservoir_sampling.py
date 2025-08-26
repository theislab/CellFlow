@dataclass
class CombinedTrainSampler:
    """
    Combined training sampler that iterates over multiple samplers.

    Need to call set_rng(rng) before using the sampler.

    Args:
        samplers: List of training samplers.
        weights: Weights for the samplers.
        dataset_names: Names for the samplers.
        rng: Random number generator.
    """

    samplers: list[TrainSampler]
    weights: np.ndarray | None = None
    dataset_names: list[str] | None = None
    rng: np.random.Generator | None = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = np.ones(len(self.samplers))
        self.weights = np.asarray(self.weights)
        assert len(self.weights) == len(self.samplers)
        self.weights = self.weights / self.weights.sum()

    def set_rng(self, rng: np.random.Generator):
        self.rng = rng

    def sample(self, *args, **kwargs) -> dict[str, Any]:
        del args, kwargs
        if self.rng is None:
            raise ValueError("Please call set_rng() before using the sampler.")
        dataset_idx = self.rng.choice(len(self.samplers), p=self.weights)
        res = self.samplers[dataset_idx].sample(self.rng)
        if self.dataset_names is not None:
            res["dataset_name"] = self.dataset_names[dataset_idx]
        return res
    
    def get_pool_stats(self) -> dict[str, dict]:
        """Get pool statistics for all samplers that support it."""
        stats = {}
        for i, sampler in enumerate(self.samplers):
            if hasattr(sampler, 'get_pool_stats'):
                name = self.dataset_names[i] if self.dataset_names else f"sampler_{i}"
                stats[name] = sampler.get_pool_stats()
        return stats

class TrainSamplerWithGradualPool(TrainSampler):
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
    pool_size
        The size of the pool of source distribution indices.
    replacement_prob
        Probability of replacing a pool element after each sample.
        Lower values = longer cache retention, less randomness.
        Higher values = faster cache turnover, more randomness.
    replace_in_pool
        Whether to allow replacement when sampling from the pool.
    """
    
    def __init__(
        self,
        data: TrainingData | ZarrTrainingData,
        batch_size: int = 1024,
        pool_size: int = 100,
        replacement_prob: float = 0.01,
        replace_in_pool: bool = True,
    ):
        super().__init__(data, batch_size)
        self._pool_size = pool_size
        self._replacement_prob = replacement_prob
        self._replace_in_pool = replace_in_pool
        self._src_idx_pool = None
        self._pool_usage_count = None
        self._all_src_idx_pool = set(range(self.n_source_dists))
        
    def _init_pool(self, rng):
        """Initialize the pool with random source distribution indices."""
        self._src_idx_pool = rng.choice(
            self.n_source_dists, 
            size=self._pool_size, 
            replace=False
        )
        self._pool_usage_count = np.zeros(self._pool_size, dtype=int)
        
    def _sample_source_dist_idx(self, rng) -> int:
        """Sample a source distribution index with gradual pool replacement."""
        if self._src_idx_pool is None:
            self._init_pool(rng)
            
        # Sample from current pool
        pool_idx = rng.choice(self._pool_size, replace=self._replace_in_pool)
        source_idx = self._src_idx_pool[pool_idx]
        
        # Increment usage count for monitoring
        self._pool_usage_count[pool_idx] += 1
        
        # Gradually replace elements based on replacement probability
        if rng.random() < self._replacement_prob:
            self._replace_pool_element(rng, pool_idx)
            
        return source_idx
    
    def _replace_pool_element(self, rng, pool_idx):
        """Replace a single pool element with a new one."""
        # Get all indices not currently in the pool
        available_indices = list(self._all_src_idx_pool - set(self._src_idx_pool))
        
        if available_indices:
            # Choose new element (could be weighted by usage count)
            new_idx = rng.choice(available_indices)
            self._src_idx_pool[pool_idx] = new_idx
            self._pool_usage_count[pool_idx] = 0
    
    def get_pool_stats(self) -> dict:
        """Get statistics about the current pool state."""
        if self._src_idx_pool is None:
            return {"pool_size": 0, "avg_usage": 0, "unique_sources": 0}
        
        return {
            "pool_size": self._pool_size,
            "avg_usage": float(np.mean(self._pool_usage_count)),
            "unique_sources": len(set(self._src_idx_pool)),
            "pool_elements": self._src_idx_pool.copy(),
            "usage_counts": self._pool_usage_count.copy()
        }
# Example usage function
def create_combined_sampler_with_pools(
    data_paths: list[str],
    batch_size: int = 1024,
    pool_size: int = 100,
    replacement_prob: float = 0.01,
    weights: np.ndarray | None = None,
    dataset_names: list[str] | None = None,
) -> CombinedTrainSampler:
    """Create a combined sampler with gradual pool replacement for each dataset.
    
    This is useful when you have multiple large datasets and want to maintain
    memory-efficient sampling with controlled randomness.
    """
    from cellflow.data import ZarrTrainingData
    
    # Create samplers with pools for each dataset
    samplers = []
    for data_path in data_paths:
        data = ZarrTrainingData.read_zarr(data_path)
        sampler = TrainSamplerWithGradualPool(
            data=data,
            batch_size=batch_size,
            pool_size=pool_size,
            replacement_prob=replacement_prob
        )
        samplers.append(sampler)
    
    # Create combined sampler
    combined_sampler = CombinedTrainSampler(
        samplers=samplers,
        weights=weights,
        dataset_names=dataset_names
    )
    
    return combined_sampler

# Example usage in training loop
def training_example():
    # Create sampler
    sampler = create_combined_sampler_with_pools(
        data_paths=["dataset1.zarr", "dataset2.zarr"],
        batch_size=1024,
        pool_size=200,  # Keep 200 source indices in memory per dataset
        replacement_prob=0.005,  # Replace ~0.5% of pool elements per sample
        weights=[0.7, 0.3],  # 70% from dataset1, 30% from dataset2
        dataset_names=["pbmc", "zebrafish"]
    )
    
    # Set random generator
    rng = np.random.default_rng(42)
    sampler.set_rng(rng)
    
    # Training loop
    for epoch in range(100):
        for step in range(1000):
            batch = sampler.sample()
            
            # Every 100 steps, check pool statistics
            if step % 100 == 0:
                pool_stats = sampler.get_pool_stats()
                print(f"Epoch {epoch}, Step {step}:")
                for dataset_name, stats in pool_stats.items():
                    print(f"  {dataset_name}: {stats['unique_sources']}/{stats['pool_size']} unique sources, avg usage: {stats['avg_usage']:.1f}")
            
            # ... training code here ...