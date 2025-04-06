import abc
from typing import Any, Literal, Dict, List, Tuple, Callable

import numpy as np

from cfp.data._cpu_dataloader import (
    CpuTrainSampler,

)
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from torch.utils import data

from dataclasses import dataclass





@dataclass	
class IterativeSample:
    sampler: CpuTrainSampler
    num_iterations: int

    def __getitem__(self, ix: int):
        del ix
        batch = self.sampler.sample()
        batch = jax.tree_util.tree_map(jnp.asarray, batch, device=jax.devices('cpu')[0])
        return batch


    def __len__(self) -> int:
        return self.num_iterations
    
    def __iter__(self):
        for i in range(self.num_iterations):
            yield self[i]
    


