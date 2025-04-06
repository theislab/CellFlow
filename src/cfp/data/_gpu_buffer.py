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

def jax_collate(batch):  # noqa: D103
    batch = data.default_collate(batch)
    batch = jax.tree_util.tree_map(jnp.asarray, batch)
    return batch



@dataclass	
class TorchDataset:
    sampler: CpuTrainSampler
    num_iterations: int

    def __getitem__(self, ix: int):
        del ix
        return self.sampler.sample()

    def __len__(self) -> int:
        return self.num_iterations
    


def jax_dataloader(  # noqa: D103
    ds: TorchDataset,
    num_workers: int = 0,
    pin_memory: bool = False,
    **kwargs,
):
    return data.DataLoader(
        ds,
        collate_fn=jax_collate,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )

