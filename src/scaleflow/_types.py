import os
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

# TODO(michalk8): polish

try:
    from numpy.typing import NDArray

    ArrayLike = NDArray[np.float64]
except (ImportError, TypeError):
    ArrayLike = np.ndarray  # type: ignore[misc]
    DTypeLike = np.dtype

ComputationCallback_t = Callable[[dict[str, ArrayLike], dict[str, ArrayLike]], dict[str, Any]]
LoggingCallback_t = Callable[[dict[str, ArrayLike]], dict[str, Any]]

Layers_t = Sequence[dict[str, Any]]
Layers_separate_input_t = dict[str, Layers_t]
PathLike = os.PathLike | str  # type: ignore[type-arg]
