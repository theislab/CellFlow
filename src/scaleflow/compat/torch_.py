from typing import TYPE_CHECKING

from scaleflow._optional import OptionalDependencyNotAvailable, torch_required_msg

try:
    from torch.utils.data import IterableDataset as TorchIterableDataset  # type: ignore

    TORCH_AVAILABLE = True
except ImportError as _:
    TORCH_AVAILABLE = False

    class TorchIterableDataset:  # noqa: D101
        def __init__(self, *args, **kwargs):
            raise OptionalDependencyNotAvailable(torch_required_msg())


if TYPE_CHECKING:
    # keeps type checkers aligned with the real type
    from torch.utils.data import IterableDataset as TorchIterableDataset  # noqa: F401
