"""Name-based registry mapping a solver name to its ``(solver, velocity-field)`` classes.

:class:`cellflow.model.CellFlow` resolves the solver and velocity-field classes for ``solver=<name>``
from :data:`SOLVER_REGISTRY` instead of a hardcoded ``if/elif`` chain, so additional solvers can be
made selectable with :func:`register_solver` without touching the model code.
"""

from __future__ import annotations

__all__ = ["SOLVER_REGISTRY", "register_solver"]

#: Maps a solver name (as passed to ``CellFlow(solver=...)``) to its ``(solver_class,
#: velocity_field_class)`` pair. Populated for the built-ins in :mod:`cellflow.solvers`.
SOLVER_REGISTRY: dict[str, tuple[type, type]] = {}


def register_solver(name: str, solver_cls: type, vf_cls: type) -> None:
    """Register a solver so it can be selected via ``CellFlow(solver=name)``.

    Parameters
    ----------
    name
        Name used to select the solver, e.g. ``"otfm"``. Re-registering an existing name overwrites it.
    solver_cls
        The solver class, e.g. :class:`cellflow.solvers.OTFlowMatching`.
    vf_cls
        The velocity-field class paired with ``solver_cls``.

    Returns
    -------
    :obj:`None`. Adds ``name -> (solver_cls, vf_cls)`` to :data:`SOLVER_REGISTRY`.
    """
    SOLVER_REGISTRY[name] = (solver_cls, vf_cls)
