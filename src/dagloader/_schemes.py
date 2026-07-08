"""Factory helpers ("the above layer") that fill a :class:`Scheme` from an obs table.

These are conveniences, not privileged: they build the same ``Node`` / ``Bind`` / ``Scheme`` a caller
could assemble by hand. :func:`perturbation_scheme` is the cellflow-shaped case (control → perturbed,
matched on context). Read parameters (batch / chunk / preload) are a separate :class:`SamplerConfig`
given to the loader, so the factory only builds structure. sc-flow-tools-shaped schemes are usually
assembled directly from a ``DataManager`` config; see ``README.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dagloader._io import obs_columns
from dagloader._schema import Bind, Container, Node, Scheme, uniform

__all__ = ["perturbation_scheme"]


def perturbation_scheme(
    source: Container,
    *,
    context: Sequence[str],
    perturbation: Sequence[str],
    control_values: Mapping[str, object],
    key: str = "X",
    seed: int = 0,
) -> Scheme:
    """Fill a perturbation Scheme from the obs table: root = perturbed combos, child = control combos.

    ``source`` is an in-memory AnnData or an on-disk DatasetCollection. There is no ``select`` step —
    control vs perturbed is encoded purely by which combinations carry weight. The control node is
    bound to the perturbed root on ``context``, so each batch's control cells come from the same
    context (cell line, …) as the perturbed cells — the source↔target matching.

    Parameters mirror cellflow: ``context`` = ``split_covariates`` (grouping/context), ``perturbation``
    = the perturbation columns, ``control_values`` = which value marks control per column, ``key`` =
    ``sample_rep``. Read parameters (batch/chunk/preload) go to the loader's ``SamplerConfig``.
    """
    cols = (*context, *perturbation)
    combos = [tuple(r) for r in obs_columns(source, cols).drop_duplicates().to_numpy()]

    def is_control(combo: tuple) -> bool:
        return all(combo[cols.index(c)] == v for c, v in control_values.items())

    pert = [c for c in combos if not is_control(c)]
    ctrl = [c for c in combos if is_control(c)]
    return Scheme(
        sources={"data": source},
        nodes={
            # non-control combos weighted (rest excluded = the selection); control combos weighted.
            "pert": Node("data", cols, key, uniform(pert)),
            "ctrl": Node("data", cols, key, uniform(ctrl)),
        },
        root="pert",
        binds=(Bind("pert", "ctrl", common=tuple(context)),),
        seed=seed,
    )
