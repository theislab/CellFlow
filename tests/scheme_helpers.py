"""Test-only Scheme factory (moved out of ``binded``'s public API).

``perturbation_scheme`` builds the cellflow-shaped two-node :class:`~binded.Scheme` (control →
perturbed, matched on context) from an obs table. Production cellflow assembles this scheme internally
(:func:`cellflow.data._annbatch.build_annbatch_training`); only the split tests need the standalone
factory, so it lives here rather than shipping as library surface. Importable as ``scheme_helpers`` via
the ``pythonpath = ["tests"]`` pytest setting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from binded import Bind, Node, Scheme, uniform
from binded._io import obs_columns
from binded._schema import Container


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
