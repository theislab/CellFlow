from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from cellflow._types import ArrayLike


@jax.jit
def ema_update(current_model_params: dict, new_model_params: dict, ema: float) -> dict:
    """
    Update parameters using exponential moving average.

    Parameters
    ----------
        current_model_parames
            Current parameters.
        new_model_params
            New parameters to be averaged.
        ema
            Exponential moving average factor
            between `0` and `1`. `0` means no update, `1` means full update.

    Returns
    -------
        Updated parameters after applying EMA.
    """
    new_inference_model_params = jax.tree.map(
        lambda p, tp: p * (1 - ema) + tp * ema, current_model_params, new_model_params
    )
    return new_inference_model_params


def predict_multi_condition(
    predict_fn: Callable[..., ArrayLike],
    predict_fn_unbatched: Callable[..., ArrayLike],
    x: dict[str, ArrayLike],
    condition: dict[str, dict[str, ArrayLike]],
) -> dict[str, ArrayLike]:
    """Predict across multiple conditions using vmap with remainder fallback.

    Stacks all conditions to ``min_cells`` (the smallest condition size) and
    runs a single vmapped prediction. Conditions with more cells than
    ``min_cells`` get a second pass via :func:`jax.tree.map` for the remainder.

    Parameters
    ----------
    predict_fn
        Callable with signature ``(x, condition) -> prediction`` used for the
        vmapped batch. Should already have solver-specific args (rng, kwargs)
        bound.
    predict_fn_unbatched
        Callable with signature ``(x, condition) -> prediction`` used for the
        remainder via :func:`jax.tree.map`. Should already have solver-specific
        args bound.
    x
        Dictionary mapping condition keys to source arrays.
    condition
        Dictionary mapping condition keys to condition dicts.

    Returns
    -------
    Dictionary mapping condition keys to predicted arrays.
    """
    keys = sorted(x.keys())
    condition_keys = sorted(set().union(*(condition[k].keys() for k in keys)))
    n_cells_per_key = {k: x[k].shape[0] for k in keys}
    min_cells = min(n_cells_per_key.values())

    batched_predict = jax.vmap(jax.jit(predict_fn), in_axes=(0, dict.fromkeys(condition_keys, 0)))

    src_inputs = jnp.stack([x[k][:min_cells] for k in keys], axis=0)
    batched_conditions: dict[str, Any] = {}
    for cond_key in condition_keys:
        batched_conditions[cond_key] = jnp.stack([condition[k][cond_key] for k in keys])

    pred_targets = batched_predict(src_inputs, batched_conditions)
    result = {k: pred_targets[i] for i, k in enumerate(keys)}

    remainder_keys = [k for k in keys if n_cells_per_key[k] > min_cells]
    if remainder_keys:
        remainder_x = {k: x[k][min_cells:] for k in remainder_keys}
        remainder_cond = {k: condition[k] for k in remainder_keys}
        remainder_pred = jax.tree.map(
            predict_fn_unbatched,
            remainder_x,
            remainder_cond,
        )
        for k in remainder_keys:
            result[k] = jnp.concatenate([result[k], remainder_pred[k]], axis=0)

    return result
