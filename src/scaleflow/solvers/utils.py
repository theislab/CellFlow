import jax
import jax.numpy as jnp


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


def _multivariate_normal(
    rng: jax.Array,
    shape: tuple[int, ...],
    dim: int,
    mean: float = 0.0,
    cov: float = 1.0,
) -> jnp.ndarray:
    mean = jnp.full(dim, fill_value=mean)
    cov = jnp.diag(jnp.full(dim, fill_value=cov))
    return jax.random.multivariate_normal(rng, mean=mean, cov=cov, shape=shape)
