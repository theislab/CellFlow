import anndata as ad

from scaleflow._types import ArrayLike


def _write_predictions(
    adata: ad.AnnData,
    predictions: dict[str, ArrayLike],
    key_added_prefix: str,
) -> None:
    for pred_key, pred_value in predictions.items():
        if pred_value.ndim == 2:
            adata.obsm[f"{key_added_prefix}{pred_key}"] = pred_value
        elif pred_value.ndim == 3:
            for i in range(pred_value.shape[2]):
                adata.obsm[f"{key_added_prefix}{pred_key}_{i}"] = pred_value[..., i]
        else:
            raise ValueError(f"Predictions for '{pred_key}' have an invalid shape: {pred_value.shape}")
