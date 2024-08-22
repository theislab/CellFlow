from collections.abc import Sequence
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import cosine_similarity

from cfp import _constants, _logging

class CellFlow:
    pass


def set_plotting_vars(
    adata: ad.AnnData,
    *,
    key: str,
    value: Any,
    override: bool = True,
) -> None:
    uns_key = _constants.CFP_KEY
    adata.uns.setdefault(uns_key, {})
    if not override and key in adata.uns[uns_key]:
        raise KeyError(
            f"Data in `adata.uns[{uns_key!r}][{key!r}]` "
            f"already exists, use `override=True`."
        )
    adata.uns[uns_key][key] = value


def _get_palette(
    n_colors: int, palette_name: str | None = "Set1"
) -> sns.palettes._ColorPalette:
    try:
        palette = sns.color_palette(palette_name)
    except ValueError:
        _logging.logger.info("Palette not found. Using default palette tab10")
        palette = sns.color_palette()
    while len(palette) < n_colors:
        palette += palette

    return palette


def _get_colors(
    labels: Sequence[str],
    palette: str | None = None,
    palette_name: str | None = None,
) -> dict[str, str]:
    n_colors = len(labels)
    if palette is None:
        palette = _get_palette(n_colors, palette_name)
    col_dict = dict(zip(labels, palette[:n_colors], strict=False))
    return col_dict


def get_plotting_vars(adata: ad.AnnData, *, key: str) -> Any:
    uns_key = _constants.CFP_KEY
    try:
        return adata.uns[uns_key][key]
    except KeyError:
        raise KeyError(f"No data found in `adata.uns[{uns_key!r}][{key!r}]`.") from None


def _input_to_adata(obj: ad.AnnData | CellFlow) -> ad.AnnData:
    if isinstance(obj, ad.AnnData):
        return obj
    elif isinstance(obj, CellFlow):
        return obj.adata
    else:
        raise ValueError(
            f"obj must be an AnnData or CellFlow object, but found {type(obj)}"
        )


def _compute_umap_from_df(
    df: pd.DataFrame,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    n_components: int = 2,
    **kwargs,
) -> pd.DataFrame:
    adata_tmp = ad.AnnData(df.values)
    if adata_tmp.n_vars > n_pcs:
        sc.pp.pca(adata_tmp, n_comps=n_pcs)
    sc.pp.neighbors(adata_tmp, n_neighbors=n_neighbors)
    sc.tl.umap(adata_tmp, n_components=n_components, **kwargs)

    return pd.DataFrame(
        data=adata_tmp.obsm["X_umap"],
        columns=list(range(n_components)),
        index=df.index,
    )


def _compute_pca_from_df(df: pd.DataFrame, n_components: int = 30) -> pd.DataFrame:
    adata_tmp = ad.AnnData(df.values)
    sc.pp.pca(adata_tmp, n_comps=n_components)
    return pd.DataFrame(
        data=adata_tmp.obsm["X_pca"],
        columns=list(range(adata_tmp.obsm["X_pca"].shape[1])),
        index=df.index,
    )


def _compute_kernel_pca_from_df(
    df: pd.DataFrame, n_components: int = 30, **kwargs
) -> pd.DataFrame:
    similarity_matrix = cosine_similarity(df.values)
    np.fill_diagonal(similarity_matrix, 1.0)
    X = KernelPCA(n_components=n_components, kernel="precomputed").fit_transform(
        similarity_matrix
    )
    return pd.DataFrame(data=X, columns=list(range(n_components)), index=df.index)
