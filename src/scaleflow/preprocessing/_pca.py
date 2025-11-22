import anndata as ad
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix

from scaleflow._types import ArrayLike

__all__ = ["centered_pca", "reconstruct_pca", "project_pca"]


def centered_pca(
    adata: ad.AnnData,
    n_comps: int = 50,
    layer: str | None = None,
    method: str = "scanpy",
    keep_centered_data: bool = True,
    copy: bool = False,
    **kwargs,
) -> ad.AnnData | None:
    """Performs PCA on the centered data matrix and stores the results in :attr:`~anndata.AnnData.obsm`.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    n_comps
        Number of principal components to compute.
    layer
        Layer in :attr:`~anndata.AnnData.layers` to use for PCA. If `None`, uses `adata.X`.
    method
        Method to use for PCA. If ``"rapids"``, uses :func:`rapids_singlecell.pp.pca` with GPU acceleration.
        Otherwise, uses :func:`scanpy.pp.pca`.
    keep_centered_data
        Whether to keep the centered data. Set to :obj:`False` to save memory.
    copy
        Return a copy of ``adata`` instead of updating it in place.
    kwargs
        Additional arguments to pass to :func:`scanpy.pp.pca` or :func:`rapids_singlecell.pp.pca`

    Returns
    -------
        If ``copy`` is :obj:`True`, returns a new :class:`~anndata.AnnData` object with the PCA
        results stored in :attr:`~anndata.AnnData.obsm`. Otherwise, updates ``adata`` in place.

        Sets the following fields:

        - ``.obsm["X_pca"]``: PCA coordinates.
        - ``.varm["PCs"]``: Principal components.
        - ``.varm["X_mean"]``: Mean of the data matrix.
        - ``.layers["X_centered"]``: Centered data matrix. # TODO adapt
        - ``.uns['pca']['variance_ratio']``: Variance ratio of each principal component.
        - ``.uns['pca']['variance']``: Variance of each principal component.
    """
    adata = adata.copy() if copy else adata
    X = adata.X if layer in [None, "X"] else adata.layers[layer]

    adata.varm["X_mean"] = np.array(X.mean(axis=0).T)
    adata.layers["X_centered"] = np.array(adata.X - adata.varm["X_mean"].T)

    if method == "rapids":
        try:
            import rapids_singlecell as rsc
        except ImportError:
            raise ImportError(
                "rapids_singlecell is not installed. To use GPU acceleration for pca computation, please install it via `pip install rapids-singlecell`."
            ) from None
        else:
            rsc.pp.pca(
                adata,
                n_comps=n_comps,
                layer="X_centered",
                zero_center=False,
                copy=False,
                **kwargs,
            )
    elif method == "scanpy" or method is None:
        sc.pp.pca(
            adata,
            n_comps=n_comps,
            layer="X_centered",
            zero_center=False,
            copy=False,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid method: {method}. Valid options are 'scanpy' and 'rapids'.")

    if keep_centered_data:
        adata.layers["X_centered"] = csr_matrix(adata.layers["X_centered"])
    else:
        del adata.layers["X_centered"]

    if copy:
        return adata


def reconstruct_pca(
    query_adata: ad.AnnData,
    use_rep: str = "X_pca",
    ref_adata: ad.AnnData | None = None,
    ref_means: ArrayLike | None = None,
    ref_pcs: ArrayLike | None = None,
    layers_key_added: str = "X_recon",
    copy: bool = False,
) -> ad.AnnData | None:
    """Performs PCA on the data matrix and projects the data to the principal components.

    Parameters
    ----------
    query_adata
        An :class:`~anndata.AnnData` object with the query data.
    use_rep : str
        Representation to use for PCA. If ``'X'``, uses :attr:`~anndata.AnnData.X`. Otherwise, uses
        ``adata.obsm[use_rep]``.
    ref_adata
        An :class:`~anndata.AnnData` object with the reference data containing
        ``adata.varm["X_mean"]`` and ``adata.varm["PCs"]``.
    ref_means
        Mean of the reference data. Only used if ``ref_adata`` is :obj:`None`.
    ref_pcs
        Principal components of the reference data. Only used if ``ref_adata`` is :obj:`None`.
    layers_key_added
        Key in :attr:`~anndata.AnnData.layers` to store the reconstructed data matrix.
    copy
        Return a copy of ``adata`` instead of updating it in place.

    Returns
    -------
        If `copy` is :obj:`True`, returns a new :class:`~anndata.AnnData` object with the PCA
        results stored in :attr:`~anndata.AnnData.obsm`. Otherwise, updates ``adata`` in place.

        Sets the following fields:

        - ``.layers["X_recon"]``: Reconstructed data matrix.
    """
    if copy:
        query_adata = query_adata.copy()

    if (ref_adata is None) and ((ref_means is None) or (ref_pcs is None)):
        raise ValueError("Either `ref_adata` or `ref_means` and `ref_pcs` must be provided.")

    X = query_adata.X if use_rep == "X" else query_adata.obsm[use_rep]
    if ref_adata is not None:
        ref_means = ref_adata.varm["X_mean"]
        ref_pcs = ref_adata.varm["PCs"]

    X_recon = np.array(X @ np.transpose(np.array(ref_pcs)) + np.transpose(np.array(ref_means)))
    query_adata.layers[layers_key_added] = X_recon

    if copy:
        return query_adata


def project_pca(
    query_adata: ad.AnnData,
    ref_adata: ad.AnnData | None = None,
    ref_means: ArrayLike | None = None,
    ref_pcs: ArrayLike | None = None,
    layer: str | None = None,
    obsm_key_added: str = "X_pca",
    copy: bool = False,
) -> ad.AnnData | None:
    """Projects the query data to the principal components of the reference data.

    Parameters
    ----------
    query_adata
        An :class:`~anndata.AnnData` object with the query data.
    ref_adata
        An :class:`~anndata.AnnData` object with the reference data containing
        ``adata.varm["X_mean"]`` and ``adata.varm["PCs"]``.
    ref_means
        Mean of the reference data. Only used if ``ref_adata`` is :obj:`None`.
    ref_pcs
        Principal components of the reference data. Only used if ``ref_adata`` is :obj:`None`.
    layer
        Layer in :attr:`~anndata.AnnData.layers` to use for PCA.
    obsm_key_added
        Key in :attr:`~anndata.AnnData.obsm` to store the PCA coordinates.

    Returns
    -------
        If ``copy`` is :obj:`True`, returns a new :class:`~anndata.AnnData` object with the PCA
        results stored in :attr:`~anndata.AnnData.obsm`. Otherwise, updates ``adata`` in place.

        Sets the following fields:

        - ``.obsm["X_pca"]``: PCA coordinates.
    """
    if copy:
        query_adata = query_adata.copy()

    if (ref_adata is None) and ((ref_means is None) or (ref_pcs is None)):
        raise ValueError("Either `ref_adata` or `ref_means` and `ref_pcs` must be provided.")

    X = query_adata.X if layer in [None, "X"] else query_adata.layers[layer]
    if ref_adata is not None:
        ref_means = ref_adata.varm["X_mean"]
        ref_pcs = ref_adata.varm["PCs"]

    query_adata.obsm[obsm_key_added] = np.array((X - np.transpose(np.array(ref_means))) @ np.array(ref_pcs))

    if copy:
        return query_adata
