from typing import Literal

import anndata as ad
import jax
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import sparse

from cfp._logging import logger

__all__ = ["compute_wknn", "transfer_labels"]


def compute_wknn(
    ref_adata: ad.AnnData = None,
    query_adata: ad.AnnData = None,
    n_neighbors: int = 100,
    ref_rep_key: str = "X_pca",
    query_rep_key: str = "X_pca",
    uns_key_added: str = "wknn",
    query2ref: bool = True,
    ref2query: bool = False,
    weighting_scheme: (
        Literal["top_n", "jaccard", "jaccard_square"] | None
    ) = "jaccard_square",
    top_n: int | None = None,
    copy: bool = False,
):
    """
    Compute the weighted k-nearest neighbors graph between the reference and query datasets

    Parameters
    ----------
    ref_adata : ad.AnnData
        An :class:`~anndata.AnnData` object with the reference representation to build ref-query neighbor graph
    query_adata : ad.AnnData
        An :class:`~anndata.AnnData` object with the query representation to build ref-query neighbor graph
    n_neighbors : int
        Number of neighbors per cell
    ref_rep_key : str
        Key in `ref_adata.obsm` containing the reference representation
    query_rep_key : str
        Key in `query_adata.obsm` containing the query representation
    uns_key_added : str
        Key to store the weighted k-nearest neighbors graph in `adata.uns`
    query2ref : bool
        Consider query-to-ref neighbors
    ref2query : bool
        Consider ref-to-query neighbors
    weighting_scheme : str
        How to weight edges in the ref-query neighbor graph. Options are:
        - `None`: No weighting
        - `"top_n"`: Binaries edges based on the top `top_n` neighbors.
        - `"jaccard"`: Weight edges based on the Jaccard index.
        - `"jaccard_square"`: Weight edges based on the square of the Jaccard index.
    top_n : int
        The number of top neighbors to consider
    copy : bool
        Return a copy of `ref_adata` instead of updating it in place

    Returns
    -------
        If `copy` is :obj:`True`, returns a new :class:`~anndata.AnnData` object with the weighted k-nearest neighbors stored in `adata.uns`. Otherwise, updates `adata` in place.

        Sets the following fields:
        `.uns[uns_key_added]`: Weighted k-nearest neighbors graph
    """
    ref_adata = ref_adata.copy() if copy else ref_adata

    ref = ref_adata.X if ref_rep_key == "X" else ref_adata.obsm[ref_rep_key]
    query = query_adata.X if query_rep_key == "X" else query_adata.obsm[query_rep_key]

    wknn = _get_wknn(
        ref,
        query,
        k=n_neighbors,
        query2ref=query2ref,
        ref2query=ref2query,
        weighting_scheme=weighting_scheme,
        top_n=top_n,
    )

    ref_adata.uns[uns_key_added] = wknn

    if copy:
        return ref_adata


def transfer_labels(
    query_adata: ad.AnnData,
    ref_adata: ad.AnnData,
    label_key: str,
    wknn_key: str = "wknn",
    copy: bool = False,
):
    """Transfer labels from the reference to the query dataset.

    Parameters
    ----------
    query_adata : ad.AnnData
        An :class:`~anndata.AnnData` object with the query data
    ref_adata : ad.AnnData
        An :class:`~anndata.AnnData` object with the reference data
    label_key : str
        Key in `ref_adata.obs` containing the labels
    wknn_key : str
        Key in `ref_adata.uns` containing the weighted k-nearest neighbors graph
    copy : bool
        Return a copy of `query_adata` instead of updating it in place

    Returns
    -------
        If `copy` is :obj:`True`, returns a new :class:`~anndata.AnnData` object with the transferred labels stored in `adata.obs`. Otherwise, updates `adata` in place.

        Sets the following fields:
        `.obs[f"{label_key}_transfer"]`: Transferred labels
        `.obs[f"{label_key}_transfer_score"]`: Confidence scores for the transferred labels

    """
    query_adata = query_adata.copy() if copy else query_adata

    if wknn_key not in ref_adata.uns:
        raise ValueError(
            f"Key {wknn_key} not found in `ref_adata.uns`. Please run `compute_wknn` first. To compute the weighted k-nearest neighbors graph."
        )

    wknn = ref_adata.uns[wknn_key]

    scores = pd.DataFrame(
        wknn @ pd.get_dummies(ref_adata.obs[label_key]),
        columns=pd.get_dummies(ref_adata.obs[label_key]).columns,
        index=query_adata.obs_names,
    )

    query_adata.obs[f"{label_key}_transfer"] = scores.idxmax(1)
    query_adata.obs[f"{label_key}_transfer_score"] = scores.max(1)

    if copy:
        return query_adata


def _nn2adj_gpu(nn, n1=None, n2=None):
    if n1 is None:
        n1 = nn[1].shape[0]
    if n2 is None:
        n2 = np.max(nn[1].flatten())

    df = pd.DataFrame(
        {
            "i": np.repeat(range(nn[1].shape[0]), nn[1].shape[1]),
            "j": nn[1].flatten(),
            "x": nn[0].flatten(),
        }
    )
    adj = sparse.csr_matrix(
        (np.repeat(1, df.shape[0]), (df["i"], df["j"])), shape=(n1, n2)
    )

    return adj


def _nn2adj_cpu(nn, n1=None, n2=None):
    if n1 is None:
        n1 = nn[0].shape[0]
    if n2 is None:
        n2 = np.max(nn[0].flatten())

    df = pd.DataFrame(
        {
            "i": np.repeat(range(nn[0].shape[0]), nn[0].shape[1]),
            "j": nn[0].flatten(),
            "x": nn[1].flatten(),
        }
    )

    adj = sparse.csr_matrix(
        (np.repeat(1, df.shape[0]), (df["i"], df["j"])), shape=(n1, n2)
    )

    return adj


def _build_nn(
    ref: ArrayLike,
    query: ArrayLike | None = None,
    k: int = 100,
):
    ref = np.array(ref)
    query = np.array(query) if query is not None else ref

    try:
        from cuml.neighbors import NearestNeighbors

        jax.devices("gpu")
    except ImportError:
        logger.info(
            "cuML is not installed or GPU is not available. Falling back to neighborhood estimation using CPU with pynndescent."
        )
    else:
        model = NearestNeighbors(n_neighbors=k)
        model.fit(ref)
        knn = model.kneighbors(query)
        return _nn2adj_gpu(knn, n1=query.shape[0], n2=ref.shape[0])

    logger.info(
        "Failed to call cuML. Falling back to neighborhood estimation using CPU with pynndescent."
    )

    try:
        from pynndescent import NNDescent
    except ImportError:
        raise ImportError(
            "pynndescent is not installed. To compute the wknn graph, please install it via `pip install pynndescent`."
        ) from None

    index = NNDescent(ref)
    knn = index.query(query, k=k)
    return _nn2adj_cpu(knn, n1=query.shape[0], n2=ref.shape[0])


def _get_wknn(
    ref: ArrayLike,
    query: ArrayLike,
    k: int = 100,
    query2ref: bool = True,
    ref2query: bool = False,
    weighting_scheme: (
        Literal["top_n", "jaccard", "jaccard_square"] | None
    ) = "jaccard_square",
    top_n: int | None = None,
):
    """
    Compute the weighted k-nearest neighbors graph between the reference and query datasets

    Parameters
    ----------
    ref : ArrayLike
        The reference representation to build ref-query neighbor graph
    query : ArrayLike
        The query representation to build ref-query neighbor graph
    k : int
        Number of neighbors per cell
    query2ref : bool
        Consider query-to-ref neighbors
    ref2query : bool
        Consider ref-to-query neighbors
    weighting_scheme : str
        How to weight edges in the ref-query neighbor graph
    top_n : int
        The number of top neighbors to consider
    """
    adj_q2r = _build_nn(ref=ref, query=query, k=k)

    adj_r2q = None
    if ref2query:
        adj_r2q = _build_nn(ref=query, query=ref, k=k)

    if query2ref and not ref2query:
        adj_knn = adj_q2r.T
    elif ref2query and not query2ref:
        adj_knn = adj_r2q
    elif ref2query and query2ref:
        adj_knn = ((adj_r2q + adj_q2r.T) > 0) + 0
    else:
        logger.warn(
            "At least one of query2ref and ref2query should be True. Reset to default with both being True."
        )
        adj_knn = ((adj_r2q + adj_q2r.T) > 0) + 0

    adj_ref = _build_nn(ref=ref, k=k)
    num_shared_neighbors = adj_q2r @ adj_ref.T
    num_shared_neighbors_nn = num_shared_neighbors.multiply(adj_knn.T)

    wknn = num_shared_neighbors_nn.copy()
    if weighting_scheme == "top_n":
        if top_n is None:
            top_n = k // 4 if k > 4 else 1
        wknn = (wknn > top_n) * 1
    elif weighting_scheme == "jaccard":
        wknn.data = wknn.data / (k + k - wknn.data)
    elif weighting_scheme == "jaccard_square":
        wknn.data = (wknn.data / (k + k - wknn.data)) ** 2

    return wknn
