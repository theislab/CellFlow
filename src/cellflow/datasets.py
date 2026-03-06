from typing import Any

import anndata as ad
from huggingface_hub import hf_hub_download

__all__ = [
    "ineurons",
    "pbmc_cytokines",
]

_HF_REPO = "theislab/cellflow-datasets"


def ineurons(
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:
    """Preprocessed and extracted data as provided in :cite:`lin2023human`.

    The :attr:`anndata.AnnData.X` is based on reprocessing of the counts data using
    :func:`scanpy.pp.normalize_total` and :func:`scanpy.pp.log1p`.

    Parameters
    ----------
    force_download
        Whether to force-download the data.
    kwargs
        Keyword arguments for :func:`anndata.read_h5ad`.

    Returns
    -------
    Annotated data object.
    """
    return _load_dataset(
        filename="ineurons.h5ad",
        force_download=force_download,
        **kwargs,
    )


def pbmc_cytokines(
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:
    """PBMC samples from 12 donors treated with 90 cytokines.

    Processed data from https://www.parsebiosciences.com/datasets/10-million-human-pbmcs-in-a-single-experiment/,
    subset to 2000 highly varibale genes, containing embeddings for
    donors and cytokines.

    Parameters
    ----------
    force_download
        Whether to force-download the data.
    kwargs
        Keyword arguments for :func:`anndata.read_h5ad`.

    Returns
    -------
    Annotated data object.
    """
    return _load_dataset(
        filename="pbmc_parse.h5ad",
        force_download=force_download,
        **kwargs,
    )


def zesta(
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:
    """Developing zebrafish with genetic perturbations.

    Dataset published in :cite:`saunders2023embryo` containing single-cell
    RNA-seq readouts of the embryonic zebrafish at 5 time points with up
    to 23 different genetic perturbations.

    Parameters
    ----------
    force_download
        Whether to force-download the data.
    kwargs
        Keyword arguments for :func:`anndata.read_h5ad`.

    Returns
    -------
    Annotated data object.
    """
    return _load_dataset(
        filename="zesta.h5ad",
        force_download=force_download,
        **kwargs,
    )


def _load_dataset(
    filename: str,
    *,
    repo_id: str = _HF_REPO,
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:
    fpath = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        force_download=force_download,
    )
    return ad.read_h5ad(fpath, **kwargs)
