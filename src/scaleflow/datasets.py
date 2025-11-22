import os
from typing import Any

import anndata as ad
from scanpy.readwrite import _check_datafile_present_and_download

from scaleflow._types import PathLike

__all__ = [
    "ineurons",
    "pbmc_cytokines",
]


def ineurons(
    path: PathLike = "~/.cache/scaleflow/ineurons.h5ad",
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:
    """Preprocessed and extracted data as provided in :cite:`lin2023human`.

    The :attr:`anndata.AnnData.X` is based on reprocessing of the counts data using
    :func:`scanpy.pp.normalize_total` and :func:`scanpy.pp.log1p`.

    Parameters
    ----------
    path
        Path where to save the file.
    force_download
        Whether to force-download the data.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    Annotated data object.
    """
    return _load_dataset_from_url(
        path,
        backup_url="https://figshare.com/ndownloader/files/52852961",
        expected_shape=(54134, 2000),  # TODO: adapt this, and enable check
        force_download=force_download,
        **kwargs,
    )


def pbmc_cytokines(
    path: PathLike = "~/.cache/scaleflow/pbmc_parse.h5ad",
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:
    """PBMC samples from 12 donors treated with 90 cytokines.

    Processed data from https://www.parsebiosciences.com/datasets/10-million-human-pbmcs-in-a-single-experiment/,
    subset to 2000 highly varibale genes, containing embeddings for
    donors and cytokines.

    Parameters
    ----------
    path
        Path where to save the file.
    force_download
        Whether to force-download the data.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    Annotated data object.
    """
    return _load_dataset_from_url(
        path,
        backup_url="https://figshare.com/ndownloader/files/53372768",
        expected_shape=(54134, 2000),  # TODO: adapt this, and enable check
        force_download=force_download,
        **kwargs,
    )


def zesta(
    path: PathLike = "~/.cache/scaleflow/zesta.h5ad",
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:
    """Developing zebrafish with genetic perturbations.

    Dataset published in :cite:`saunders2023embryo` containing single-cell
    RNA-seq readouts of the embryonic zebrafish at 5 time points with up
    to 23 different genetic perturbations.

    Parameters
    ----------
    path
        Path where to save the file.
    force_download
        Whether to force-download the data.
    kwargs
        Keyword arguments for :func:`scanpy.read`.

    Returns
    -------
    Annotated data object.
    """
    return _load_dataset_from_url(
        path,
        backup_url="https://figshare.com/ndownloader/files/52966469",
        expected_shape=(54134, 2000),  # TODO: adapt this, and enable check
        force_download=force_download,
        **kwargs,
    )


def _load_dataset_from_url(
    fpath: PathLike,
    *,
    backup_url: str,
    expected_shape: tuple[int, int],
    force_download: bool = False,
    **kwargs: Any,
) -> ad.AnnData:
    fpath = os.path.expanduser(fpath)
    if not fpath.endswith(".h5ad"):
        fpath += ".h5ad"
    if force_download and os.path.exists(fpath):
        os.remove(fpath)
    if not _check_datafile_present_and_download(backup_url=backup_url, path=fpath):
        raise FileNotFoundError(f"File `{fpath}` not found or download failed.")
    data = ad.read_h5ad(filename=fpath, **kwargs)

    # TODO: enable the dataset shape check
    # if data.shape != expected_shape:
    #    raise ValueError(
    #        f"Expected AnnData object to have shape `{expected_shape}`, found `{data.shape}`."
    #    )

    return data
