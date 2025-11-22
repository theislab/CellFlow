from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from scaleflow._types import ArrayLike

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData

logger = logging.getLogger(__name__)

__all__ = ["CFJaxSCVI"]

try:
    from scvi.model import JaxSCVI
except ImportError:
    JaxSCVI = object


class CFJaxSCVI(JaxSCVI):
    from scaleflow.external._scvi_utils import CFJaxVAE

    _module_cls = CFJaxVAE

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        dropout_rate: float = 0.1,
        gene_likelihood: Literal["nb", "poisson", "normal"] = "normal",
        **model_kwargs,
    ):
        if JaxSCVI is not object:
            super().__init__(adata)
        else:
            pass

        n_batch = self.summary_stats.n_batch

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_latent=n_latent,
            dropout_rate=dropout_rate,
            gene_likelihood=gene_likelihood,
            **model_kwargs,
        )

        self._model_summary_string = ""
        self.init_params_ = self._get_init_params(locals())

    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        n_samples: int = 1,
        batch_size: int | None = None,
    ) -> np.ndarray:  # type: ignore[type-arg]
        r"""Return the latent representation for each cell.

        This is denoted as :math:`z_n` in our manuscripts.

        Parameters
        ----------
        adata
            :class:`~anndata.AnnData` object with equivalent structure to initial
            :class:`~anndata.AnnData` object. If `:obj:`None`, defaults to the
            :class:`~anndata.AnnData` object used to initialize the model.
        indices
            Indices of cells in adata to use. If :obj:`None`, all cells are used.
        give_mean
            Whether to return the mean of the posterior distribution or a sample.
        n_samples
            Number of samples to use for computing the latent representation.
        batch_size
            Minibatch size for data loading into model. Defaults to
            :attr:`scvi.settings.ScviConfig.batch_size`.

        Returns
        -------
        Low-dimensional representation for each cell
        """
        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size, iter_ndarray=True)

        jit_inference_fn = self.module.get_jit_inference_fn(inference_kwargs={"n_samples": n_samples})
        latent = []
        for array_dict in scdl:
            out = jit_inference_fn(self.module.rngs, array_dict)
            if give_mean:
                z = out["qz"].mean
            else:
                z = out["z"]
            latent.append(z)
        concat_axis = 0 if ((n_samples == 1) or give_mean) else 1
        latent = jnp.concatenate(latent, axis=concat_axis)

        return self.module.as_numpy_array(latent)

    def get_reconstructed_expression(
        self,
        data: ArrayLike | AnnData,
        use_rep: str = "X_scVI",
        indices: Sequence[int] | None = None,
        give_mean: bool = False,
        batch_size: int | None = 1024,
    ) -> ArrayLike:
        r"""
        Return the reconstructed expression for each cell.

        This is denoted as :math:`z_n` in our manuscripts.

        Parameters
        ----------
        data
            TODO
        use_rep
            Key for :attr:`~anndata.AnnData.obsm` that contains the latent representation to use.
        indices
            Indices of cells in adata to use. If :obj:`None`, all cells are used.
        give_mean
            Whether to return the mean of the negative binomial distribution or the
            unscaled expression.
        n_samples
            Number of samples to use for computing the latent representation.
        batch_size
            Minibatch size for data loading into model. Defaults to :attr:`scvi.settings.batch_size`.

        Returns
        -------
        reconstructed_expression
        """
        if batch_size is None:
            batch_size = data.obsm[use_rep].shape[0]  # type: ignore[union-attr]

        self._check_if_trained(warn=False)

        data = self._validate_anndata(data)
        scdl = self._make_data_loader(adata=data, indices=indices, batch_size=batch_size, iter_ndarray=True)

        jit_generative_fn = self.module.get_jit_generative_fn()
        # Make dummy dict to conform with scVI functions
        # inference_outputs = {"z": adata.obsm[use_rep]}
        # We also have to batch over z here, so we split it in batches
        split_indixes = np.arange(0, data.obsm[use_rep].shape[0], batch_size)
        recon = []
        for array_dict, z_idx in zip(scdl, split_indixes, strict=False):
            z_batch = data.obsm[use_rep][z_idx : z_idx + batch_size, :]
            # Make dummy dict to conform with scVI functions
            inference_outputs = {"z": z_batch}
            out = jit_generative_fn(self.module.rngs, array_dict, inference_outputs)
            if give_mean and self.module.gene_likelihood != "normal":
                x = out["px"].mean
            else:
                x = out["rho"]
            recon.append(x)
        recon = jnp.concatenate(recon, axis=0)

        return self.module.as_numpy_array(recon)
