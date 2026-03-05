from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from cellflow._types import ArrayLike

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData

logger = logging.getLogger(__name__)

__all__ = ["CFJaxSCVI"]

_SCVI_ERR_MSG = "scvi-tools is required for cellflow.external. Please install via `pip install 'cellflow[external]'`."

try:
    from scvi import REGISTRY_KEYS
    from scvi.data import AnnDataManager
    from scvi.data.fields import CategoricalObsField, LayerField
    from scvi.model.base import BaseModelClass, JaxTrainingMixin
    from scvi.utils import setup_anndata_dsp

    _HAS_SCVI = True
except ImportError:
    _HAS_SCVI = False

    class BaseModelClass:  # type: ignore[no-redef]
        pass

    class JaxTrainingMixin:  # type: ignore[no-redef]
        pass


def _check_scvi_deps() -> None:
    if not _HAS_SCVI:
        raise ImportError(_SCVI_ERR_MSG)


class CFJaxSCVI(JaxTrainingMixin, BaseModelClass):
    """CellFlow-specific JAX scVI model.

    A lightweight VAE that inherits the scvi-tools training and data
    infrastructure but constructs its own :class:`CFJaxVAE` module,
    avoiding coupling to upstream ``JaxSCVI.__init__`` kwargs.

    Parameters
    ----------
    adata
        AnnData registered via :meth:`CFJaxSCVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    dropout_rate
        Dropout rate for neural networks.
    gene_likelihood
        One of ``'nb'``, ``'poisson'``, ``'normal'``.
    **model_kwargs
        Forwarded to :class:`CFJaxVAE`.
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        dropout_rate: float = 0.1,
        gene_likelihood: Literal["nb", "poisson", "normal"] = "normal",
        **model_kwargs,
    ):
        _check_scvi_deps()
        from cellflow.external._scvi_utils import CFJaxVAE

        BaseModelClass.__init__(self, adata)

        self.module = CFJaxVAE(
            n_input=self.summary_stats.n_vars,
            n_batch=self.summary_stats.n_batch,
            n_hidden=n_hidden,
            n_latent=n_latent,
            dropout_rate=dropout_rate,
            gene_likelihood=gene_likelihood,
            **model_kwargs,
        )

        self._model_summary_string = ""
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        batch_key: str | None = None,
        **kwargs,
    ):
        """Set up :class:`~anndata.AnnData` for use with :class:`CFJaxSCVI`.

        Parameters
        ----------
        adata
            AnnData object.
        layer
            Key for :attr:`~anndata.AnnData.layers` to use as expression data.
        batch_key
            Key for :attr:`~anndata.AnnData.obs` to use as batch information.
        """
        _check_scvi_deps()
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

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
            :class:`~anndata.AnnData` object. If :obj:`None`, defaults to the
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
        r"""Return the reconstructed expression for each cell.

        Parameters
        ----------
        data
            AnnData or array-like with the data to reconstruct.
        use_rep
            Key for :attr:`~anndata.AnnData.obsm` that contains the latent representation to use.
        indices
            Indices of cells in adata to use. If :obj:`None`, all cells are used.
        give_mean
            Whether to return the mean of the negative binomial distribution or the
            unscaled expression.
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
        split_indixes = np.arange(0, data.obsm[use_rep].shape[0], batch_size)
        recon = []
        for array_dict, z_idx in zip(scdl, split_indixes, strict=False):
            z_batch = data.obsm[use_rep][z_idx : z_idx + batch_size, :]
            inference_outputs = {"z": z_batch}
            out = jit_generative_fn(self.module.rngs, array_dict, inference_outputs)
            if give_mean and self.module.gene_likelihood != "normal":
                x = out["px"].mean
            else:
                x = out["rho"]
            recon.append(x)
        recon = jnp.concatenate(recon, axis=0)

        return self.module.as_numpy_array(recon)

    def to_device(self, device):
        pass

    @property
    def device(self):
        return self.module.device
