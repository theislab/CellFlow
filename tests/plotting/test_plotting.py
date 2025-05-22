import matplotlib.pyplot as plt
import pytest

from cellflow.plotting import plot_condition_embedding


class TestCallbacks:
    @pytest.mark.skip()
    @pytest.mark.parametrize("embedding", ["raw_embedding", "UMAP", "PCA", "Kernel_PCA"])
    @pytest.mark.parametrize("dimensions", [(0, 1), (4, 5)])
    @pytest.mark.parametrize("hue", ["dosage", "drug1", None])
    def test_plot_embeddings(self, plotting_df, embedding, dimensions, hue):
        if embedding == "UMAP" and dimensions != (0, 1):
            embedding_kwargs = {"n_components": max(dimensions)}
        else:
            embedding_kwargs = {}
        fig = plot_condition_embedding(
            plotting_df,
            embedding=embedding,
            dimensions=dimensions,
            hue=hue,
            embedding_kwargs=embedding_kwargs,
        )

        assert isinstance(fig, plt.Figure)
