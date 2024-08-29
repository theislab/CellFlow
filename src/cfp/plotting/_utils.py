from collections.abc import Sequence
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as stats
import seaborn as sns
from pandas.plotting._matplotlib.tools import create_subplots as _subplots
from pandas.plotting._matplotlib.tools import flatten_axes as _flatten
from scipy.stats import gaussian_kde
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import cosine_similarity

from cfp import _constants, _logging
from cfp._types import ArrayLike


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


def _x_range(data: ArrayLike | list[float], extra: float = 0.2) -> ArrayLike:
    """Compute the x_range for density estimation."""
    try:
        sample_range = np.nanmax(data) - np.nanmin(data)
    except ValueError:
        return np.array([])
    if sample_range < 1e-6:
        return np.array([np.nanmin(data), np.nanmax(data)])
    return np.linspace(
        np.nanmin(data) - extra * sample_range,
        np.nanmax(data) + extra * sample_range,
        1000,
    )


def _setup_axis(
    ax: plt.Axes,
    x_range: ArrayLike,
    col_name: str | None = None,
    grid: bool = False,
    ylabelsize: int | None = None,
    yrot: int | None = None,
) -> None:
    """Setup the axis for the joyplot."""
    if col_name is not None:
        ax.set_yticks([0])
        ax.set_yticklabels([col_name], fontsize=ylabelsize, rotation=yrot)
        ax.yaxis.grid(grid)
    else:
        ax.yaxis.set_visible(False)
    ax.patch.set_alpha(0)
    ax.set_xlim([x_range.min(), x_range.max()])
    ax.tick_params(axis="both", which="both", length=0, pad=10)


def _get_alpha(i: int, n: int, start: float = 0.4, end: float = 1.0) -> float:
    """Compute alpha value for plotting."""
    return start + (1 + i) * (end - start) / n


def _remove_na(data: list[Any] | ArrayLike | pd.Series) -> ArrayLike:
    """Remove NA values from the data."""
    return pd.Series(data).dropna().values


def _moving_average(a: ArrayLike, n: int = 3, zero_padded: bool = False) -> ArrayLike:
    """Calculate the moving average of order n."""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    if zero_padded:
        return ret / n
    else:
        return ret[n - 1 :] / n


def _joyplot(
    data,
    grid=False,
    labels=None,
    sublabels=None,
    xlabels=True,
    xlabelsize=None,
    xrot=None,
    ylabelsize=None,
    yrot=None,
    ax=None,
    figsize=None,
    hist=False,
    bins=10,
    fade=False,
    xlim=None,
    ylim="max",
    fill=True,
    linecolor=None,
    overlap=1,
    background=None,
    range_style="all",
    x_range=None,
    tails=0.2,
    title=None,
    legend=False,
    loc="upper right",
    colormap=None,
    color=None,
    normalize=True,
    floc=None,
    **kwargs,
):
    if fill is True and linecolor is None:
        linecolor = "k"

    if sublabels is None:
        legend = False

    def _get_color(i, num_axes, j, num_subgroups):
        if isinstance(color, list):
            return color[j] if num_subgroups > 1 else color[i]
        elif color is not None:
            return color
        elif isinstance(colormap, list):
            return colormap[j](i / num_axes)
        elif color is None and colormap is None:
            num_cycle_colors = len(plt.rcParams["axes.prop_cycle"].by_key()["color"])
            return plt.rcParams["axes.prop_cycle"].by_key()["color"][
                j % num_cycle_colors
            ]
        else:
            return colormap(i / num_axes)

    ygrid = grid is True or grid == "y" or grid == "both"
    xgrid = grid is True or grid == "x" or grid == "both"

    num_axes = len(data)

    if x_range is None:
        global_x_range = _x_range([v for g in data for sg in g for v in sg])
    else:
        global_x_range = _x_range(x_range, 0.0)

    # Each plot will have its own axis
    fig, axes = _subplots(
        naxes=num_axes,
        ax=ax,
        squeeze=False,
        sharex=True,
        sharey=False,
        figsize=figsize,
        layout_type="vertical",
    )
    _axes = _flatten(axes)

    # The legend must be drawn in the last axis if we want it at the bottom.
    if loc in (3, 4, 8) or "lower" in str(loc):
        legend_axis = num_axes - 1
    else:
        legend_axis = 0

    # A couple of simple checks.
    if labels is not None:
        assert len(labels) == num_axes
    if sublabels is not None:
        assert all(len(g) == len(sublabels) for g in data)
    if isinstance(color, list):
        assert all(len(g) <= len(color) for g in data)
    if isinstance(colormap, list):
        assert all(len(g) == len(colormap) for g in data)

    for i, group in enumerate(data):

        a = _axes[i]
        group_zorder = i
        if fade:
            kwargs["alpha"] = _get_alpha(i, num_axes)

        num_subgroups = len(group)

        if hist:
            # matplotlib hist() already handles multiple subgroups in a histogram
            a.hist(
                group,
                label=sublabels,
                bins=bins,
                color=color,
                range=[min(global_x_range), max(global_x_range)],
                edgecolor=linecolor,
                zorder=group_zorder,
                **kwargs,
            )
        else:
            for j, subgroup in enumerate(group):

                # Compute the x_range of the current plot
                if range_style == "all":
                    # All plots have the same range
                    x_range = global_x_range
                elif range_style == "own":
                    # Each plot has its own range
                    x_range = _x_range(subgroup, tails)
                elif range_style == "group":
                    # Each plot has a range that covers the whole group
                    x_range = _x_range(group, tails)
                elif isinstance(range_style, list | np.ndarray):
                    # All plots have exactly the range passed as argument
                    x_range = _x_range(range_style, 0.0)
                else:
                    raise NotImplementedError("Unrecognized range style.")

                if sublabels is None:
                    sublabel = None
                else:
                    sublabel = sublabels[j]

                element_zorder = group_zorder + j / (num_subgroups + 1)
                element_color = _get_color(i, num_axes, j, num_subgroups)

                _plot_density(
                    a,
                    x_range,
                    subgroup,
                    fill=fill,
                    linecolor=linecolor,
                    label=sublabel,
                    zorder=element_zorder,
                    color=element_color,
                    bins=bins,
                    **kwargs,
                )

        # Setup the current axis: transparency, labels, spines.
        col_name = None if labels is None else labels[i]
        _setup_axis(
            a,
            global_x_range,
            col_name=col_name,
            grid=ygrid,
            ylabelsize=ylabelsize,
            yrot=yrot,
        )

        # When needed, draw the legend
        if legend and i == legend_axis:
            a.legend(loc=loc)
            # Bypass alpha values, in case
            for p in a.get_legend().get_patches():
                p.set_facecolor(p.get_facecolor())
                p.set_alpha(1.0)
            for l in a.get_legend().get_lines():
                l.set_alpha(1.0)

    # Final adjustments

    # Set the y limit for the density plots.
    # Since the y range in the subplots can vary significantly,
    # different options are available.
    if ylim == "max":
        # Set all yaxis limit to the same value (max range among all)
        max_ylim = max(a.get_ylim()[1] for a in _axes)
        min_ylim = min(a.get_ylim()[0] for a in _axes)
        for a in _axes:
            a.set_ylim([min_ylim - 0.1 * (max_ylim - min_ylim), max_ylim])

    elif ylim == "own":
        # Do nothing, each axis keeps its own ylim
        pass

    else:
        # Set all yaxis lim to the argument value ylim
        try:
            for a in _axes:
                a.set_ylim(ylim)
        except ValueError:
            raise ValueError(
                "Warning: the value of ylim must be either 'max', 'own', or a tuple of length 2. The value you provided has no effect."
            ) from None

    # Compute a final axis, used to apply global settings
    last_axis = fig.add_subplot(1, 1, 1)

    # Background color
    if background is not None:
        last_axis.patch.set_facecolor(background)

    # This looks hacky, but all the axes share the x-axis,
    # so they have the same lims and ticks
    last_axis.set_xlim(_axes[0].get_xlim())
    if xlabels is True:
        last_axis.set_xticks(np.array(_axes[0].get_xticks()[1:-1]))
        for t in last_axis.get_xticklabels():
            t.set_visible(True)
            t.set_fontsize(xlabelsize)
            t.set_rotation(xrot)

        # If grid is enabled, do not allow xticks (they are ugly)
        if xgrid:
            last_axis.tick_params(axis="both", which="both", length=0)
    else:
        last_axis.xaxis.set_visible(False)

    last_axis.yaxis.set_visible(False)
    last_axis.grid(xgrid)

    # Last axis on the back
    last_axis.zorder = min(a.zorder for a in _axes) - 1
    _axes = list(_axes) + [last_axis]

    if title is not None:
        plt.title(title)

    # The magic overlap happens here.
    h_pad = 5 + (-5 * (1 + overlap))
    fig.tight_layout(h_pad=h_pad)

    return fig, _axes


def _plot_density(
    ax,
    x_range,
    v,
    kind="kde",
    bw_method=None,
    bins=50,
    fill=False,
    linecolor=None,
    clip_on=True,
    normalize=True,
    floc=None,
    **kwargs,
):
    v = _remove_na(v)
    if len(v) == 0 or len(x_range) == 0:
        return

    if kind == "kde":
        try:
            gkde = gaussian_kde(v, bw_method=bw_method)
            y = gkde.evaluate(x_range)
            y = np.log(y + 1.0)
        except ValueError:
            # Handle cases where there is no data in a group.
            y = np.zeros_like(x_range)
        except np.linalg.LinAlgError as e:
            # Handle singular matrix in kde computation.
            distinct_values = np.unique(v)
            if len(distinct_values) == 1:
                # In case of a group with a single value val,
                # that should have infinite density,
                # return a δ(val)
                val = distinct_values[0]
                _logging.logger.warning(
                    f"The data contains a group with a single distinct value ({val}) "
                    "having infinite probability density. "
                    "Consider using a different visualization."
                )

                # Find index i of x_range
                # such that x_range[i-1] < val ≤ x_range[i]
                i = np.searchsorted(x_range, val)

                y = np.zeros_like(x_range)
                y[i] = 1
            else:
                raise e

    elif kind == "lognorm":
        if floc is not None:
            lnparam = stats.lognorm.fit(v, loc=floc)
        else:
            lnparam = stats.lognorm.fit(v)

        lpdf = stats.lognorm.pdf(x_range, lnparam[0], lnparam[1], lnparam[2])
        if normalize:
            y = lpdf / lpdf.sum()
        else:
            y = lpdf
    elif kind == "counts":
        y, bin_edges = np.histogram(v, bins=bins, range=(min(x_range), max(x_range)))
        # np.histogram returns the edges of the bins.
        # We compute here the middle of the bins.
        x_range = _moving_average(bin_edges, 2)
    elif kind == "normalized_counts":
        y, bin_edges = np.histogram(
            v, bins=bins, density=False, range=(min(x_range), max(x_range))
        )
        # np.histogram returns the edges of the bins.
        # We compute here the middle of the bins.
        y = y / len(v)
        x_range = _moving_average(bin_edges, 2)
    elif kind == "values":
        # Warning: to use values and get a meaningful visualization,
        # x_range must also be manually set in the main function.
        y = v
        x_range = list(range(len(y)))
    else:
        raise NotImplementedError

    if fill:

        ax.fill_between(x_range, 0.0, y, clip_on=clip_on, **kwargs)

        # Hack to have a border at the bottom at the fill patch
        # (of the same color of the fill patch)
        # so that the fill reaches the same bottom margin as the edge lines
        # with y value = 0.0
        kw = kwargs
        kw["label"] = None
        ax.plot(x_range, [0.0] * len(x_range), clip_on=clip_on, **kw)

    if linecolor is not None:
        kwargs["color"] = linecolor

    # Remove the legend labels if we are plotting filled curve:
    # we only want one entry per group in the legend (if shown).
    if fill:
        kwargs["label"] = None

    ax.plot(x_range, y, clip_on=clip_on, **kwargs)
