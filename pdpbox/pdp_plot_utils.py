from .utils import (
    _axes_modify,
    _modify_legend_axes,
    _get_string,
    _make_subplots,
    _display_percentile,
)
from .styles import _prepare_plot_style
from .pdp_calc_utils import _cluster_ice_lines, _sample_ice_lines

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import copy
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def _draw_pdp_countplot(data, cmap, plot_style, axes):

    count_norm = data["count_norm"].values
    norm = mpl.colors.Normalize(vmin=0, vmax=np.max(count_norm))
    _modify_legend_axes(axes, plot_style.font_family)

    xticks = data["x"].values
    axes.imshow(
        np.expand_dims(count_norm, 0),
        aspect="auto",
        cmap=cmap,
        norm=norm,
        alpha=plot_style.dist["fill_alpha"],
        extent=(np.min(xticks) - 0.5, np.max(xticks) + 0.5, 0, 0.5),
    )

    for i in range(len(count_norm)):
        text_color = "black"
        if count_norm[i] >= np.max(count_norm) * 0.5:
            text_color = "white"
        axes.text(
            xticks[i],
            0.25,
            round(count_norm[i], 3),
            ha="center",
            va="center",
            color=text_color,
            fontdict={
                "family": plot_style.font_family,
                "fontsize": plot_style.dist["font_size"],
            },
        )

    axes.set_xticks(xticks)
    # draw the white gaps
    axes.set_xticks(xticks[:-1] + 0.5, minor=True)
    axes.grid(which="minor", color="w", linestyle="-", linewidth=1.5)
    axes.tick_params(which="minor", bottom=False, left=False)


def _draw_pdp_distplot(data, color, axes, plot_style):
    axes.plot(
        data,
        [1] * len(data),
        "|",
        color=color,
        markersize=plot_style.dist["markersize"],
    )
    _modify_legend_axes(axes, plot_style.font_family)


def _set_pdp_xticks(
    xlabel, xticklabels, plot_style, axes, is_numeric_line=False, show_xlabels=True
):
    if xticklabels is not None:
        xticks = np.arange(len(xticklabels))
        if is_numeric_line:
            axes.set_xlim(xticks[0], xticks[-1])
        else:
            axes.set_xlim(xticks[0] - 0.5, xticks[-1] + 0.5)

        axes.set_xticks(xticks)
        axes.set_xticklabels(xticklabels, rotation=plot_style.tick["xticks_rotation"])

    if show_xlabels:
        axes.set_xlabel(
            xlabel,
            fontsize=plot_style.title["subplot_title"]["font_size"],
            fontdict={"family": plot_style.font_family},
        )
    else:
        axes.set_xlabel("")


def _pdp_plot(pdp_isolate_obj, feat_name, target, plot_params):
    plot_style = _prepare_plot_style(feat_name, target, plot_params, "pdp_isolate")
    fig, inner_grid, title_axes = _make_subplots(plot_style)

    line_axes, dist_axes = [], []
    cmaps = plot_style.line["cmaps"]
    for i, t in enumerate(target):
        cmap = cmaps[i % len(cmaps)]
        colors = [cmap, plt.get_cmap(cmap)(0.1), plt.get_cmap(cmap)(1.0)]

        if plot_style.plot_pts_dist:
            inner = GridSpecFromSubplotSpec(
                2,
                1,
                subplot_spec=inner_grid[i],
                wspace=0,
                hspace=0.2,
                height_ratios=[7, 0.5],
            )
            inner_line_axes = plt.subplot(inner[0])
            inner_dist_axes = plt.subplot(inner[1])
        else:
            inner_line_axes = plt.subplot(inner_grid[i])
            inner_dist_axes = None

        fig.add_subplot(inner_line_axes)
        _pdp_line_plot(
            t,
            pdp_isolate_obj,
            colors,
            plot_style,
            inner_line_axes,
        )

        dist_xticklabels = line_xticklabels = plot_style.display_columns
        is_numeric = pdp_isolate_obj.feature_type == "numeric"
        if is_numeric:
            if plot_style.x_quantile:
                line_xticklabels = [
                    _get_string(v) for v in pdp_isolate_obj.feature_grids
                ]
            else:
                dist_xticklabels = line_xticklabels = None
                plot_style.show_percentile = False

        tick_params = {"xlabel": f"class_{t}", "plot_style": plot_style}
        if inner_dist_axes is not None:
            fig.add_subplot(inner_dist_axes)
            _set_pdp_xticks(
                xticklabels=line_xticklabels,
                axes=inner_line_axes,
                is_numeric_line=is_numeric,
                show_xlabels=False,
                **tick_params,
            )
            _set_pdp_xticks(
                xticklabels=dist_xticklabels,
                axes=inner_dist_axes,
                show_xlabels=True,
                **tick_params,
            )
            _pdp_dist_plot(
                pdp_isolate_obj,
                colors,
                plot_style,
                inner_dist_axes,
                inner_line_axes,
            )
            dist_axes.append(inner_dist_axes)
        else:
            _set_pdp_xticks(
                xticklabels=line_xticklabels,
                axes=inner_line_axes,
                is_numeric_line=is_numeric,
                show_xlabels=True,
                **tick_params,
            )

        _display_percentile(inner_line_axes, plot_style)
        line_axes.append(inner_line_axes)

    axes = {"title_axes": title_axes, "line_axes": line_axes, "dist_axes": dist_axes}
    return fig, axes


def _pdp_line_plot(
    class_id,
    pdp_isolate_obj,
    colors,
    plot_style,
    axes,
):
    pdp_result = pdp_isolate_obj.results[class_id]
    pdp = copy.deepcopy(pdp_result.pdp)
    ice_lines = copy.deepcopy(pdp_result.ice_lines)
    feature_grids = pdp_isolate_obj.feature_grids
    cmap, light_color, dark_color = colors

    x = np.arange(len(feature_grids))
    if pdp_isolate_obj.feature_type == "numeric" and not plot_style.x_quantile:
        x = feature_grids

    if plot_style.center:
        pdp -= pdp[0]
        for feat in feature_grids[1:]:
            ice_lines[feat] -= ice_lines[feature_grids[0]]
        ice_lines[feature_grids[0]] = 0

    if plot_style.plot_lines:
        if plot_style.clustering["on"]:
            line_data = _cluster_ice_lines(ice_lines, feature_grids, plot_style)
        else:
            line_data = _sample_ice_lines(ice_lines, plot_style.frac_to_plot)
        _ice_line_plot(x, line_data, axes, cmap)

    std = ice_lines[feature_grids].std().values
    _pdp_std_plot(x, pdp, std, (light_color, dark_color), plot_style, axes)
    _axes_modify(axes, plot_style)


def _pdp_std_plot(x, pdp, std, colors, plot_style, axes):
    upper = pdp + std
    lower = pdp - std

    line_style = plot_style.line
    if plot_style.pdp_hl:
        axes.plot(
            x,
            pdp,
            color=line_style["hl_color"],
            linewidth=line_style["width"] * 3,
            alpha=line_style["hl_alpha"],
        )

    light_color, dark_color = colors
    axes.plot(
        x,
        pdp,
        color=dark_color,
        linewidth=line_style["width"],
        marker="o",
        markersize=line_style["markersize"],
    )
    axes.plot(
        x,
        [0] * len(pdp),
        linestyle="--",
        linewidth=line_style["width"],
        color=line_style["zero_color"],
    )

    if plot_style.std_fill:
        axes.fill_between(
            x, upper, lower, alpha=line_style["fill_alpha"], color=light_color
        )

    ymin, ymax = np.min([np.min(lower) * 2, 0]), np.max([np.max(upper) * 2, 0])
    axes.set_ylim(ymin, ymax)


def _pdp_dist_plot(pdp_isolate_obj, colors, plot_style, axes, line_axes):
    cmap, light_color, dark_color = colors

    if pdp_isolate_obj.feature_type == "numeric" and not plot_style.x_quantile:
        _draw_pdp_distplot(pdp_isolate_obj.dist_data, dark_color, axes, plot_style)
        vmin, vmax = pdp_isolate_obj.dist_data.min(), pdp_isolate_obj.dist_data.max()
        axes.set_xlim(vmin, vmax)
        axes.set_xticks([])
        line_axes.set_xlim(vmin, vmax)
    else:
        _draw_pdp_countplot(pdp_isolate_obj.count_data, cmap, plot_style, axes)

    axes.set_title(
        "distribution of data points",
        fontdict={
            "family": plot_style.font_family,
            "color": plot_style.tick["tick_params"]["labelcolor"],
        },
        fontsize=plot_style.tick["tick_params"]["labelsize"],
    )
    axes.tick_params(**plot_style.tick["tick_params"])


def _ice_line_plot(x, lines, axes, cmap):
    total = len(lines)
    linewidth = np.max([1.0 / np.log10(total), 0.3])
    linealpha = np.min([np.max([1.0 / np.log10(total), 0.3]), 0.8])
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, 20))[5:15]

    for i, line in enumerate(lines.values):
        axes.plot(x, line, linewidth=linewidth, c=colors[i % 10], alpha=linealpha)


def _pdp_contour_plot(
    X, Y, pdp_mx, inter_ax, cmap, norm, inter_fill_alpha, fontsize, plot_params
):
    """Interact contour plot"""

    contour_color = plot_params.get("contour_color", "white")

    level = np.min([X.shape[0], X.shape[1]])
    c1 = inter_ax.contourf(
        X,
        Y,
        pdp_mx,
        N=level,
        origin="lower",
        cmap=cmap,
        norm=norm,
        alpha=inter_fill_alpha,
    )
    c2 = inter_ax.contour(c1, levels=c1.levels, colors=contour_color, origin="lower")
    inter_ax.clabel(c2, fontsize=fontsize, inline=1)
    inter_ax.set_aspect("auto")

    # return the color mapping object for colorbar
    return c1


def _pdp_inter_grid(
    pdp_mx, inter_ax, cmap, norm, inter_fill_alpha, fontsize, plot_params
):
    """Interact grid plot (heatmap)"""

    font_family = plot_params.get("font_family", "Arial")
    im = inter_ax.imshow(
        pdp_mx,
        cmap=cmap,
        norm=norm,
        origin="lower",
        aspect="auto",
        alpha=inter_fill_alpha,
    )

    for r in range(pdp_mx.shape[0]):
        for c in range(pdp_mx.shape[1]):
            text_color = "w"
            if pdp_mx[r, c] >= norm.vmin + (norm.vmax - norm.vmin) * 0.5:
                text_color = "black"
            # column -> x, row -> y
            inter_ax.text(
                c,
                r,
                round(pdp_mx[r, c], 3),
                ha="center",
                va="center",
                color=text_color,
                size=fontsize,
                fontdict={"family": font_family},
            )

    # draw the white gaps
    inter_ax.set_xticks(np.arange(pdp_mx.shape[1] - 1) + 0.5, minor=True)
    inter_ax.set_yticks(np.arange(pdp_mx.shape[0] - 1) + 0.5, minor=True)
    inter_ax.grid(which="minor", color="w", linestyle="-", linewidth=1)

    # return the color mapping object for colorbar
    return im


def _pdp_inter_one(
    pdp_interact_out,
    feature_names,
    plot_type,
    inter_ax,
    x_quantile,
    plot_params,
    norm,
    ticks=True,
):
    """Plot single PDP interact

    Parameters
    ----------

    norm: matplotlib colors normalize
    ticks: bool, default=True
        whether to set ticks for the plot,
        False when it is called by _pdp_inter_three

    """
    cmap = plot_params.get("cmap", "viridis")
    inter_fill_alpha = plot_params.get("inter_fill_alpha", 0.8)
    fontsize = plot_params.get("inter_fontsize", 9)
    font_family = plot_params.get("font_family", "Arial")

    # prepare pdp_mx
    pdp_mx_temp = copy.deepcopy(pdp_interact_out.pdp)
    for feature, feature_type, mark in zip(
        pdp_interact_out.features, pdp_interact_out.feature_types, ["x", "y"]
    ):
        if feature_type in ["numeric", "binary"]:
            pdp_mx_temp[mark] = pdp_mx_temp[feature]
        else:
            # for onehot encoding feature, need to map to numeric representation
            pdp_mx_temp[mark] = pdp_mx_temp[feature].apply(
                lambda x: list(x).index(1), axis=1
            )
    pdp_mx_temp = pdp_mx_temp[["x", "y", "preds"]].sort_values(
        by=["x", "y"], ascending=True
    )

    pdp_inter = copy.deepcopy(pdp_mx_temp["preds"].values)
    n_grids_x, n_grids_y = len(pdp_interact_out.feature_grids[0]), len(
        pdp_interact_out.feature_grids[1]
    )
    # pdp_inter.reshape((n_grids_x, n_grids_y)): each row represents grids_x
    # pdp_inter.reshape((n_grids_x, n_grids_y)).T: each row represents grids_y
    pdp_mx = pdp_inter.reshape((n_grids_x, n_grids_y)).T

    # if it is called by _pdp_inter_three, norm is not None
    if norm is None:
        pdp_min, pdp_max = np.min(pdp_inter), np.max(pdp_inter)
        norm = mpl.colors.Normalize(vmin=pdp_min, vmax=pdp_max)

    inter_params = {
        "pdp_mx": pdp_mx,
        "inter_ax": inter_ax,
        "cmap": cmap,
        "norm": norm,
        "inter_fill_alpha": inter_fill_alpha,
        "fontsize": fontsize,
        "plot_params": plot_params,
    }
    if plot_type == "contour":
        if x_quantile:
            # because we have transpose the matrix
            # pdp_max.shape[1]: x, pdp_max.shape[0]: y
            X, Y = np.meshgrid(range(pdp_mx.shape[1]), range(pdp_mx.shape[0]))
        else:
            # for numeric not quantile
            X, Y = np.meshgrid(
                pdp_interact_out.feature_grids[0], pdp_interact_out.feature_grids[1]
            )
        im = _pdp_contour_plot(X=X, Y=Y, **inter_params)
    elif plot_type == "grid":
        im = _pdp_inter_grid(**inter_params)
    else:
        raise ValueError("plot_type: should be 'contour' or 'grid'")

    if ticks:
        # if it is call by _pdp_inter_three, no need to set ticks
        _axes_modify(font_family=font_family, ax=inter_ax, grid=True)

        if pdp_interact_out.feature_types[0] != "numeric" or x_quantile:
            inter_ax.set_xticks(
                range(len(pdp_interact_out.pdp_isolate_outs[0].display_columns))
            )
            inter_ax.set_xticklabels(
                pdp_interact_out.pdp_isolate_outs[0].display_columns
            )

        if pdp_interact_out.feature_types[1] != "numeric" or x_quantile:
            inter_ax.set_yticks(
                range(len(pdp_interact_out.pdp_isolate_outs[1].display_columns))
            )
            inter_ax.set_yticklabels(
                pdp_interact_out.pdp_isolate_outs[1].display_columns
            )

        inter_ax.set_xlabel(
            feature_names[0], fontsize=12, fontdict={"family": font_family}
        )
        inter_ax.set_ylabel(
            feature_names[1], fontsize=12, fontdict={"family": font_family}
        )

        # insert colorbar
        inter_ax_divider = make_axes_locatable(inter_ax)
        cax = inter_ax_divider.append_axes("right", size="5%", pad="2%")
        if plot_type == "grid":
            cb_num_grids = np.max([np.min([n_grids_x, n_grids_y, 8]), 8])
            boundaries = [
                round(v, 3) for v in np.linspace(norm.vmin, norm.vmax, cb_num_grids)
            ]
            cb = plt.colorbar(im, cax=cax, boundaries=boundaries)
        else:
            cb = plt.colorbar(im, cax=cax, format="%.3f")
        _axes_modify(font_family=font_family, ax=cax, right=True, grid=True)
        cb.outline.set_visible(False)

    inter_ax.tick_params(which="minor", bottom=False, left=False)
    return im


def _pdp_xy(
    pdp_values,
    vmean,
    pdp_ax,
    ticklabels,
    feature_name,
    cmap,
    norm,
    plot_type,
    plot_params,
    y=False,
):
    """PDP isolate on x, y axis

    Parameters
    ----------

    pdp_values: 1-d array
        pdp values
    vmean: float
        threshold to determine the text color
    pdp_ax: matplotlib Axes
        PDP interact axes
    ticklabels: list
        list of tick labels
    feature_name: str
        name of the feature
    cmap: matplotlib color map
    norm: matplotlib color normalize
    y: bool, default=False
        whether it is on y axis
    """

    font_family = plot_params.get("font_family", "Arial")
    fontsize = plot_params.get("inter_fontsize", 9)
    inter_fill_alpha = plot_params.get("inter_fill_alpha", 0.8)

    pdp_ax.imshow(
        np.expand_dims(pdp_values, int(y)),
        cmap=cmap,
        norm=norm,
        origin="lower",
        alpha=inter_fill_alpha,
    )

    for idx in range(len(pdp_values)):
        text_color = "w"
        if pdp_values[idx] >= vmean:
            text_color = "black"

        text_params = {
            "s": round(pdp_values[idx], 3),
            "ha": "center",
            "va": "center",
            "color": text_color,
            "size": fontsize,
            "fontdict": {"family": font_family},
        }
        if y:
            pdp_ax.text(x=0, y=idx, rotation="vertical", **text_params)
        else:
            pdp_ax.text(x=idx, y=0, **text_params)

    pdp_ax.set_frame_on(False)
    pdp_ax.axes.axis("tight")

    if y:
        pdp_ax.set_yticks(range(len(ticklabels)))
        pdp_ax.set_yticklabels(ticklabels)
        pdp_ax.set_ylabel(
            feature_name, fontdict={"family": font_family, "fontsize": 12}
        )
        if plot_type == "contour":
            pdp_ax.get_yaxis().set_label_position("right")
        pdp_ax.get_xaxis().set_visible(False)
    else:
        pdp_ax.set_xticks(range(len(ticklabels)))
        pdp_ax.get_xaxis().tick_top()
        pdp_ax.set_xticklabels(ticklabels)
        pdp_ax.set_xlabel(
            feature_name, fontdict={"family": font_family, "fontsize": 12}
        )
        if plot_type == "grid":
            pdp_ax.get_xaxis().set_label_position("top")
        pdp_ax.get_yaxis().set_visible(False)

    pdp_ax.grid(which="minor", color="w", linestyle="-", linewidth=1)
    pdp_ax.tick_params(which="minor", top=False, left=False)
    pdp_ax.tick_params(
        axis="both", which="major", labelsize=10, labelcolor="#424242", colors="#9E9E9E"
    )


def _pdp_inter_three(
    pdp_interact_out,
    feature_names,
    plot_type,
    chart_grids,
    x_quantile,
    fig,
    plot_params,
):
    """Plot PDP interact with pdp isolate color bar

    Parameters
    ----------
    chart_grids: matplotlib subplot gridspec

    """
    cmap = plot_params.get("cmap", "viridis")
    font_family = plot_params.get("font_family", "Arial")

    pdp_x_ax = fig.add_subplot(chart_grids[1])
    pdp_y_ax = fig.add_subplot(chart_grids[2])
    inter_ax = fig.add_subplot(chart_grids[3], sharex=pdp_x_ax, sharey=pdp_y_ax)

    pdp_x = copy.deepcopy(pdp_interact_out.pdp_isolate_outs[0].pdp)
    pdp_y = copy.deepcopy(pdp_interact_out.pdp_isolate_outs[1].pdp)
    pdp_inter = copy.deepcopy(pdp_interact_out.pdp["preds"].values)
    pdp_values = np.concatenate((pdp_x, pdp_y, pdp_inter))
    pdp_min, pdp_max = np.min(pdp_values), np.max(pdp_values)

    norm = mpl.colors.Normalize(vmin=pdp_min, vmax=pdp_max)
    vmean = norm.vmin + (norm.vmax - norm.vmin) * 0.5
    feature_grids = pdp_interact_out.feature_grids

    pdp_xy_params = {
        "cmap": cmap,
        "norm": norm,
        "vmean": vmean,
        "plot_params": plot_params,
        "plot_type": plot_type,
    }
    _pdp_xy(
        pdp_values=pdp_x,
        pdp_ax=pdp_x_ax,
        ticklabels=pdp_interact_out.pdp_isolate_outs[0].display_columns,
        feature_name=feature_names[0],
        y=False,
        **pdp_xy_params,
    )
    _pdp_xy(
        pdp_values=pdp_y,
        pdp_ax=pdp_y_ax,
        ticklabels=pdp_interact_out.pdp_isolate_outs[1].display_columns,
        feature_name=feature_names[1],
        y=True,
        **pdp_xy_params,
    )

    im = _pdp_inter_one(
        pdp_interact_out=pdp_interact_out,
        feature_names=feature_names,
        plot_type=plot_type,
        inter_ax=inter_ax,
        x_quantile=x_quantile,
        plot_params=plot_params,
        norm=norm,
        ticks=False,
    )

    inter_ax.set_frame_on(False)
    plt.setp(inter_ax.get_xticklabels(), visible=False)
    plt.setp(inter_ax.get_yticklabels(), visible=False)
    inter_ax.tick_params(which="minor", bottom=False, left=False)
    inter_ax.tick_params(which="major", bottom=False, left=False)

    # insert colorbar
    if plot_type == "grid":
        cax = inset_axes(
            inter_ax,
            width="100%",
            height="100%",
            loc="right",
            bbox_to_anchor=(1.05, 0.0, 0.05, 1),
            bbox_transform=inter_ax.transAxes,
            borderpad=0,
        )
        cb_num_grids = np.max(
            [np.min([len(feature_grids[0]), len(feature_grids[1]), 8]), 8]
        )
        boundaries = [
            round(v, 3) for v in np.linspace(norm.vmin, norm.vmax, cb_num_grids)
        ]
        cb = plt.colorbar(im, cax=cax, boundaries=boundaries)
    else:
        cax = inset_axes(inter_ax, width="5%", height="80%", loc="right")
        cb = plt.colorbar(im, cax=cax, format="%.3f")
    _axes_modify(font_family=font_family, ax=cax, right=True, grid=True)
    cb.outline.set_visible(False)

    return {"_pdp_x_ax": pdp_x_ax, "_pdp_y_ax": pdp_y_ax, "_pdp_inter_ax": inter_ax}
