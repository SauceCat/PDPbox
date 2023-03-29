from .utils import (
    _axes_modify,
    _modify_legend_axes,
    _get_string,
    _make_subplots,
    _make_subplots_plotly,
    _display_percentile,
    _to_rgba,
    _get_ticks_plotly,
    _display_percentile_inter,
)
from .styles import _prepare_plot_style
from .pdp_calc_utils import (
    _cluster_ice_lines,
    _sample_ice_lines,
    _prepare_pdp_line_data,
)

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import copy
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import plotly.graph_objects as go


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


def _get_pdp_xticks(pdp_isolate_obj, plot_style):
    dist_xticklabels = line_xticklabels = copy.deepcopy(plot_style.display_columns)
    is_numeric = pdp_isolate_obj.feature_type == "numeric"
    if is_numeric:
        if plot_style.x_quantile:
            line_xticklabels = [_get_string(v) for v in pdp_isolate_obj.feature_grids]
        else:
            dist_xticklabels = line_xticklabels = None
            plot_style.show_percentile = False

    if plot_style.show_percentile and plot_style.engine == "plotly":
        for j, p in enumerate(plot_style.percentile_columns):
            dist_xticklabels[j] += f"<br><sup><b>{p}</b></sup>"

    return is_numeric, line_xticklabels, dist_xticklabels


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

        is_numeric, line_xticklabels, dist_xticklabels = _get_pdp_xticks(
            pdp_isolate_obj, plot_style
        )
        tick_params = {"xlabel": f"class_{t}", "plot_style": plot_style}
        if inner_dist_axes is not None:
            fig.add_subplot(inner_dist_axes)

            if line_xticklabels is not None and not is_numeric:
                line_xticklabels = [""] * len(line_xticklabels)

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

        if len(plot_style.percentile_columns) > 0:
            plot_style.percentile_columns = [
                str(v) for v in pdp_isolate_obj.percentile_grids
            ]
        _display_percentile(inner_line_axes, plot_style)
        line_axes.append(inner_line_axes)

    axes = {"title_axes": title_axes, "line_axes": line_axes, "dist_axes": dist_axes}
    return fig, axes


def _pdp_plot_plotly(pdp_isolate_obj, feat_name, target, plot_params):
    plot_style = _prepare_plot_style(feat_name, target, plot_params, "pdp_isolate")
    nrows, ncols = plot_style.nrows, plot_style.ncols

    row_heights = []
    if plot_style.plot_pts_dist:
        for i in range(nrows):
            row_heights += [7, 0.5]
    else:
        row_heights = [7] * nrows
    row_heights = [v / sum(row_heights) for v in row_heights]

    plot_args = {
        "rows": len(row_heights),
        "cols": ncols,
        "horizontal_spacing": plot_style.horizontal_spacing,
        "vertical_spacing": plot_style.vertical_spacing,
        "row_heights": row_heights,
    }
    fig = _make_subplots_plotly(plot_args, plot_style)

    cmaps = plot_style.line["cmaps"]
    for i, t in enumerate(target):
        cmap = cmaps[i % len(cmaps)]
        colors = [cmap, plt.get_cmap(cmap)(0.1), plt.get_cmap(cmap)(1.0)]
        grids = {"col": i % ncols + 1, "row": i // ncols + 1}
        line_grids = {"col": grids["col"], "row": grids["row"] * 2 - 1}
        dist_grids = {"col": grids["col"], "row": grids["row"] * 2}

        line_traces = _pdp_line_plot(
            t,
            pdp_isolate_obj,
            colors,
            plot_style,
            axes=None,
        )
        for trace in line_traces:
            if trace is not None:
                fig.add_trace(trace, **line_grids)

        is_numeric, line_xticklabels, dist_xticklabels = _get_pdp_xticks(
            pdp_isolate_obj, plot_style
        )
        if plot_style.plot_pts_dist:
            dist_trace = _pdp_dist_plot_plotly(pdp_isolate_obj, colors, plot_style)
            fig.add_trace(dist_trace, **dist_grids)

            if line_xticklabels is not None and not is_numeric:
                line_xticklabels = [""] * len(line_xticklabels)
            _set_pdp_xticks_plotly(
                fig, "", line_xticklabels, line_grids, is_numeric_line=is_numeric
            )
            _set_pdp_xticks_plotly(fig, f"class_{t}", dist_xticklabels, dist_grids)
            fig.update_yaxes(
                showticklabels=False,
                **dist_grids,
            )

            if is_numeric and not plot_style.x_quantile:
                fig.update_xaxes(showgrid=False, showticklabels=False, **dist_grids)
                fig.update_yaxes(showgrid=False, **dist_grids)
        else:
            _set_pdp_xticks_plotly(fig, f"class_{t}", line_xticklabels, line_grids)

    return fig


def _set_pdp_xticks_plotly(fig, xlabel, xticklabels, grids, is_numeric_line=False):
    if xticklabels is not None:
        xticks = np.arange(len(xticklabels))
        if is_numeric_line:
            x_range = [xticks[0], xticks[-1]]
        else:
            x_range = [xticks[0] - 0.5, xticks[-1] + 0.5]
        fig.update_xaxes(
            title_text=xlabel,
            ticktext=xticklabels,
            tickvals=np.arange(len(xticklabels)),
            range=x_range,
            **grids,
        )
    else:
        fig.update_xaxes(
            title_text=xlabel,
            **grids,
        )


def _pdp_line_plot(
    class_id,
    pdp_isolate_obj,
    colors,
    plot_style,
    axes,
):
    cmap, light_color, dark_color = colors
    x, pdp, line_data, line_std = _prepare_pdp_line_data(
        class_id,
        pdp_isolate_obj,
        plot_style,
    )
    ice_line_traces = []
    if plot_style.plot_lines:
        ice_line_traces = _ice_line_plot(x, line_data, cmap, axes, plot_style.engine)

    std_params = {
        "x": x,
        "pdp": pdp,
        "std": line_std,
        "colors": (light_color, dark_color),
        "plot_style": plot_style,
    }
    if plot_style.engine == "matplotlib":
        _pdp_std_plot(axes=axes, **std_params)
        _axes_modify(axes, plot_style)
    else:
        std_traces = _pdp_std_plot_plotly(axes=None, **std_params)
        return ice_line_traces + std_traces


def _pdp_std_plot(x, pdp, std, colors, plot_style, axes):
    upper = pdp + std
    lower = pdp - std
    light_color, dark_color = colors
    line_style = plot_style.line

    if plot_style.std_fill:
        axes.fill_between(
            x, upper, lower, alpha=line_style["fill_alpha"], color=light_color
        )

    if plot_style.pdp_hl:
        axes.plot(
            x,
            pdp,
            color=line_style["hl_color"],
            linewidth=line_style["width"] * 3,
            alpha=line_style["hl_alpha"],
        )

    axes.plot(
        x,
        [0] * len(pdp),
        linestyle="--",
        linewidth=line_style["width"],
        color=line_style["zero_color"],
    )
    axes.plot(
        x,
        pdp,
        color=dark_color,
        linewidth=line_style["width"],
        marker="o",
        markersize=line_style["markersize"],
    )


def _pdp_std_plot_plotly(x, pdp, std, colors, plot_style, axes):
    upper = pdp + std
    lower = pdp - std
    light_color, dark_color = colors
    line_style = plot_style.line
    light_color = _to_rgba(light_color, line_style["fill_alpha"])
    dark_color = _to_rgba(dark_color)
    trace_params = {"x": x, "mode": "lines", "hoverinfo": "none"}

    fill_traces = [None, None]
    if plot_style.std_fill:
        trace_upper = go.Scatter(
            y=upper,
            line=dict(color=light_color),
            **trace_params,
        )
        trace_lower = go.Scatter(
            y=lower,
            line=dict(color=light_color),
            fill="tonexty",
            fillcolor=light_color,
            **trace_params,
        )
        fill_traces = [trace_upper, trace_lower]

    pdp_hl_trace = None
    if plot_style.pdp_hl:
        pdp_hl_trace = go.Scatter(
            y=pdp,
            line=dict(color=line_style["hl_color"], width=line_style["width"] * 3),
            **trace_params,
        )

    zero_trace = go.Scatter(
        y=[0] * len(pdp),
        line=dict(
            color=line_style["zero_color"], width=line_style["width"], dash="dash"
        ),
        **trace_params,
    )
    pdp_trace = go.Scatter(
        x=x,
        y=pdp,
        mode="lines+markers",
        line=dict(color=dark_color, width=line_style["width"]),
        marker=dict(color=dark_color, size=line_style["markersize"]),
        name="pdp",
    )

    return fill_traces + [pdp_hl_trace, zero_trace, pdp_trace]


def _draw_pdp_distplot(data, color, axes, plot_style):
    axes.plot(
        data,
        [1] * len(data),
        "|",
        color=color,
        markersize=plot_style.dist["markersize"],
    )
    _modify_legend_axes(axes, plot_style.font_family)


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


def _draw_pdp_countplot_plotly(data, cmap, plot_style):
    count_norm = data["count_norm"].values
    xticks = data["x"].values

    heatmap = go.Heatmap(
        x=xticks,
        y=[0],
        z=[count_norm],
        text=[[round(v, 3) for v in count_norm]],
        texttemplate="%{text}",
        colorscale=cmap,
        zmin=0,
        zmax=np.max(count_norm),
        opacity=plot_style.dist["fill_alpha"],
        showscale=False,
        xgap=2,
        name="dist",
    )
    return heatmap


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


def _pdp_dist_plot_plotly(pdp_isolate_obj, colors, plot_style):
    cmap, light_color, dark_color = colors

    if pdp_isolate_obj.feature_type == "numeric" and not plot_style.x_quantile:
        data = pdp_isolate_obj.dist_data
        dist_trace = go.Scatter(
            x=data,
            y=[1] * len(data),
            mode="text",
            text=["|"] * len(data),
            textposition="middle center",
            textfont=dict(
                size=plot_style.dist["markersize"], color=_to_rgba(dark_color)
            ),
            showlegend=False,
            name="dist",
            hoverinfo="none",
        )
    else:
        dist_trace = _draw_pdp_countplot_plotly(
            pdp_isolate_obj.count_data, cmap, plot_style
        )
    return dist_trace


def _ice_line_plot(x, lines, cmap, axes, engine):
    total = len(lines)
    linewidth = np.max([1.0 / np.log10(total), 0.3])
    linealpha = np.min([np.max([1.0 / np.log10(total), 0.3]), 0.8])
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, 20))[5:15]

    if engine == "matplotlib":
        for i, line in enumerate(lines.values):
            axes.plot(x, line, linewidth=linewidth, c=colors[i % 10], alpha=linealpha)
    else:
        colors = [_to_rgba(v, linealpha) for v in colors]
        traces = []
        for i, line in enumerate(lines.values):
            traces.append(
                go.Scatter(
                    x=x,
                    y=line,
                    mode="lines",
                    line=dict(color=colors[i % 10], width=linewidth),
                    hoverinfo="none",
                )
            )

        return traces


def _pdp_inter_plot(pdp_inter_obj, feat_names, target, plot_params):
    plot_style = _prepare_plot_style(feat_names, target, plot_params, "pdp_interact")
    fig, inner_grid, title_axes = _make_subplots(plot_style)
    pdp_iso_obj_x, pdp_iso_obj_y = pdp_inter_obj.pdp_isolate_objs
    grids_x, grids_y = pdp_inter_obj.feature_grids

    interact_axes, isolate_axes = [], []
    cmap = plot_style.interact["cmap"]
    for i, t in enumerate(target):
        pdp_x = copy.deepcopy(pdp_iso_obj_x.results[t].pdp)
        pdp_y = copy.deepcopy(pdp_iso_obj_y.results[t].pdp)
        pdp_inter = copy.deepcopy(pdp_inter_obj.results[t].pdp)

        if plot_style.plot_pdp:
            inner = GridSpecFromSubplotSpec(
                2,
                2,
                subplot_spec=inner_grid[i],
                wspace=0.2
                if plot_style.show_percentile
                and not plot_style.x_quantile
                and plot_style.plot_pdp
                else 0.1,
                hspace=0.2,
                height_ratios=[0.5, 7],
                width_ratios=[0.5, 7],
            )
            inner_iso_x_axes = plt.subplot(inner[1])
            inner_iso_y_axes = plt.subplot(inner[2])
            inner_inter_axes = plt.subplot(inner[3])
            pdp_values = np.concatenate((pdp_x, pdp_y, pdp_inter))
        else:
            inner_inter_axes = plt.subplot(inner_grid[i])
            inner_iso_x_axes = None
            inner_iso_y_axes = None
            pdp_values = pdp_inter

        interact_axes.append(inner_inter_axes)
        isolate_axes.append([inner_iso_x_axes, inner_iso_y_axes])

        pdp_min, pdp_max = np.min(pdp_values), np.max(pdp_values)
        norm = mpl.colors.Normalize(vmin=pdp_min, vmax=pdp_max)
        vmean = norm.vmin + (norm.vmax - norm.vmin) * 0.5

        fig.add_subplot(inner_inter_axes)
        im = _pdp_inter_xy_plot(
            pdp_inter,
            pdp_inter_obj.feature_grids,
            norm,
            cmap,
            plot_style,
            inner_inter_axes,
        )

        iso_plot_params = {
            "vmean": vmean,
            "norm": norm,
            "cmap": cmap,
            "plot_style": plot_style,
        }
        cb_num_grids = np.max([np.min([len(grids_x), len(grids_y), 8]), 8])
        boundaries = [
            round(v, 3) for v in np.linspace(norm.vmin, norm.vmax, cb_num_grids)
        ]

        if inner_iso_x_axes is None and inner_iso_y_axes is None:
            _set_xy_ticks(
                feat_names[0],
                plot_style.display_columns[0],
                plot_style,
                inner_inter_axes,
                xy=True,
                y=False,
            )
            _set_xy_ticks(
                feat_names[1],
                plot_style.display_columns[1],
                plot_style,
                inner_inter_axes,
                xy=True,
                y=True,
            )
            _display_percentile_inter(plot_style, inner_inter_axes, x=True, y=True)
        else:
            _pdp_iso_xy_plot(pdp_x, axes=inner_iso_x_axes, y=False, **iso_plot_params)
            fig.add_subplot(inner_iso_x_axes)
            _set_xy_ticks(
                feat_names[0],
                plot_style.display_columns[0],
                plot_style,
                inner_iso_x_axes,
                xy=False,
                y=False,
            )
            _display_percentile_inter(plot_style, inner_iso_x_axes, x=True, y=False)

            _pdp_iso_xy_plot(pdp_y, axes=inner_iso_y_axes, y=True, **iso_plot_params)
            fig.add_subplot(inner_iso_y_axes)
            _set_xy_ticks(
                feat_names[1],
                plot_style.display_columns[1],
                plot_style,
                inner_iso_y_axes,
                xy=False,
                y=True,
            )
            _display_percentile_inter(plot_style, inner_iso_y_axes, x=False, y=True)
            inner_inter_axes.grid(False)
            inner_inter_axes.get_xaxis().set_visible(False)
            inner_inter_axes.get_yaxis().set_visible(False)

        cax = inset_axes(
            inner_inter_axes,
            width="100%",
            height="100%",
            loc="right",
            bbox_to_anchor=(1.05, 0.0, 0.03, 1),
            bbox_transform=inner_inter_axes.transAxes,
            borderpad=0,
        )
        cb = plt.colorbar(im, cax=cax, boundaries=boundaries)
        cb.ax.tick_params(**plot_style.tick["tick_params"])
        cb.outline.set_visible(False)

    axes = {
        "title_axes": title_axes,
        "interact_axes": interact_axes,
        "isolate_axes": isolate_axes,
    }
    return fig, axes


def _pdp_inter_plot_plotly(pdp_inter_obj, feat_names, target, plot_params):
    plot_style = _prepare_plot_style(feat_names, target, plot_params, "pdp_interact")
    nrows, ncols = plot_style.nrows, plot_style.ncols
    pdp_iso_obj_x, pdp_iso_obj_y = pdp_inter_obj.pdp_isolate_objs
    grids_x, grids_y = pdp_inter_obj.feature_grids

    row_heights = []
    column_widths = []
    if plot_style.plot_pdp:
        for i in range(nrows):
            row_heights += [0.5, 7]
        for i in range(ncols):
            column_widths += [0.5, 7]
    else:
        row_heights = [7] * nrows
        column_widths = [7] * ncols

    row_heights = [v / sum(row_heights) for v in row_heights]
    column_widths = [v / sum(column_widths) for v in column_widths]

    plot_args = {
        "rows": len(row_heights),
        "cols": len(column_widths),
        "horizontal_spacing": plot_style.horizontal_spacing,
        "vertical_spacing": plot_style.vertical_spacing,
        "row_heights": row_heights,
        "column_widths": column_widths,
    }
    fig = _make_subplots_plotly(plot_args, plot_style)

    cmap = plot_style.interact["cmap"]
    for i, t in enumerate(target):
        pdp_x = copy.deepcopy(pdp_iso_obj_x.results[t].pdp)
        pdp_y = copy.deepcopy(pdp_iso_obj_y.results[t].pdp)
        pdp_inter = copy.deepcopy(pdp_inter_obj.results[t].pdp)

        grids = {"col": i % ncols + 1, "row": i // ncols + 1}
        inter_grids = copy.deepcopy(grids)
        iso_x_grids, iso_y_grids = None, None
        pdp_values = pdp_inter
        if plot_style.plot_pdp:
            inter_grids = {"col": grids["col"] * 2, "row": grids["row"] * 2}
            iso_x_grids = {"col": grids["col"] * 2, "row": grids["row"] * 2 - 1}
            iso_y_grids = {"col": grids["col"] * 2 - 1, "row": grids["row"] * 2}
            pdp_values = np.concatenate((pdp_x, pdp_y, pdp_inter))

        pdp_min, pdp_max = np.min(pdp_values), np.max(pdp_values)

        inter_trace = _pdp_inter_xy_plot_plotly(
            pdp_inter,
            pdp_inter_obj.feature_grids,
            cmap,
            plot_style,
            (
                np.sum(column_widths[: inter_grids["col"]]),
                np.sum(row_heights[: inter_grids["row"]]),
                1.0 / nrows,
            ),
            (pdp_min, pdp_max),
        )
        fig.add_trace(inter_trace, **inter_grids)

        if iso_x_grids is not None:
            iso_x_trace = _pdp_iso_xy_plot_plotly(
                pdp_x,
                cmap,
                plot_style,
                (pdp_min, pdp_max),
                yaxes=False,
            )
            fig.add_trace(iso_x_trace, **iso_x_grids)

        if iso_y_grids is not None:
            iso_y_trace = _pdp_iso_xy_plot_plotly(
                pdp_y,
                cmap,
                plot_style,
                (pdp_min, pdp_max),
                yaxes=True,
            )
            fig.add_trace(iso_y_trace, **iso_y_grids)

        xlabel, xticklabels = _get_ticks_plotly(feat_names[0], plot_style, idx=0)
        ylabel, yticklabels = _get_ticks_plotly(feat_names[1], plot_style, idx=1)
        feat_type_x, feat_type_y = pdp_inter_obj.feature_types
        if plot_style.plot_pdp:
            _set_xy_ticks_plotly(feat_names[0], [], inter_grids, fig, xy=True, y=False)
            _set_xy_ticks_plotly("", xticklabels, iso_x_grids, fig, xy=False, y=False)
            _set_xy_ticks_plotly(
                feat_names[1], yticklabels, iso_y_grids, fig, xy=False, y=True
            )

            fig.update_xaxes(showgrid=False, showticklabels=False, **inter_grids)
            fig.update_yaxes(showgrid=False, showticklabels=False, **inter_grids)
        else:
            if feat_type_x == "numeric" and not plot_style.x_quantile:
                xticklabels = None
            if feat_type_y == "numeric" and not plot_style.x_quantile:
                yticklabels = None

            _set_xy_ticks_plotly(
                feat_names[0], xticklabels, inter_grids, fig, xy=True, y=False
            )
            _set_xy_ticks_plotly(
                feat_names[1], yticklabels, inter_grids, fig, xy=True, y=True
            )

    return fig


def _pdp_contour_plot(X, Y, pdp_mx, norm, cmap, plot_style, axes):
    """Interact contour plot"""
    level = np.min([X.shape[0], X.shape[1]])
    c1 = axes.contourf(
        X,
        Y,
        pdp_mx,
        N=level,
        origin="lower",
        cmap=cmap,
        norm=norm,
        alpha=plot_style.interact["fill_alpha"],
    )
    c2 = axes.contour(c1, levels=c1.levels, colors="white", origin="lower")
    axes.clabel(c2, fontsize=plot_style.interact["font_size"], inline=1)
    axes.set_aspect("auto")

    # return the color mapping object for colorbar
    return c1


def _pdp_inter_grid(pdp_mx, norm, cmap, plot_style, axes):
    """Interact grid plot (heatmap)"""
    im = axes.imshow(
        pdp_mx,
        cmap=cmap,
        norm=norm,
        origin="lower",
        aspect="auto",
        alpha=plot_style.interact["fill_alpha"],
    )
    R, C = pdp_mx.shape

    for r in range(R):
        for c in range(C):
            text_color = "w"
            if pdp_mx[r, c] >= norm.vmin + (norm.vmax - norm.vmin) * 0.5:
                text_color = "black"
            # column -> x, row -> y
            axes.text(
                c,
                r,
                round(pdp_mx[r, c], 3),
                ha="center",
                va="center",
                color=text_color,
                size=plot_style.interact["font_size"],
            )

    axes.set_xticks(np.arange(C - 1) + 0.5, minor=True)
    axes.set_yticks(np.arange(R - 1) + 0.5, minor=True)

    return im


def _pdp_inter_xy_plot(
    pdp_inter,
    feature_grids,
    norm,
    cmap,
    plot_style,
    axes,
):
    """Plot single PDP interact"""
    grids_x, grids_y = feature_grids
    pdp_mx = pdp_inter.reshape((len(grids_x), len(grids_y))).T

    if plot_style.interact["type"] == "grid":
        im = _pdp_inter_grid(pdp_mx, norm, cmap, plot_style, axes)
    else:
        if plot_style.x_quantile:
            X, Y = np.meshgrid(range(pdp_mx.shape[1]), range(pdp_mx.shape[0]))
        else:
            X, Y = np.meshgrid(grids_x, grids_y)
        im = _pdp_contour_plot(X, Y, pdp_mx, norm, cmap, plot_style, axes)

    # axes.grid(which="minor", color="w", linestyle="-", linewidth=1)
    plt.setp(axes.get_xticklabels(), visible=False)
    plt.setp(axes.get_yticklabels(), visible=False)
    axes.tick_params(which="minor", bottom=False, left=False)
    axes.tick_params(which="major", bottom=False, left=False)
    axes.set_frame_on(False)
    _axes_modify(axes, plot_style)

    return im


def _pdp_inter_xy_plot_plotly(
    pdp_inter,
    feature_grids,
    cmap,
    plot_style,
    cb_xyz,
    v_range,
):
    """Plot single PDP interact"""
    grids_x, grids_y = feature_grids
    pdp_mx = pdp_inter.reshape((len(grids_x), len(grids_y))).T
    cb_x, cb_y, cb_z = cb_xyz

    plot_params = dict(
        z=pdp_mx,
        colorscale=cmap,
        opacity=plot_style.interact["fill_alpha"],
        showscale=True,
        colorbar=dict(len=cb_z, yanchor="top", y=cb_y, x=cb_x, xanchor="left"),
        zmin=v_range[0],
        zmax=v_range[1],
        name="pdp interact",
    )

    if plot_style.interact["type"] == "grid":
        trace = go.Heatmap(
            x=np.arange(len(grids_x)),
            y=np.arange(len(grids_y)),
            text=np.array([["{:.3f}".format(v) for v in row] for row in pdp_mx]),
            texttemplate="%{text}",
            **plot_params,
        )
    else:
        plot_params.update(
            dict(
                line=dict(width=1, color="white"),
                contours=dict(showlabels=True),
            )
        )
        if plot_style.x_quantile:
            trace = go.Contour(
                x=np.arange(len(grids_x)),
                y=np.arange(len(grids_y)),
                **plot_params,
            )
        else:
            trace = go.Contour(
                x=grids_x,
                y=grids_y,
                **plot_params,
            )
    return trace


def _pdp_iso_xy_plot(
    pdp_values,
    vmean,
    norm,
    cmap,
    plot_style,
    axes,
    y=False,
):
    axes.imshow(
        np.expand_dims(pdp_values, int(y)),
        cmap=cmap,
        norm=norm,
        origin="lower",
        alpha=plot_style.isolate["fill_alpha"],
    )

    for i in range(len(pdp_values)):
        text_color = "w"
        if pdp_values[i] >= vmean:
            text_color = "black"

        text_params = {
            "s": round(pdp_values[i], 3),
            "ha": "center",
            "va": "center",
            "color": text_color,
            "size": plot_style.isolate["font_size"],
            "fontdict": {"family": plot_style.font_family},
        }
        if y:
            axes.text(x=0, y=i, rotation="vertical", **text_params)
        else:
            axes.text(x=i, y=0, **text_params)

    axes.set_frame_on(False)
    axes.axes.axis("tight")


def _pdp_iso_xy_plot_plotly(
    pdp_values,
    cmap,
    plot_style,
    v_range,
    yaxes=False,
):
    ticks = np.arange(len(pdp_values))
    text = np.array([["{:.3f}".format(v) for v in pdp_values]])
    pdp_values = pdp_values.reshape((1, -1))
    x, y = ticks, [0]

    if yaxes:
        x, y = [0], ticks
        pdp_values = pdp_values.T
        text = text.T

    heatmap = go.Heatmap(
        x=x,
        y=y,
        z=pdp_values,
        text=text,
        texttemplate="%{text}",
        colorscale=cmap,
        opacity=plot_style.interact["fill_alpha"],
        zmin=v_range[0],
        zmax=v_range[1],
        showscale=False,
        name="pdp isolate",
    )
    return heatmap


def _set_xy_ticks(feat_name, ticklabels, plot_style, axes, xy=False, y=False):
    fontdict = {
        "family": plot_style.font_family,
        "fontsize": plot_style.title["subplot_title"]["font_size"],
    }

    if y:
        axes.set_ylabel(feat_name, fontdict=fontdict)
    else:
        axes.set_xlabel(feat_name, fontdict=fontdict)
    _axes_modify(axes, plot_style)
    axes.tick_params(axis="both", which="minor", length=0)
    axes.grid(False)

    if xy and plot_style.interact["type"] == "contour" and not plot_style.x_quantile:
        return

    if y:
        axes.set_yticks(range(len(ticklabels)), ticklabels)
        if not xy:
            axes.get_xaxis().set_visible(False)
    else:
        if not plot_style.show_percentile and not xy:
            axes.xaxis.set_label_position("top")
        axes.set_xticks(range(len(ticklabels)), ticklabels)
        if not xy:
            axes.get_yaxis().set_visible(False)


def _set_xy_ticks_plotly(feat_name, ticklabels, grids, fig, xy=False, y=False):
    tick_params = dict(
        title_text=feat_name,
    )
    if ticklabels is not None:
        tick_params.update(
            dict(ticktext=ticklabels, tickvals=np.arange(len(ticklabels)))
        )
    tick_params.update(grids)

    if y:
        fig.update_yaxes(**tick_params)
        if not xy:
            fig.update_xaxes(showgrid=False, showticklabels=False, **grids)
    else:
        if not xy:
            tick_params.update({"side": "top"})
        fig.update_xaxes(**tick_params)
        if not xy:
            fig.update_yaxes(showgrid=False, showticklabels=False, **grids)
