from .utils import (
    _get_string,
    _to_rgba,
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


def _set_pdp_xticks(xticklabels, plot_style, axes, is_numeric_line=False):
    if xticklabels is not None:
        xticks = np.arange(len(xticklabels))
        if is_numeric_line:
            axes.set_xlim(xticks[0], xticks[-1])
        else:
            axes.set_xlim(xticks[0] - 0.5, xticks[-1] + 0.5)

        axes.set_xticks(xticks)
        axes.set_xticklabels(xticklabels, rotation=plot_style.tick["xticks_rotation"])


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
                wspace=plot_style.gaps["inner_x"],
                hspace=plot_style.gaps["inner_y"],
                height_ratios=plot_style.subplot_ratio["y"],
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
        if plot_style.plot_pts_dist:
            fig.add_subplot(inner_dist_axes)

            if line_xticklabels is not None and not is_numeric:
                line_xticklabels = [""] * len(line_xticklabels)

            _set_pdp_xticks(line_xticklabels, plot_style, inner_line_axes, is_numeric)
            _set_pdp_xticks(
                dist_xticklabels,
                plot_style,
                inner_dist_axes,
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
            _set_pdp_xticks(line_xticklabels, plot_style, inner_line_axes, is_numeric)

        inner_line_axes.set_title(
            f"class_{t}",
            **plot_style.title["subplot_title"],
        )

        if len(plot_style.percentile_columns) > 0:
            plot_style.percentile_columns = [
                str(v) for v in pdp_isolate_obj.percentile_grids
            ]
        _display_percentile(inner_line_axes, plot_style)
        line_axes.append(inner_line_axes)

    axes = {"title_axes": title_axes, "line_axes": line_axes, "dist_axes": dist_axes}
    return fig, axes


def _update_pdp_domains(fig, nc, nr, grids, title_text, plot_style):
    plot_sizes = plot_style.plot_sizes
    gaps = plot_style.gaps
    group_w = plot_sizes["group_w"] + gaps["outer_x"]
    group_h = plot_sizes["group_h"] + gaps["outer_y"]
    line_grids, dist_grids = grids

    line_x0, line_y0 = group_w * (nc - 1), group_h * (nr - 1) + gaps["top"]
    line_domain_x = [line_x0, line_x0 + plot_sizes["line_w"]]
    line_domain_y = [1.0 - (line_y0 + plot_sizes["line_h"]), 1.0 - line_y0]
    line_domain_x = [np.clip(v, 0, 1) for v in line_domain_x]
    line_domain_y = [np.clip(v, 0, 1) for v in line_domain_y]

    fig.update_xaxes(domain=line_domain_x, **line_grids)
    fig.update_yaxes(domain=line_domain_y, **line_grids)

    if plot_style.plot_pts_dist:
        dist_y0 = line_y0 + plot_sizes["line_h"] + gaps["inner_y"]
        fig.update_xaxes(domain=line_domain_x, **dist_grids)
        dist_domain_y = [1.0 - (dist_y0 + plot_sizes["dist_h"]), 1.0 - dist_y0]
        dist_domain_y = [np.clip(v, 0, 1) for v in dist_domain_y]
        fig.update_yaxes(
            domain=dist_domain_y,
            **dist_grids,
        )

    title = go.layout.Annotation(
        x=sum(line_domain_x) / 2,
        y=line_domain_y[1],
        xref="paper",
        yref="paper",
        text=title_text,
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
    )
    return title


def _pdp_plot_plotly(pdp_isolate_obj, feat_name, target, plot_params):
    plot_style = _prepare_plot_style(feat_name, target, plot_params, "pdp_isolate")
    nrows, ncols = plot_style.nrows, plot_style.ncols
    plot_args = {
        "rows": nrows * (2 if plot_style.plot_pts_dist else 1),
        "cols": ncols,
        "horizontal_spacing": 0,
        "vertical_spacing": 0,
    }
    fig = _make_subplots_plotly(plot_args, plot_style)
    subplot_titles = []

    cmaps = plot_style.line["cmaps"]
    for i, t in enumerate(target):
        cmap = cmaps[i % len(cmaps)]
        colors = [cmap, plt.get_cmap(cmap)(0.1), plt.get_cmap(cmap)(1.0)]
        grids = {"col": i % ncols + 1, "row": i // ncols + 1}
        line_grids = {"col": grids["col"], "row": grids["row"] * 2 - 1}
        dist_grids = {"col": grids["col"], "row": grids["row"] * 2}
        nr, nc = grids["row"], grids["col"]
        title = _update_pdp_domains(
            fig, nc, nr, (line_grids, dist_grids), f"class_{t}", plot_style
        )
        subplot_titles.append(title)

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
            _set_pdp_xticks_plotly(fig, "", dist_xticklabels, dist_grids)
            fig.update_yaxes(
                showticklabels=False,
                **dist_grids,
            )

            if is_numeric and not plot_style.x_quantile:
                fig.update_xaxes(showgrid=False, showticklabels=False, **dist_grids)
                fig.update_yaxes(showgrid=False, **dist_grids)
        else:
            _set_pdp_xticks_plotly(fig, "", line_xticklabels, line_grids)

    fig.update_layout(annotations=subplot_titles)

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
                wspace=plot_style.gaps["inner_x"],
                hspace=plot_style.gaps["inner_y"],
                height_ratios=plot_style.subplot_ratio["y"],
                width_ratios=plot_style.subplot_ratio["x"],
            )
            iso_x_axes = plt.subplot(inner[1])
            iso_y_axes = plt.subplot(inner[2])
            inter_axes = plt.subplot(inner[3])
            pdp_values = np.concatenate((pdp_x, pdp_y, pdp_inter))
        else:
            inter_axes = plt.subplot(inner_grid[i])
            iso_x_axes = None
            iso_y_axes = None
            pdp_values = pdp_inter

        interact_axes.append(inter_axes)
        isolate_axes.append([iso_x_axes, iso_y_axes])

        pdp_min, pdp_max = np.min(pdp_values), np.max(pdp_values)
        norm = mpl.colors.Normalize(vmin=pdp_min, vmax=pdp_max)
        vmean = norm.vmin + (norm.vmax - norm.vmin) * 0.5

        fig.add_subplot(inter_axes)
        im = _pdp_inter_xy_plot(
            pdp_inter,
            pdp_inter_obj.feature_grids,
            norm,
            cmap,
            plot_style,
            inter_axes,
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
        if plot_style.plot_pdp:
            _pdp_iso_xy_plot(pdp_x, axes=iso_x_axes, y=False, **iso_plot_params)
            fig.add_subplot(iso_x_axes)
            _set_xy_ticks(
                feat_names[0],
                plot_style.display_columns[0],
                plot_style,
                iso_x_axes,
                xy=False,
                y=False,
            )
            _display_percentile_inter(plot_style, iso_x_axes, x=True, y=False)

            _pdp_iso_xy_plot(pdp_y, axes=iso_y_axes, y=True, **iso_plot_params)
            fig.add_subplot(iso_y_axes)
            _set_xy_ticks(
                feat_names[1],
                plot_style.display_columns[1],
                plot_style,
                iso_y_axes,
                xy=False,
                y=True,
            )
            _display_percentile_inter(plot_style, iso_y_axes, x=False, y=True)
            inter_axes.grid(False)
            inter_axes.get_xaxis().set_visible(False)
            inter_axes.get_yaxis().set_visible(False)

            iso_x_axes.set_title(
                f"target_{t}",
                **plot_style.title["subplot_title"],
            )
        else:
            _set_xy_ticks(
                feat_names[0],
                plot_style.display_columns[0],
                plot_style,
                inter_axes,
                xy=True,
                y=False,
            )
            _set_xy_ticks(
                feat_names[1],
                plot_style.display_columns[1],
                plot_style,
                inter_axes,
                xy=True,
                y=True,
            )
            _display_percentile_inter(plot_style, inter_axes, x=True, y=True)
            inter_axes.set_title(f"target_{t}", **plot_style.title["subplot_title"])

        cax = inset_axes(
            inter_axes,
            width="100%",
            height="100%",
            loc="right",
            bbox_to_anchor=(
                1.08
                if plot_style.show_percentile and not plot_style.plot_pdp
                else 1.02,
                0.0,
                0.03,
                1,
            ),
            bbox_transform=inter_axes.transAxes,
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


def _update_pdp_inter_domains(fig, nc, nr, grids, plot_style):
    plot_sizes = plot_style.plot_sizes
    gaps = plot_style.gaps
    group_w = plot_sizes["group_w"] + gaps["outer_x"]
    group_h = plot_sizes["group_h"] + gaps["outer_y"]
    iso_x_grids, iso_y_grids, inter_grids = grids

    inter_x0, inter_y0 = group_w * (nc - 1), group_h * (nr - 1) + gaps["top"]
    if plot_style.plot_pdp:
        inter_x0 += plot_sizes["iso_y_w"] + gaps["inner_x"]
        inter_y0 += plot_sizes["iso_x_h"] + gaps["inner_y"]
    inter_domain_x = [inter_x0, inter_x0 + plot_sizes["inter_w"]]
    inter_domain_y = [1.0 - (inter_y0 + plot_sizes["inter_h"]), 1.0 - inter_y0]
    inter_domain_x = [np.clip(v, 0, 1) for v in inter_domain_x]
    inter_domain_y = [np.clip(v, 0, 1) for v in inter_domain_y]

    fig.update_xaxes(domain=inter_domain_x, **inter_grids)
    fig.update_yaxes(domain=inter_domain_y, **inter_grids)

    if plot_style.plot_pdp:
        iso_x_y0 = group_h * (nr - 1) + gaps["top"]
        fig.update_xaxes(domain=inter_domain_x, **iso_x_grids)
        iso_x_domain_y = [1.0 - (iso_x_y0 + plot_sizes["iso_x_h"]), 1.0 - iso_x_y0]
        iso_x_domain_y = [np.clip(v, 0, 1) for v in iso_x_domain_y]
        fig.update_yaxes(
            domain=iso_x_domain_y,
            **iso_x_grids,
        )

        iso_y_x0 = group_w * (nc - 1)
        iso_y_domain_x = [iso_y_x0, iso_y_x0 + plot_sizes["iso_y_w"]]
        iso_y_domain_x = [np.clip(v, 0, 1) for v in iso_y_domain_x]
        fig.update_xaxes(domain=iso_y_domain_x, **iso_y_grids)
        fig.update_yaxes(domain=inter_domain_y, **iso_y_grids)

    cb_xyz = (
        inter_domain_x[1] + plot_style.gaps["inner_x"] / 2,
        inter_domain_y[0],
        plot_style.plot_sizes["inter_h"],
    )

    return cb_xyz


def _get_pdp_inter_grids(i, plot_style):
    grids = {"col": i % plot_style.ncols + 1, "row": i // plot_style.ncols + 1}
    inter_grids = copy.deepcopy(grids)
    iso_x_grids, iso_y_grids = None, None

    if plot_style.plot_pdp:
        inter_grids = {"col": grids["col"] * 2, "row": grids["row"] * 2}
        iso_x_grids = {"col": grids["col"] * 2, "row": grids["row"] * 2 - 1}
        iso_y_grids = {"col": grids["col"] * 2 - 1, "row": grids["row"] * 2}

    return grids, inter_grids, iso_x_grids, iso_y_grids


def _pdp_inter_plot_plotly(pdp_inter_obj, feat_names, target, plot_params):
    plot_style = _prepare_plot_style(feat_names, target, plot_params, "pdp_interact")
    nrows, ncols = plot_style.nrows, plot_style.ncols
    pdp_iso_obj_x, pdp_iso_obj_y = pdp_inter_obj.pdp_isolate_objs
    grids_x, grids_y = pdp_inter_obj.feature_grids

    plot_args = {
        "rows": nrows * (2 if plot_style.plot_pdp else 1),
        "cols": ncols * (2 if plot_style.plot_pdp else 1),
        "horizontal_spacing": 0,
        "vertical_spacing": 0,
    }
    fig = _make_subplots_plotly(plot_args, plot_style)

    cmap = plot_style.interact["cmap"]
    for i, t in enumerate(target):
        pdp_x = copy.deepcopy(pdp_iso_obj_x.results[t].pdp)
        pdp_y = copy.deepcopy(pdp_iso_obj_y.results[t].pdp)
        pdp_inter = copy.deepcopy(pdp_inter_obj.results[t].pdp)

        grids, inter_grids, iso_x_grids, iso_y_grids = _get_pdp_inter_grids(
            i, plot_style
        )
        nc, nr = grids["col"], grids["row"]

        pdp_values = pdp_inter
        if plot_style.plot_pdp:
            pdp_values = np.concatenate((pdp_x, pdp_y, pdp_inter))
        pdp_min, pdp_max = np.min(pdp_values), np.max(pdp_values)

        cb_xyz = _update_pdp_inter_domains(
            fig, nc, nr, (iso_x_grids, iso_y_grids, inter_grids), plot_style
        )
        inter_trace = _pdp_inter_xy_plot_plotly(
            pdp_inter,
            pdp_inter_obj.feature_grids,
            cmap,
            plot_style,
            cb_xyz,
            (pdp_min, pdp_max),
        )
        fig.add_trace(inter_trace, **inter_grids)

        _, xticklabels = _get_ticks_plotly(feat_names[0], plot_style, idx=0)
        _, yticklabels = _get_ticks_plotly(feat_names[1], plot_style, idx=1)
        feat_type_x, feat_type_y = pdp_inter_obj.feature_types
        xlabel = f"<b>target_{t}</b><br>{feat_names[0]}"
        if plot_style.plot_pdp:
            iso_x_trace = _pdp_iso_xy_plot_plotly(
                pdp_x,
                cmap,
                plot_style,
                (pdp_min, pdp_max),
                yaxes=False,
            )
            fig.add_trace(iso_x_trace, **iso_x_grids)

            iso_y_trace = _pdp_iso_xy_plot_plotly(
                pdp_y,
                cmap,
                plot_style,
                (pdp_min, pdp_max),
                yaxes=True,
            )
            fig.add_trace(iso_y_trace, **iso_y_grids)

            _set_xy_ticks_plotly(
                xlabel, xticklabels, iso_x_grids, fig, xy=False, y=False
            )
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
                xlabel, xticklabels, inter_grids, fig, xy=True, y=False
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
        colorbar=dict(len=cb_z, yanchor="bottom", y=cb_y, x=cb_x, xanchor="left"),
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
    if y:
        axes.set_ylabel(feat_name, fontdict=plot_style.label["fontdict"])
    else:
        axes.set_xlabel(feat_name, fontdict=plot_style.label["fontdict"])
        if not plot_style.show_percentile:
            axes.xaxis.set_label_position("top")
    _axes_modify(axes, plot_style, top=True)
    axes.tick_params(axis="both", which="minor", length=0)
    axes.grid(False)

    if xy and plot_style.interact["type"] == "contour" and not plot_style.x_quantile:
        return

    if y:
        axes.set_yticks(range(len(ticklabels)), ticklabels)
        if not xy:
            axes.get_xaxis().set_visible(False)
    else:
        axes.set_xticks(range(len(ticklabels)), ticklabels)
        if not xy:
            axes.get_yaxis().set_visible(False)


def _set_xy_ticks_plotly(feat_name, ticklabels, grids, fig, xy=False, y=False):
    tick_params = dict(
        title_text=feat_name,
        title_standoff=0,
    )
    if ticklabels is not None:
        tick_params.update(
            dict(
                ticktext=ticklabels,
                tickvals=np.arange(len(ticklabels)),
            )
        )
    tick_params.update(grids)

    if y:
        if not xy:
            fig.update_xaxes(showgrid=False, showticklabels=False, **grids)
            # tick_params.update({"tickangle": -90})
        fig.update_yaxes(**tick_params)
    else:
        tick_params.update({"side": "top"})
        fig.update_xaxes(**tick_params)
        if not xy:
            fig.update_yaxes(showgrid=False, showticklabels=False, **grids)
