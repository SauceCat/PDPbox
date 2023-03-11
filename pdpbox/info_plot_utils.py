from .utils import (
    _axes_modify,
    _modify_legend_axes,
    _find_bucket,
    _make_bucket_column_names,
    _find_onehot_actual,
    _make_bucket_column_names_percentile,
    _check_dataset,
    _check_percentile_range,
    _check_feature,
    _check_grid_type,
    _expand_default,
    _plot_title,
    _get_grids,
    _check_model,
    _check_classes,
    _q2,
)
from .styles import infoPlotStyle, infoPlotInterStyle

import numpy as np
import pandas as pd
import copy
from itertools import product
import plotly.express as px

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patheffects as PathEffects

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _make_subplots(plot_style):
    fig = plt.figure(figsize=plot_style.figsize, dpi=plot_style.dpi)
    title_ratio = 2
    outer_grid = GridSpec(
        nrows=2,
        ncols=1,
        wspace=0.0,
        hspace=0.0,
        height_ratios=[title_ratio, plot_style.figsize[1] - title_ratio],
    )
    title_axes = plt.subplot(outer_grid[0])
    fig.add_subplot(title_axes)
    _plot_title(title_axes, plot_style)

    inner_grid = GridSpecFromSubplotSpec(
        plot_style.nrows,
        plot_style.ncols,
        subplot_spec=outer_grid[1],
        wspace=0.3,
        hspace=0.2,
    )

    return fig, inner_grid, title_axes


def _target_plot(feat_name, target, bar_data, target_lines, plot_params):
    """Internal call for target_plot"""
    plot_style = _prepare_plot_style(feat_name, target, plot_params, "target")
    fig, inner_grid, title_axes = _make_subplots(plot_style)

    bar_axes, line_axes = [], []
    line_colors = plot_style.line["colors"]
    for i, t in enumerate(target):
        inner_bar_axes = plt.subplot(inner_grid[i])
        inner_line_axes = inner_bar_axes.twinx()
        fig.add_subplot(inner_bar_axes)

        line_color = line_colors[i % len(line_colors)]
        line_data = (
            target_lines[i].rename(columns={t: "y"}).sort_values("x", ascending=True)
        )
        _draw_barplot(feat_name, [bar_data, inner_bar_axes], plot_style)
        _draw_lineplot(
            "Average " + t,
            [line_data, inner_line_axes],
            plot_style,
            line_color,
        )
        bar_axes.append(inner_bar_axes)
        line_axes.append(inner_line_axes)

    axes = {"title_axes": title_axes, "bar_axes": bar_axes, "line_axes": line_axes}
    return fig, axes


def _actual_plot(
    feat_name,
    pred_cols,
    plot_data,
    bar_data,
    box_lines,
    plot_params,
):
    """Internal call for actual_plot"""
    plot_style = _prepare_plot_style(feat_name, pred_cols, plot_params, "actual")
    fig, inner_grid, title_axes = _make_subplots(plot_style)

    box_axes, bar_axes = [], []
    box_colors = plot_style.box["colors"]
    for i, p in enumerate(pred_cols):
        box_color = box_colors[i % len(box_colors)]
        inner = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=inner_grid[i], wspace=0, hspace=0.2
        )
        inner_box_axes = plt.subplot(inner[0])
        inner_bar_axes = plt.subplot(inner[1], sharex=inner_box_axes)
        fig.add_subplot(inner_box_axes)
        fig.add_subplot(inner_bar_axes)

        inner_box_data = plot_data[["x", p]].rename(columns={p: "y"})
        inner_box_line_data = box_lines[i].rename(columns={p + "_q2": "y"})

        _draw_boxplot(
            inner_box_data,
            inner_box_line_data,
            inner_box_axes,
            plot_style,
            box_color,
            show_percentile=True,
        )
        inner_box_axes.set_ylabel("%s dist" % p)
        _draw_barplot(
            feat_name, (bar_data, inner_bar_axes), plot_style, show_percentile=False
        )

        if i % plot_style.ncols != 0:
            inner_bar_axes.set_yticklabels([])
            inner_box_axes.set_yticklabels([])

        box_axes.append(inner_box_axes)
        bar_axes.append(inner_bar_axes)

    axes = {"title_axes": title_axes, "box_axes": box_axes, "bar_axes": bar_axes}
    return fig, axes


def _info_plot_interact(
    feat_names,
    target,
    plot_data,
    plot_params,
    plot_type="target_interact",
):
    """Internal call for _info_plot_interact"""

    plot_style, (count_min, count_max), marker_sizes = _prepare_info_interact_plot(
        feat_names, target, plot_data, plot_params, plot_type
    )
    fig, value_grid, title_axes = _make_subplots(plot_style)

    value_axes, legend_axes = [], []
    cmaps = plot_style.marker["cmaps"]
    for i, t in enumerate(target):
        inner_grid = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=value_grid[i], height_ratios=[7, 1], hspace=0.3
        )
        inner_value_axes = plt.subplot(inner_grid[0])
        fig.add_subplot(inner_value_axes)
        value_axes.append(inner_value_axes)
        cmap = cmaps[i % len(cmaps)]

        per_xaxes, per_yaxes, (value_min, value_max) = _plot_interact(
            feat_names,
            t,
            plot_data,
            inner_value_axes,
            plot_style,
            marker_sizes,
            cmap,
        )

        # draw legend
        inner_legend_grid = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=inner_grid[1], wspace=0, width_ratios=[1, 1]
        )
        inner_legend_axes = []
        for j in range(2):
            subplot = plt.subplot(inner_legend_grid[j])
            inner_legend_axes.append(subplot)
            fig.add_subplot(subplot)

        _plot_legend_colorbar(
            value_min,
            value_max,
            inner_legend_axes[0],
            cmap,
            plot_style.legend["colorbar"],
        )
        _plot_legend_circles(
            count_min,
            count_max,
            inner_legend_axes[1],
            cmap,
            plot_style.legend["circles"],
        )
        legend_axes.append(inner_legend_axes)

        if plot_type == "target_interact":
            subplot_title = f"Average {t}"
        else:
            subplot_title = f"{t}: median prediction"

        if len(plot_style.percentile_columns[0]) > 0:
            subplot_title += "\n\n\n"

        inner_value_axes.set_title(
            subplot_title,
            fontdict={
                "fontsize": plot_style.title["subplot_title"]["font_size"],
                "fontname": plot_style.font_family,
            },
        )

    axes = {
        "title_axes": title_axes,
        "value_axes": value_axes,
        "legend_axes": legend_axes,
    }
    return fig, axes


def _make_subplots_plotly(plot_args, plot_style):
    fig = make_subplots(**plot_args)
    fig.update_layout(
        width=plot_style.figsize[0],
        height=plot_style.figsize[1],
        template=plot_style.template,
        showlegend=False,
        title=go.layout.Title(
            text=f"{plot_style.title['title']['text']} <br><sup>{plot_style.title['subtitle']['text']}</sup>",
            xref="paper",
            x=0,
        ),
    )

    return fig


def _draw_barplot_plotly(fig, bar_data, plot_style, row, col):
    bx = bar_data["x"].values
    by = bar_data["fake_count"].values

    fig.add_trace(
        go.Bar(
            x=bx,
            y=by,
            text=by,
            textposition="outside",
            width=plot_style.bar["width"],
            name="count",
            marker=dict(
                color=plot_style.bar["color"],
                opacity=0.5,
            ),
        ),
        row=row,
        col=col,
        secondary_y=False,
    )


def _get_ticks_plotly(feat_name, plot_style):
    ticktext = plot_style.display_columns.copy()
    if len(plot_style.percentile_columns) > 0:
        for j, p in enumerate(plot_style.percentile_columns):
            ticktext[j] += f"<br><sup><b>{p}</b></sup>"
        title_text = f"<b>{feat_name}</b> (value+percentile)"
    else:
        title_text = f"<b>{feat_name}</b> (value)"

    return title_text, ticktext


def _target_plot_plotly(feat_name, target, bar_data, target_lines, plot_params):
    """Internal call for target_plot, drawing with plotly"""
    plot_style = _prepare_plot_style(feat_name, target, plot_params, "target")
    nrows, ncols = plot_style.nrows, plot_style.ncols
    plot_args = {
        "rows": nrows,
        "cols": ncols,
        "specs": [[{"secondary_y": True} for _ in range(ncols)] for _ in range(nrows)],
        "horizontal_spacing": plot_style.horizontal_spacing,
        "vertical_spacing": plot_style.vertical_spacing,
    }
    fig = _make_subplots_plotly(plot_args, plot_style)

    line_colors = plot_style.line["colors"]
    for i, t in enumerate(target):
        line_color = line_colors[i % len(line_colors)]
        grids = {"col": i % ncols + 1, "row": i // ncols + 1}

        _draw_barplot_plotly(fig, bar_data, plot_style, **grids)

        line_data = (
            target_lines[i].rename(columns={t: "y"}).sort_values("x", ascending=True)
        )
        lx = line_data["x"].values
        ly = line_data["y"].values

        fig.add_trace(
            go.Scatter(
                x=lx,
                y=ly,
                text=[round(v, 3) for v in ly],
                textposition="top center",
                mode="lines+markers+text",
                yaxis="y2",
                name="Average " + t,
                line=dict(color=line_color),
                marker=dict(color=line_color),
            ),
            secondary_y=True,
            **grids,
        )

        title_text, ticktext = _get_ticks_plotly(feat_name, plot_style)
        fig.update_xaxes(
            title_text=title_text,
            ticktext=ticktext,
            tickvals=bar_data["x"].values,
            **grids,
        )

        if grids["col"] == 1:
            fig.update_yaxes(title_text="count", secondary_y=False, **grids)

        fig.update_yaxes(
            title_text="Average " + t,
            secondary_y=True,
            **grids,
        )

    return fig


def _actual_plot_plotly(
    feat_name,
    pred_cols,
    plot_data,
    bar_data,
    box_lines,
    plot_params,
):
    """Internal call for actual_plot"""
    plot_style = _prepare_plot_style(feat_name, pred_cols, plot_params, "actual")
    nrows, ncols = plot_style.nrows, plot_style.ncols
    plot_args = {
        "rows": nrows * 2,
        "cols": ncols,
        "shared_xaxes": True,
        "shared_yaxes": True,
        "horizontal_spacing": plot_style.horizontal_spacing,
        "vertical_spacing": plot_style.vertical_spacing,
    }
    fig = _make_subplots_plotly(plot_args, plot_style)

    box_colors = plot_style.box["colors"]
    for i, p in enumerate(pred_cols):
        box_color = box_colors[i % len(box_colors)]
        grids = {"col": i % ncols + 1, "row": i // ncols + 1}

        box_data = plot_data[["x", p]].rename(columns={p: "y"})
        xs, ys = _prepare_box_data(box_data)
        box_grids = {"col": grids["col"], "row": grids["row"] * 2 - 1}
        for iy, y in enumerate(ys):
            fig.add_trace(
                go.Box(
                    x0=iy,
                    y=y,
                    width=plot_style.box["width"],
                    marker=dict(
                        color=box_color,
                        opacity=0.5,
                    ),
                    name=f"{p} dist",
                    boxpoints=False,
                ),
                **box_grids,
            )

        box_line_data = box_lines[i].rename(columns={p + "_q2": "y"})
        lx = box_line_data["x"].values
        ly = box_line_data["y"].values
        fig.add_trace(
            go.Scatter(
                x=lx,
                y=ly,
                text=[round(v, 3) for v in ly],
                textposition="top center",
                mode="lines+markers+text",
                line=dict(color=box_color),
                marker=dict(color=box_color),
            ),
            **box_grids,
        )
        fig.update_yaxes(title_text=f"{p} dist", **box_grids)

        bar_grids = {"row": grids["row"] * 2, "col": grids["col"]}
        _draw_barplot_plotly(fig, bar_data, plot_style, **bar_grids)

        title_text, ticktext = _get_ticks_plotly(feat_name, plot_style)
        fig.update_xaxes(
            title_text=title_text,
            ticktext=ticktext,
            tickvals=bar_data["x"].values,
            **bar_grids,
        )

        by = bar_data["fake_count"].values
        fig.update_yaxes(title_text="count", range=[0, np.max(by) * 1.15], **bar_grids)

    return fig


def _info_plot_interact_plotly(
    feat_names,
    target,
    plot_data,
    plot_params,
    plot_type="target_interact",
):
    """Internal call for _info_plot_interact"""

    plot_style, (count_min, count_max), marker_sizes = _prepare_info_interact_plot(
        feat_names, target, plot_data, plot_params, plot_type
    )

    nrows, ncols = plot_style.nrows, plot_style.ncols
    row_heights = []
    specs = []
    for ir in range(nrows * 2):
        if ir % 2 == 0:
            row_heights.append(10 if len(target) > 1 else 12)
            specs.append([{"colspan": 2}, None] * ncols)
        else:
            row_heights.append(1)
            specs.append([{"colspan": 1}, {"colspan": 1}] * ncols)

    plot_args = {
        "rows": nrows * 2,
        "cols": ncols * 2,
        "horizontal_spacing": plot_style.horizontal_spacing,
        "vertical_spacing": plot_style.vertical_spacing,
        "specs": specs,
        "row_heights": row_heights,
    }
    fig = _make_subplots_plotly(plot_args, plot_style)

    cmaps = plot_style.marker["cmaps"]
    for i, t in enumerate(target):
        grids = {"col": i % ncols + 1, "row": i // ncols + 1}
        cmap = cmaps[i % len(cmaps)]
        sx = plot_data["x1"].values
        sy = plot_data["x2"].values
        sv = plot_data[t].values
        texts = [
            f"{item['fake_count']}<br>{item[t]:.3f}"
            for item in plot_data.to_dict("records")
        ]
        colors = getattr(px.colors.sequential, cmap)
        line_color = colors[len(colors) // 2]
        edge_color = colors[-1]

        scatter_grids = {"row": grids["row"] * 2 - 1, "col": grids["col"] * 2 - 1}
        fig.add_trace(
            go.Scatter(
                x=sx,
                y=sy,
                text=texts,
                textposition="middle center",
                mode="markers+text" if plot_style.annotate else "markers",
                marker=dict(
                    size=marker_sizes,
                    color=sv,
                    colorscale=cmap,
                    line=dict(width=plot_style.marker["line_width"], color=edge_color),
                ),
            ),
            **scatter_grids,
        )

        # display percentile
        for k in range(2):
            ticktext = plot_style.display_columns[k].copy()
            if (
                len(plot_style.percentile_columns) > 0
                and len(plot_style.percentile_columns[k]) > 0
            ):
                for j, p in enumerate(plot_style.percentile_columns[k]):
                    ticktext[j] += f"<br><sup><b>{p}</b></sup>"
                title_text = f"<b>{feat_names[k]}</b> (value+percentile)"
            else:
                title_text = f"<b>{feat_names[k]}</b> (value)"

            kwargs = dict(
                title_text=title_text,
                ticktext=ticktext,
                tickvals=np.arange(len(ticktext)),
                **scatter_grids,
            )
            if k == 0:
                fig.update_xaxes(**kwargs)
            else:
                fig.update_yaxes(**kwargs)

        cb_grids = {"row": grids["row"] * 2, "col": grids["col"] * 2 - 1}
        colorbar = go.Figure(
            data=go.Heatmap(z=[np.arange(1000)], showscale=False, colorscale=cmap),
        )
        fig.add_trace(colorbar.data[0], **cb_grids)
        fig.update_yaxes(showticklabels=False, **cb_grids)
        fig.update_xaxes(showticklabels=False, **cb_grids)

        fig.add_annotation(
            x=0,
            y=0,
            text=str(round(np.min(sv), 3)),
            showarrow=False,
            xanchor="left",
            font=dict(color=edge_color),
            **cb_grids,
        )
        fig.add_annotation(
            x=999,
            y=0,
            text=str(round(np.max(sv), 3)),
            showarrow=False,
            xanchor="right",
            font=dict(color="#ffffff"),
            **cb_grids,
        )

        size_grids = {"row": grids["row"] * 2, "col": grids["col"] * 2}
        fig.add_trace(
            go.Scatter(
                x=[0.25, 0.75],
                y=[0.5, 0.5],
                text=[count_min, count_max],
                mode="markers+lines+text",
                marker=dict(
                    size=[15, 25],
                    color=line_color,
                ),
            ),
            **size_grids,
        )
        fig.update_yaxes(visible=False, **size_grids)
        fig.update_xaxes(visible=False, range=[0, 1], **size_grids)

    return fig


def _prepare_data_x(
    feature,
    feature_type,
    data,
    num_grid_points,
    grid_type,
    percentile_range,
    grid_range,
    cust_grid_points,
    show_percentile,
    show_outliers,
    endpoint,
):
    """Map value to bucket based on feature grids"""
    cols, uppers, lowers = [], [], []
    per_cols, per_uppers, per_lowers = [], [], []
    data_x = data.copy()

    if feature_type == "binary":
        grids = np.array([0, 1])
        cols = [f"{feature}_{g}" for g in grids]
        data_x["x"] = data_x[feature]

    if feature_type == "numeric":
        if cust_grid_points is None:
            grids, percentiles = _get_grids(
                data_x[feature].values,
                num_grid_points,
                grid_type,
                percentile_range,
                grid_range,
            )
        else:
            grids = np.array(sorted(cust_grid_points))
            percentiles = None

        if endpoint:
            grids[-1] *= 1.01

        if not show_outliers:
            data_x = data_x[
                (data_x[feature] >= grids[0]) & (data_x[feature] <= grids[-1])
            ].reset_index(drop=True)
        else:
            grids = np.array(
                [data_x[feature].min() * 0.99]
                + list(grids)
                + [data_x[feature].max() * 1.01]
            )

        # map feature value into value buckets
        # data_x["x"] = data_x[feature].apply(lambda x: _find_bucket(x, grids_raw, endpoint))
        data_x["x"] = pd.cut(data_x[feature].values, bins=grids, right=False).codes
        if not show_outliers:
            data_x["x"] += 1
        uni_xs = sorted(data_x["x"].unique())

        # create bucket names
        ranges = np.arange(uni_xs[0], uni_xs[-1] + 1)
        if endpoint:
            grids[-1] /= 1.01
        cols, lowers, uppers = _make_bucket_column_names(grids, endpoint, ranges)

        # create percentile bucket names
        if show_percentile and grid_type == "percentile":
            (
                per_cols,
                per_lowers,
                per_uppers,
            ) = _make_bucket_column_names_percentile(percentiles, endpoint, ranges)

        # offset results
        data_x["x"] = data_x["x"] - data_x["x"].min()

    if feature_type == "onehot":
        grids = cols = np.array(feature)
        data_x["x"] = np.argmax(data_x[feature].values, axis=1)
        data_x = data_x[~data_x["x"].isnull()].reset_index(drop=True)

    data_x["x"] = data_x["x"].map(int)
    results = {
        "data": data_x,
        "value_display": [list(lst) for lst in [cols, lowers, uppers]],
        "percentile_display": [list(lst) for lst in [per_cols, per_lowers, per_uppers]],
    }

    return results


def _prepare_plot_params(
    plot_params,
    ncols,
    display_columns,
    percentile_columns,
    figsize,
    dpi,
    template,
    engine,
):
    if plot_params is None:
        plot_params = {}
    if figsize is not None:
        plot_params["figsize"] = figsize
    if template is not None:
        plot_params["template"] = template

    plot_params.update(
        {
            "ncols": ncols,
            "display_columns": display_columns,
            "percentile_columns": percentile_columns,
            "dpi": dpi,
            "engine": engine,
        }
    )
    return plot_params


def _autolabel(rects, axes, color, fontdict):
    """Create label for bar plot"""
    for rect in rects:
        height = rect.get_height()
        bbox_props = {
            "facecolor": "white",
            "edgecolor": color,
            "boxstyle": "square,pad=0.5",
        }
        axes.text(
            rect.get_x() + rect.get_width() / 2.0,
            height,
            str(int(height)),
            ha="center",
            va="center",
            bbox=bbox_props,
            color=color,
            fontdict=fontdict,
        )


def _display_percentile(axes, show_percentile, plot_style):
    percentile_columns = plot_style.percentile_columns
    if len(percentile_columns) > 0 and show_percentile:
        per_axes = axes.twiny()
        per_axes.set_xticks(axes.get_xticks())
        per_axes.set_xbound(axes.get_xbound())
        per_axes.set_xticklabels(
            percentile_columns, rotation=plot_style.tick["xticks_rotation"]
        )
        per_axes.set_xlabel("percentile buckets", fontdict=plot_style.label["fontdict"])
        _axes_modify(per_axes, plot_style, top=True)


def _draw_barplot(feat_name, data_axes, plot_style, show_percentile=True):
    """Draw bar plot"""
    data, axes = data_axes

    # add value label for bar plot
    rects = axes.bar(
        x=data["x"],
        height=data["fake_count"],
        width=plot_style.bar["width"],
        color=plot_style.bar["color"],
        alpha=0.5,
    )
    _autolabel(rects, axes, plot_style.bar["color"], plot_style.bar["fontdict"])
    _axes_modify(axes, plot_style)
    num_xticks = len(plot_style.display_columns)

    # bar plot
    axes.set_xlabel(feat_name, fontdict=plot_style.label["fontdict"])
    axes.set_ylabel("count", fontdict=plot_style.label["fontdict"])
    axes.set_xticks(range(num_xticks))
    axes.set_xticklabels(
        plot_style.display_columns, rotation=plot_style.tick["xticks_rotation"]
    )
    axes.set_xlim(-0.5, num_xticks - 0.5)

    _display_percentile(axes, show_percentile, plot_style)


def _draw_lineplot(target_ylabel, data_axes, plot_style, line_color=None):
    """Draw line plot"""
    data, axes = data_axes
    if line_color is None:
        line_color = plot_style.line["color"]

    axes.plot(
        data["x"],
        data["y"],
        linewidth=plot_style.line["width"],
        c=line_color,
        marker="o",
    )
    for idx in data.index.values:
        bbox_props = {
            "facecolor": line_color,
            "edgecolor": "none",
            "boxstyle": "square,pad=0.5",
        }
        y = data.loc[idx, "y"]
        axes.text(
            data.loc[idx, "x"],
            y,
            f"{y:.3f}",
            ha="center",
            va="top",
            bbox=bbox_props,
            color="#ffffff",
            fontdict=plot_style.line["fontdict"],
        )

    _axes_modify(axes, plot_style)
    axes.get_yaxis().tick_right()
    axes.grid(False)
    axes.set_ylabel(target_ylabel, fontdict=plot_style.label["fontdict"])


def _prepare_plot_style(feat_name, target, plot_params, plot_type):
    if plot_type in ["target", "actual"]:
        plot_style = infoPlotStyle(feat_name, target, plot_params, plot_type)
    elif plot_type in ["target_interact", "actual_interact"]:
        plot_style = infoPlotInterStyle(feat_name, target, plot_params, plot_type)
    return plot_style


def _prepare_info_interact_plot(
    feat_names, target, plot_data, plot_params, plot_type="target_interact"
):
    plot_style = _prepare_plot_style(feat_names, target, plot_params, plot_type)
    count_min, count_max = plot_data["fake_count"].min(), plot_data["fake_count"].max()

    marker_sizes = []
    for count in plot_data["fake_count"].values:
        size = (
            float(count - count_min)
            / (count_max - count_min)
            * (plot_style.marker["max_size"] - plot_style.marker["min_size"])
            + plot_style.marker["min_size"]
        )
        marker_sizes.append(size)

    return plot_style, (count_min, count_max), marker_sizes


def _prepare_box_data(box_data):
    xs = sorted(box_data["x"].unique())
    ys = []
    for x in xs:
        ys.append(box_data[box_data["x"] == x]["y"].values)
    return xs, ys


def _draw_boxplot(
    box_data, box_line_data, box_axes, plot_style, box_color=None, show_percentile=True
):
    """Draw box plot"""
    xs, ys = _prepare_box_data(box_data)
    if box_color is None:
        box_color = plot_style.box["color"]

    boxprops = dict(linewidth=plot_style.box["line_width"], color=box_color)
    medianprops = dict(linewidth=0)
    whiskerprops = copy.deepcopy(boxprops)
    capprops = copy.deepcopy(boxprops)

    box_axes.boxplot(
        ys,
        positions=xs,
        showfliers=False,
        widths=plot_style.box["width"],
        whiskerprops=whiskerprops,
        capprops=capprops,
        boxprops=boxprops,
        medianprops=medianprops,
    )
    _axes_modify(box_axes, plot_style)

    box_axes.plot(
        box_line_data["x"], box_line_data["y"], linewidth=1, c=box_color, linestyle="--"
    )
    for idx in box_line_data.index.values:
        bbox_props = {
            "facecolor": "white",
            "edgecolor": box_color,
            "boxstyle": "square,pad=0.5",
            "lw": 1,
        }
        box_axes.text(
            box_line_data.loc[idx, "x"],
            box_line_data.loc[idx, "y"],
            "%.3f" % box_line_data.loc[idx, "y"],
            ha="center",
            va="top",
            size=10,
            bbox=bbox_props,
            color=box_color,
        )
    box_axes.set_xticklabels([])

    _display_percentile(box_axes, show_percentile, plot_style)


def _plot_interact(
    feat_names,
    target,
    plot_data,
    plot_axes,
    plot_style,
    marker_sizes,
    cmap=None,
):
    """Interact scatter plot"""
    if cmap is None:
        cmap = plot_style.marker["cmap"]

    xs, ys = plot_style.display_columns
    plot_axes.set_xticks(range(len(xs)))
    plot_axes.set_xticklabels(xs, rotation=plot_style.tick["xticks_rotation"])
    plot_axes.set_xlim(-0.5, len(xs) - 0.5)
    plot_axes.set_yticks(range(len(ys)))
    plot_axes.set_yticklabels(ys)
    plot_axes.set_ylim(-0.5, len(ys) - 0.5)

    per_xaxes, per_yaxes = None, None
    xs, ys = plot_style.percentile_columns
    if len(xs) > 0:
        per_xaxes = plot_axes.twiny()
        per_xaxes.set_xticks(plot_axes.get_xticks())
        per_xaxes.set_xbound(plot_axes.get_xbound())
        per_xaxes.set_xticklabels(xs, rotation=plot_style.tick["xticks_rotation"])
        per_xaxes.set_xlabel("percentile buckets")
        _axes_modify(per_xaxes, plot_style, top=True)
        per_xaxes.grid(False)

    if len(ys) > 0:
        per_yaxes = plot_axes.twinx()
        per_yaxes.set_yticks(plot_axes.get_yticks())
        per_yaxes.set_ybound(plot_axes.get_ybound())
        per_yaxes.set_yticklabels(ys)
        per_yaxes.set_ylabel("percentile buckets")
        _axes_modify(per_yaxes, plot_style, right=True)
        per_yaxes.grid(False)

    value_min, value_max = plot_data[target].min(), plot_data[target].max()
    colors = [
        plt.get_cmap(cmap)(float(v - value_min) / (value_max - value_min))
        for v in plot_data[target].values
    ]

    plot_axes.scatter(
        plot_data["x1"].values,
        plot_data["x2"].values,
        s=marker_sizes,
        c=colors,
        linewidth=plot_style.marker["line_width"],
        edgecolors=plt.get_cmap(cmap)(1.0),
    )

    if plot_style.annotate:
        for plot_item in plot_data.to_dict("records"):
            text = plot_axes.text(
                x=plot_item["x1"],
                y=plot_item["x2"],
                s=f"{plot_item['fake_count']}\n{plot_item[target]:.3f}",
                fontdict={
                    "family": plot_style.font_family,
                    "color": plt.get_cmap(cmap)(1.0),
                    "size": plot_style.marker["fontsize"],
                },
                va="center",
                ha="left",
            )
            text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="w")])

    plot_axes.set_xlabel(feat_names[0])
    plot_axes.set_ylabel(feat_names[1])
    _axes_modify(plot_axes, plot_style)
    return per_xaxes, per_yaxes, (value_min, value_max)


def _plot_legend_colorbar(value_min, value_max, axes, cmap, style):
    """Plot colorbar legend"""

    norm = mpl.colors.Normalize(vmin=float(value_min), vmax=float(value_max))

    # color bar
    cax = inset_axes(axes, height=style["height"], width=style["width"], loc=10)
    cb = mpl.colorbar.ColorbarBase(
        cax, cmap=plt.get_cmap(cmap), norm=norm, orientation="horizontal"
    )
    cb.outline.set_linewidth(0)
    cb.set_ticks([])

    width_float = float(style["width"].replace("%", "")) / 100
    text_params = {
        "fontdict": {
            "color": plt.get_cmap(cmap)(1.0),
            "fontsize": style["fontsize"],
            "fontfamily": style["fontfamily"],
        },
        "transform": axes.transAxes,
        "va": "center",
    }
    tmin = cb.ax.text(
        (1.0 - width_float) / 2, 0.5, f"{value_min:.3f}", ha="left", **text_params
    )
    tmin.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="w")])

    tmax = cb.ax.text(
        1.0 - (1.0 - width_float) / 2,
        0.5,
        f"{value_max:.3f}",
        ha="right",
        **text_params,
    )
    tmax.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="w")])
    _modify_legend_axes(axes, style["fontfamily"])


def _plot_legend_circles(count_min, count_max, axes, cmap, style):
    """Plot circle legend"""

    xs = [0.75, 2]
    ys = [1] * 2
    axes.plot(xs, ys, color=plt.get_cmap(cmap)(1.0), zorder=1, ls="--", lw=1)
    axes.scatter(
        xs,
        ys,
        s=[200, 500],
        edgecolors=plt.get_cmap(cmap)(1.0),
        color="white",
        zorder=2,
    )
    axes.set_xlim(0.0, 3)

    text_params = {
        "fontdict": {
            "color": plt.get_cmap(cmap)(1.0),
            "fontsize": style["fontsize"],
            "fontfamily": style["fontfamily"],
        },
        "ha": "center",
        "va": "center",
    }
    for i, count in enumerate([count_min, count_max]):
        count_text = axes.text(xs[i], ys[i], count, **text_params)
        count_text.set_path_effects(
            [PathEffects.withStroke(linewidth=3, foreground="w")]
        )

    _modify_legend_axes(axes, style["fontfamily"])


def _prepare_info_plot_data(
    feature,
    feature_type,
    data,
    num_grid_points,
    grid_type,
    percentile_range,
    grid_range,
    cust_grid_points,
    show_percentile,
    show_outliers,
    endpoint,
):
    """Prepare data for information plots"""
    prepared_results = _prepare_data_x(
        feature,
        feature_type,
        data,
        num_grid_points,
        grid_type,
        percentile_range,
        grid_range,
        cust_grid_points,
        show_percentile,
        show_outliers,
        endpoint,
    )
    data_x = prepared_results["data"]
    cols, lowers, uppers = prepared_results["value_display"]
    per_cols, per_lowers, per_uppers = prepared_results["percentile_display"]

    data_x["fake_count"] = 1
    bar_data = (
        data_x.groupby("x", as_index=False)
        .agg({"fake_count": "count"})
        .sort_values("x", ascending=True)
    )

    summary_df = pd.DataFrame(
        np.arange(data_x["x"].min(), data_x["x"].max() + 1), columns=["x"]
    )
    summary_df = summary_df.merge(
        bar_data.rename(columns={"fake_count": "count"}), on="x", how="left"
    ).fillna(0)
    summary_df["x"] = summary_df["x"].map(int)
    summary_df["value"] = summary_df["x"].apply(lambda x: cols[x])
    info_cols = ["x", "value"]

    if len(per_cols) != 0:
        summary_df["percentile"] = summary_df["x"].apply(lambda x: per_cols[x])
        info_cols.append("percentile")
    summary_df = summary_df[info_cols + ["count"]]

    return data_x, bar_data, summary_df, cols, per_cols


def _prepare_info_plot_interact_data(
    features,
    feature_types,
    target,
    data,
    num_grid_points,
    grid_types,
    percentile_ranges,
    grid_ranges,
    cust_grid_points,
    show_percentile,
    show_outliers,
    endpoint,
    plot_type="target_interact",
):
    """Prepare data for information interact plots"""
    prepared_results = []
    for i in range(2):
        res = _prepare_data_x(
            features[i],
            feature_types[i],
            data,
            num_grid_points[i],
            grid_types[i],
            percentile_ranges[i],
            grid_ranges[i],
            cust_grid_points[i],
            show_percentile,
            show_outliers[i],
            endpoint,
        )
        prepared_results.append(res)
        if i == 0:
            data = res["data"].rename(columns={"x": "x1"})

    data_x = prepared_results[1]["data"].rename(columns={"x": "x2"})
    data_x["fake_count"] = 1

    agg_dict = {}
    for t in target:
        agg_dict[t] = "mean" if plot_type == "target_interact" else _q2
    agg_dict["fake_count"] = "count"
    plot_data = data_x.groupby(["x1", "x2"], as_index=False).agg(agg_dict)

    x1_values = list(np.arange(data_x["x1"].min(), data_x["x1"].max() + 1))
    x2_values = list(np.arange(data_x["x2"].min(), data_x["x2"].max() + 1))
    summary_df = pd.DataFrame(
        [{"x1": x1, "x2": x2} for x1, x2 in list(product(x1_values, x2_values))]
    )
    for x_col in ["x1", "x2"]:
        summary_df[x_col] = summary_df[x_col].map(int)
    summary_df = summary_df.merge(
        plot_data.rename(columns={"fake_count": "count"}), on=["x1", "x2"], how="left"
    ).fillna(0)

    info_cols = ["x1", "x2", "value_1", "value_2"]
    cols, per_cols = [], []
    for i in range(1, 3):
        cols_, lowers_, uppers_ = prepared_results[i - 1]["value_display"]
        per_cols_, per_lowers_, per_uppers_ = prepared_results[i - 1][
            "percentile_display"
        ]
        cols.append(cols_)
        per_cols.append(per_cols_)

        summary_df[f"value_{i}"] = summary_df[f"x{i}"].apply(lambda x: cols_[x])
        if len(per_cols_) != 0:
            summary_df[f"percentile_{i}"] = summary_df[f"x{i}"].apply(
                lambda x: per_cols_[x]
            )
            info_cols.append(f"percentile_{i}")

    summary_df = summary_df[info_cols + ["count"] + target]

    return plot_data, plot_data, summary_df, cols, per_cols


def _check_info_plot_params(
    df,
    feature,
    grid_type,
    percentile_range,
    grid_range,
    cust_grid_points,
    show_outliers,
):
    """Check information plot parameters"""
    _check_dataset(df)
    feature_type = _check_feature(feature, df)
    _check_grid_type(grid_type)
    _check_percentile_range(percentile_range)

    # show_outliers should be only turned on when necessary
    if all(v is None for v in [percentile_range, grid_range, cust_grid_points]):
        show_outliers = False

    return feature_type, show_outliers


def _check_info_plot_interact_params(
    df,
    features,
    grid_types,
    percentile_ranges,
    grid_ranges,
    num_grid_points,
    cust_grid_points,
    show_outliers,
):
    """Check interact information plot parameters"""
    _check_dataset(df)
    num_grid_points = _expand_default(num_grid_points, 10)

    grid_types = _expand_default(grid_types, "percentile")
    [_check_grid_type(v) for v in grid_types]

    percentile_ranges = _expand_default(percentile_ranges, None)
    [_check_percentile_range(v) for v in percentile_ranges]

    grid_ranges = _expand_default(grid_ranges, None)
    cust_grid_points = _expand_default(cust_grid_points, None)

    if not show_outliers:
        show_outliers = [False, False]
    else:
        show_outliers = [True, True]
        for i, vs in enumerate(zip(percentile_ranges, grid_ranges, cust_grid_points)):
            if all(v is None for v in vs):
                show_outliers[i] = False

    feature_types = [_check_feature(f, df) for f in features]

    return {
        "num_grid_points": num_grid_points,
        "grid_types": grid_types,
        "percentile_ranges": percentile_ranges,
        "grid_ranges": grid_ranges,
        "cust_grid_points": cust_grid_points,
        "show_outliers": show_outliers,
        "feature_types": feature_types,
    }


def _prepare_actual_plot_data(
    model, X, n_classes, pred_func, predict_kwds, features, which_classes
):
    model_n_classes, model_pred_func = _check_model(model)
    if model_n_classes is None:
        assert (
            n_classes is not None
        ), "n_classes is required when it can't be accessed through model.n_classes_."
    else:
        n_classes = model_n_classes

    if pred_func is None:
        print("predict using model predict func...")
        assert (
            model_pred_func is not None
        ), "pred_func is required when model.predict_proba or model.predict doesn't exist."
        prediction = model_pred_func(X, **predict_kwds)
    else:
        print("predict using customized func...")
        prediction = pred_func(model, X, **predict_kwds)

    info_df = X[features]
    pred_cols = ["pred"]
    if n_classes == 0:
        info_df["pred"] = prediction
    elif n_classes == 2:
        info_df["pred"] = prediction[:, 1]
    else:
        plot_classes = range(n_classes)
        if which_classes is not None:
            _check_classes(which_classes, n_classes)
            plot_classes = sorted(which_classes)

        pred_cols = []
        for class_idx in plot_classes:
            info_df[f"pred_{class_idx}"] = prediction[:, class_idx]
            pred_cols.append(f"pred_{class_idx}")

    return info_df, pred_cols
