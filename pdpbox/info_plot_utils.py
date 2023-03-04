from .utils import (
    _axes_modify,
    _modify_legend_ax,
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
)
from .styles import infoPlotStyle

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patheffects as PathEffects

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


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

        grids_raw = grids.copy()

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


def _draw_barplot(feat_name, data_axes, plot_style):
    """Draw bar plot"""
    data, axes = data_axes
    display_columns = plot_style.display_columns
    bar_width = plot_style.bar["width"]
    if bar_width is None:
        bar_width = np.min([0.4, 0.4 / (10.0 / len(display_columns))])
    bar_color = plot_style.bar["color"]

    # add value label for bar plot
    rects = axes.bar(
        x=data["x"],
        height=data["fake_count"],
        width=bar_width,
        color=bar_color,
        alpha=0.5,
    )
    _autolabel(rects, axes, bar_color, plot_style.bar["fontdict"])
    _axes_modify(axes, plot_style)

    # bar plot
    axes.set_xlabel(feat_name, fontdict=plot_style.label["fontdict"])
    axes.set_ylabel("count", fontdict=plot_style.label["fontdict"])
    axes.set_xticks(range(len(display_columns)))
    axes.set_xticklabels(display_columns, rotation=plot_style.tick["xticks_rotation"])
    axes.set_xlim(-0.5, len(display_columns) - 0.5)

    # display percentile
    percentile_columns = plot_style.percentile_columns
    if len(percentile_columns) > 0:
        axes_per = axes.twiny()
        axes_per.set_xticks(axes.get_xticks())
        axes_per.set_xbound(axes.get_xbound())
        axes_per.set_xticklabels(
            percentile_columns, rotation=plot_style.tick["xticks_rotation"]
        )
        axes_per.set_xlabel("percentile buckets", fontdict=plot_style.label["fontdict"])
        _axes_modify(axes_per, plot_style, top=True)


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


def _prepare_for_target_plot(feat_name, target, target_lines, plot_params):
    plot_style = infoPlotStyle(feat_name, plot_params)

    # set up graph parameters
    width, height = plot_style.figsize
    nrows, ncols = plot_style.nrows, plot_style.ncols
    if len(target) > 1:
        nrows = int(np.ceil(len(target) * 1.0 / ncols))
        ncols = np.min([len(target), ncols])
        height = width * 1.0 / ncols * nrows * 0.8
    else:
        ncols = 1

    plot_style.figsize = (width, height)
    plot_style.nrows, plot_style.ncols = int(nrows), int(ncols)

    # get max average target value
    ys = []
    for i, t in enumerate(target):
        ys += list(target_lines[i][t].values)
    y_max = np.max(ys)

    return plot_style, y_max


def _target_plot_plotly(feat_name, target, bar_data, target_lines, plot_params):
    """Internal call for target_plot, drawing with plotly"""
    plot_style, y_max = _prepare_for_target_plot(
        feat_name, target, target_lines, plot_params
    )

    fig = make_subplots(
        rows=plot_style.nrows,
        cols=plot_style.ncols,
        specs=[
            [{"secondary_y": True} for _ in range(plot_style.ncols)]
            for _ in range(plot_style.nrows)
        ],
        shared_yaxes=True,
        horizontal_spacing=0.1,
        vertical_spacing=0.05,
    )

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

    line_colors = plot_style.line["colors"]
    if line_colors is None:
        line_colors = px.colors.qualitative.Plotly

    for i, t in enumerate(target):
        line_color = line_colors[i % len(line_colors)]
        ic, ir = i % plot_style.ncols + 1, i // plot_style.ncols + 1

        line_data = (
            target_lines[i].rename(columns={t: "y"}).sort_values("x", ascending=True)
        )

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
            row=ir,
            col=ic,
            secondary_y=False,
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
            row=ir,
            col=ic,
            secondary_y=True,
        )

        # display percentile
        ticktext = plot_style.display_columns.copy()
        if len(plot_style.percentile_columns) > 0:
            for j, p in enumerate(plot_style.percentile_columns):
                ticktext[j] += f"<br><sup><b>{p}</b></sup>"
            title_text = f"<b>{feat_name}</b> (value+percentile)"
        else:
            title_text = f"<b>{feat_name}</b> (value)"

        fig.update_xaxes(
            title_text=title_text, ticktext=ticktext, tickvals=bx, row=ir, col=ic
        )

        if ic == 1:
            fig.update_yaxes(title_text="count", secondary_y=False, row=ir, col=ic)
        fig.update_yaxes(
            title_text="Average " + t,
            secondary_y=True,
            row=ir,
            col=ic,
            range=[0, y_max * 1.1],
        )

    return fig


def _target_plot(feat_name, target, bar_data, target_lines, plot_params):
    """Internal call for target_plot"""
    plot_style, y_max = _prepare_for_target_plot(
        feat_name, target, target_lines, plot_params
    )

    fig = plt.figure(figsize=plot_style.figsize, dpi=300)
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
        wspace=0.2,
        hspace=0.35,
    )
    plot_axes, bar_axes, line_axes = [], [], []
    for inner_idx in range(len(target)):
        axes = plt.subplot(inner_grid[inner_idx])
        plot_axes.append(axes)
        fig.add_subplot(axes)

    line_colors = plot_style.line["colors"]
    if line_colors is None:
        line_colors = plt.get_cmap(plot_style.line["cmap"])(
            np.arange(np.min([20, len(target)]))
        )

    for i, t in enumerate(target):
        inner_line_color = line_colors[i % len(line_colors)]
        inner_bar_axes = plot_axes[i]
        inner_line_axes = inner_bar_axes.twinx()

        line_data = (
            target_lines[i].rename(columns={t: "y"}).sort_values("x", ascending=True)
        )
        _draw_barplot(feat_name, [bar_data, inner_bar_axes], plot_style)
        _draw_lineplot(
            "Average " + t,
            [line_data, inner_line_axes],
            plot_style,
            inner_line_color,
        )
        inner_line_axes.set_ylim(0.0, y_max)
        bar_axes.append(inner_bar_axes)
        line_axes.append(inner_line_axes)

        if i % plot_style.ncols != 0:
            inner_bar_axes.set_yticklabels([])
            inner_bar_axes.tick_params(which="major", left=False)

        if (i % plot_style.ncols + 1 != plot_style.ncols) and i != len(plot_axes) - 1:
            inner_line_axes.set_yticklabels([])
            inner_line_axes.tick_params(which="major", right=False)

    if len(plot_axes) > len(target):
        for i in range(len(target), len(plot_axes)):
            plot_axes[i].axis("off")

    axes = {"title_axes": title_axes, "bar_axes": bar_axes, "line_axes": line_axes}
    return fig, axes


def _draw_boxplot(
    box_data, box_line_data, box_ax, display_columns, box_color, plot_params
):
    """Draw box plot"""
    font_family = plot_params.get("font_family", "Arial")
    box_line_width = plot_params.get("box_line_width", 1.5)
    box_width = plot_params.get(
        "box_width", np.min([0.4, 0.4 / (10.0 / len(display_columns))])
    )

    xs = sorted(box_data["x"].unique())
    ys = []
    for x in xs:
        ys.append(box_data[box_data["x"] == x]["y"].values)

    boxprops = dict(linewidth=box_line_width, color=box_color)
    medianprops = dict(linewidth=0)
    whiskerprops = dict(linewidth=box_line_width, color=box_color)
    capprops = dict(linewidth=box_line_width, color=box_color)

    box_ax.boxplot(
        ys,
        positions=xs,
        showfliers=False,
        widths=box_width,
        whiskerprops=whiskerprops,
        capprops=capprops,
        boxprops=boxprops,
        medianprops=medianprops,
    )

    _axes_modify(font_family=font_family, ax=box_ax)

    box_ax.plot(
        box_line_data["x"], box_line_data["y"], linewidth=1, c=box_color, linestyle="--"
    )
    for idx in box_line_data.index.values:
        bbox_props = {
            "facecolor": "white",
            "edgecolor": box_color,
            "boxstyle": "square,pad=0.5",
            "lw": 1,
        }
        box_ax.text(
            box_line_data.loc[idx, "x"],
            box_line_data.loc[idx, "y"],
            "%.3f" % box_line_data.loc[idx, "y"],
            ha="center",
            va="top",
            size=10,
            bbox=bbox_props,
            color=box_color,
        )


def _draw_box_bar(
    bar_data,
    bar_ax,
    box_data,
    box_line_data,
    box_color,
    box_ax,
    feature_name,
    display_columns,
    percentile_columns,
    plot_params,
    target_ylabel,
):
    """Draw box plot and bar plot"""

    font_family = plot_params.get("font_family", "Arial")
    xticks_rotation = plot_params.get("xticks_rotation", 0)

    _draw_boxplot(
        box_data=box_data,
        box_line_data=box_line_data,
        box_ax=box_ax,
        display_columns=display_columns,
        box_color=box_color,
        plot_params=plot_params,
    )
    box_ax.set_ylabel("%sprediction dist" % target_ylabel)
    box_ax.set_xticklabels([])

    _draw_barplot(
        bar_data=bar_data,
        bar_ax=bar_ax,
        display_columns=display_columns,
        plot_params=plot_params,
    )

    # bar plot
    bar_ax.set_xlabel(feature_name)
    bar_ax.set_ylabel("count")

    bar_ax.set_xticks(range(len(display_columns)))
    bar_ax.set_xticklabels(display_columns, rotation=xticks_rotation)
    bar_ax.set_xlim(-0.5, len(display_columns) - 0.5)

    plt.setp(box_ax.get_xticklabels(), visible=False)

    # display percentile
    if len(percentile_columns) > 0:
        percentile_ax = box_ax.twiny()
        percentile_ax.set_xticks(box_ax.get_xticks())
        percentile_ax.set_xbound(box_ax.get_xbound())
        percentile_ax.set_xticklabels(percentile_columns, rotation=xticks_rotation)
        percentile_ax.set_xlabel("percentile buckets")
        _axes_modify(font_family=font_family, ax=percentile_ax, top=True)


def _actual_plot(
    plot_data,
    bar_data,
    box_lines,
    actual_prediction_columns,
    feature_name,
    display_columns,
    percentile_columns,
    figsize,
    ncols,
    plot_params,
):
    """Internal call for actual_plot"""

    # set up graph parameters
    width, height = 15, 9
    nrows = 1

    if plot_params is None:
        plot_params = dict()

    # font_family = plot_params.get('font_family', 'Arial')
    box_color = plot_params.get("box_color", "#3288bd")
    box_colors_cmap = plot_params.get("box_colors_cmap", "tab20")
    box_colors = plot_params.get(
        "box_colors",
        plt.get_cmap(box_colors_cmap)(
            range(np.min([20, len(actual_prediction_columns)]))
        ),
    )
    title = plot_params.get("title", "Actual predictions plot for %s" % feature_name)
    subtitle = plot_params.get(
        "subtitle",
        "Distribution of actual prediction through different feature values.",
    )

    if len(actual_prediction_columns) > 1:
        nrows = int(np.ceil(len(actual_prediction_columns) * 1.0 / ncols))
        ncols = np.min([len(actual_prediction_columns), ncols])
        width = np.min([7.5 * len(actual_prediction_columns), 15])
        height = width * 1.0 / ncols * nrows

    if figsize is not None:
        width, height = figsize

    if plot_params is None:
        plot_params = dict()

    fig = plt.figure(figsize=(width, height))
    outer_grid = GridSpec(2, 1, wspace=0.0, hspace=0.1, height_ratios=[2, height - 2])
    title_ax = plt.subplot(outer_grid[0])
    fig.add_subplot(title_ax)
    _plot_title(
        title=title, subtitle=subtitle, title_ax=title_ax, plot_params=plot_params
    )

    box_bar_params = {
        "bar_data": bar_data,
        "feature_name": feature_name,
        "display_columns": display_columns,
        "percentile_columns": percentile_columns,
        "plot_params": plot_params,
    }

    if len(actual_prediction_columns) == 1:
        inner_grid = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[1])

        box_ax = plt.subplot(inner_grid[0])
        bar_ax = plt.subplot(inner_grid[1], sharex=box_ax)
        fig.add_subplot(box_ax)
        fig.add_subplot(bar_ax)

        if actual_prediction_columns[0] == "actual_prediction":
            target_ylabel = ""
        else:
            target_ylabel = "target_%s: " % actual_prediction_columns[0].split("_")[-1]

        box_data = plot_data[["x", actual_prediction_columns[0]]].rename(
            columns={actual_prediction_columns[0]: "y"}
        )
        box_line_data = box_lines[0].rename(
            columns={actual_prediction_columns[0] + "_q2": "y"}
        )
        _draw_box_bar(
            bar_ax=bar_ax,
            box_data=box_data,
            box_line_data=box_line_data,
            box_color=box_color,
            box_ax=box_ax,
            target_ylabel=target_ylabel,
            **box_bar_params,
        )
    else:
        inner_grid = GridSpecFromSubplotSpec(
            nrows, ncols, subplot_spec=outer_grid[1], wspace=0.2, hspace=0.35
        )

        box_ax = []
        bar_ax = []

        # get max average target value
        ys = []
        for idx in range(len(box_lines)):
            ys += list(box_lines[idx][actual_prediction_columns[idx] + "_q2"].values)
        y_max = np.max(ys)

        for idx in range(len(actual_prediction_columns)):
            box_color = box_colors[idx % len(box_colors)]

            inner = GridSpecFromSubplotSpec(
                2, 1, subplot_spec=inner_grid[idx], wspace=0, hspace=0.2
            )
            inner_box_ax = plt.subplot(inner[0])
            inner_bar_ax = plt.subplot(inner[1], sharex=inner_box_ax)
            fig.add_subplot(inner_box_ax)
            fig.add_subplot(inner_bar_ax)

            inner_box_data = plot_data[["x", actual_prediction_columns[idx]]].rename(
                columns={actual_prediction_columns[idx]: "y"}
            )
            inner_box_line_data = box_lines[idx].rename(
                columns={actual_prediction_columns[idx] + "_q2": "y"}
            )
            _draw_box_bar(
                bar_ax=inner_bar_ax,
                box_data=inner_box_data,
                box_line_data=inner_box_line_data,
                box_color=box_color,
                box_ax=inner_box_ax,
                target_ylabel="target_%s: "
                % actual_prediction_columns[idx].split("_")[-1],
                **box_bar_params,
            )

            inner_box_ax.set_ylim(0.0, y_max)

            if idx % ncols != 0:
                inner_bar_ax.set_yticklabels([])
                inner_box_ax.set_yticklabels([])

            box_ax.append(inner_box_ax)
            bar_ax.append(inner_bar_ax)

    axes = {"title_ax": title_ax, "box_ax": box_ax, "bar_ax": bar_ax}
    return fig, axes


def _plot_interact(
    plot_data,
    y,
    plot_ax,
    feature_names,
    display_columns,
    percentile_columns,
    marker_sizes,
    cmap,
    line_width,
    xticks_rotation,
    font_family,
    annotate,
):
    """Interact scatter plot"""

    plot_ax.set_xticks(range(len(display_columns[0])))
    plot_ax.set_xticklabels(display_columns[0], rotation=xticks_rotation)
    plot_ax.set_yticks(range(len(display_columns[1])))
    plot_ax.set_yticklabels(display_columns[1])
    plot_ax.set_xlim(-0.5, len(display_columns[0]) - 0.5)
    plot_ax.set_ylim(-0.5, len(display_columns[1]) - 0.5)

    percentile_ax = None
    percentile_ay = None
    # display percentile
    if len(percentile_columns[0]) > 0:
        percentile_ax = plot_ax.twiny()
        percentile_ax.set_xticks(plot_ax.get_xticks())
        percentile_ax.set_xbound(plot_ax.get_xbound())
        percentile_ax.set_xticklabels(percentile_columns[0], rotation=xticks_rotation)
        percentile_ax.set_xlabel("percentile buckets")
        _axes_modify(font_family=font_family, ax=percentile_ax, top=True)
        percentile_ax.grid(False)

    # display percentile
    if len(percentile_columns[1]) > 0:
        percentile_ay = plot_ax.twinx()
        percentile_ay.set_yticks(plot_ax.get_yticks())
        percentile_ay.set_ybound(plot_ax.get_ybound())
        percentile_ay.set_yticklabels(percentile_columns[1])
        percentile_ay.set_ylabel("percentile buckets")
        _axes_modify(font_family=font_family, ax=percentile_ay, right=True)
        percentile_ay.grid(False)

    value_min, value_max = plot_data[y].min(), plot_data[y].max()
    colors = [
        plt.get_cmap(cmap)(float(v - value_min) / (value_max - value_min))
        for v in plot_data[y].values
    ]

    plot_ax.scatter(
        plot_data["x1"].values,
        plot_data["x2"].values,
        s=marker_sizes,
        c=colors,
        linewidth=line_width,
        edgecolors=plt.get_cmap(cmap)(1.0),
    )

    if annotate:
        for text_idx in range(plot_data.shape[0]):
            plot_data_idx = plot_data.iloc[text_idx]
            text_s = "%d\n%.3f" % (plot_data_idx["fake_count"], plot_data_idx[y])
            txt = plot_ax.text(
                x=plot_data_idx["x1"],
                y=plot_data_idx["x2"],
                s=text_s,
                fontdict={
                    "family": font_family,
                    "color": plt.get_cmap(cmap)(1.0),
                    "size": 11,
                },
                va="center",
                ha="left",
            )
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="w")])

    plot_ax.set_xlabel(feature_names[0])
    plot_ax.set_ylabel(feature_names[1])

    _axes_modify(font_family=font_family, ax=plot_ax)
    return value_min, value_max, percentile_ax, percentile_ay


def _plot_legend_colorbar(
    value_min, value_max, colorbar_ax, cmap, font_family, height="50%", width="100%"
):
    """Plot colorbar legend"""

    norm = mpl.colors.Normalize(vmin=float(value_min), vmax=float(value_max))

    # color bar
    cax = inset_axes(colorbar_ax, height=height, width=width, loc=10)
    cb = mpl.colorbar.ColorbarBase(
        cax, cmap=plt.get_cmap(cmap), norm=norm, orientation="horizontal"
    )
    cb.outline.set_linewidth(0)
    cb.set_ticks([])

    width_float = float(width.replace("%", "")) / 100
    text_params = {
        "fontdict": {"color": plt.get_cmap(cmap)(1.0), "fontsize": 10},
        "transform": colorbar_ax.transAxes,
        "va": "center",
    }
    tmin = cb.ax.text(
        (1.0 - width_float) / 2, 0.5, "%.3f " % value_min, ha="left", **text_params
    )
    tmin.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="w")])

    tmax = cb.ax.text(
        1.0 - (1.0 - width_float) / 2,
        0.5,
        "%.3f " % value_max,
        ha="right",
        **text_params,
    )
    tmax.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="w")])
    _modify_legend_ax(colorbar_ax, font_family)


def _plot_legend_circles(count_min, count_max, circle_ax, cmap, font_family):
    """Plot circle legend"""
    # circle size
    circle_ax.plot([0.75, 2], [1] * 2, color=plt.get_cmap(cmap)(1.0), zorder=1, ls="--")
    circle_ax.scatter(
        [0.75, 2],
        [1] * 2,
        s=[500, 1500],
        edgecolors=plt.get_cmap(cmap)(1.0),
        color="white",
        zorder=2,
    )
    circle_ax.set_xlim(0.0, 3)

    text_params = {
        "fontdict": {"color": "#424242", "fontsize": 10},
        "ha": "center",
        "va": "center",
    }
    circle_ax.text(0.75, 1.0, count_min, **text_params)
    circle_ax.text(2, 1.0, count_max, **text_params)
    _modify_legend_ax(circle_ax, font_family)


def _info_plot_interact(
    feature_names,
    display_columns,
    percentile_columns,
    ys,
    plot_data,
    title,
    subtitle,
    figsize,
    ncols,
    annotate,
    plot_params,
    is_target_plot=True,
):
    """Internal call for _info_plot_interact"""

    width, height = 15, 10
    nrows = 1

    if len(ys) > 1:
        nrows = int(np.ceil(len(ys) * 1.0 / ncols))
        ncols = np.min([len(ys), ncols])
        width = np.min([7.5 * len(ys), 15])
        height = width * 1.2 / ncols * nrows

    if figsize is not None:
        width, height = figsize

    font_family = plot_params.get("font_family", "Arial")
    cmap = plot_params.get("cmap", "Blues")
    cmaps = plot_params.get(
        "cmaps", ["Blues", "Greens", "Oranges", "Reds", "Purples", "Greys"]
    )
    line_width = plot_params.get("line_width", 1)
    xticks_rotation = plot_params.get("xticks_rotation", 0)
    marker_size_min = plot_params.get("marker_size_min", 50)
    marker_size_max = plot_params.get("marker_size_max", 1500)

    # plot title
    fig = plt.figure(figsize=(width, height))
    outer_grid = GridSpec(2, 1, wspace=0.0, hspace=0.1, height_ratios=[2, height - 2])
    title_ax = plt.subplot(outer_grid[0])
    fig.add_subplot(title_ax)
    _plot_title(
        title=title, subtitle=subtitle, title_ax=title_ax, plot_params=plot_params
    )

    # draw value plots and legend
    count_min, count_max = plot_data["fake_count"].min(), plot_data["fake_count"].max()
    marker_sizes = []
    for count in plot_data["fake_count"].values:
        size = (
            float(count - count_min)
            / (count_max - count_min)
            * (marker_size_max - marker_size_min)
            + marker_size_min
        )
        marker_sizes.append(size)

    interact_params = {
        "plot_data": plot_data,
        "feature_names": feature_names,
        "display_columns": display_columns,
        "percentile_columns": percentile_columns,
        "marker_sizes": marker_sizes,
        "line_width": line_width,
        "xticks_rotation": xticks_rotation,
        "font_family": font_family,
        "annotate": annotate,
    }
    if len(ys) == 1:
        inner_grid = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_grid[1], height_ratios=[height - 3, 1], hspace=0.25
        )
        value_ax = plt.subplot(inner_grid[0])
        fig.add_subplot(value_ax)
        value_min, value_max, percentile_ax, percentile_ay = _plot_interact(
            y=ys[0], plot_ax=value_ax, cmap=cmap, **interact_params
        )

        # draw legend
        legend_grid = GridSpecFromSubplotSpec(
            1, 4, subplot_spec=inner_grid[1], wspace=0
        )
        legend_ax = [plt.subplot(legend_grid[0]), plt.subplot(legend_grid[1])]
        fig.add_subplot(legend_ax[0])
        fig.add_subplot(legend_ax[1])
        _plot_legend_colorbar(
            value_min=value_min,
            value_max=value_max,
            colorbar_ax=legend_ax[0],
            cmap=cmap,
            font_family=font_family,
        )
        _plot_legend_circles(
            count_min=count_min,
            count_max=count_max,
            circle_ax=legend_ax[1],
            cmap=cmap,
            font_family=font_family,
        )

        if is_target_plot:
            subplot_title = "Average %s" % ys[0]
        elif ys[0] == "actual_prediction_q2":
            subplot_title = "Median Prediction"
        else:
            subplot_title = "target_%s: median prediction" % ys[0].split("_")[-2]
        if len(percentile_columns[0]) > 0:
            subplot_title += "\n\n\n"
        value_ax.set_title(
            subplot_title, fontdict={"fontsize": 11, "fontname": font_family}
        )
    else:
        value_grid = GridSpecFromSubplotSpec(
            nrows, ncols, subplot_spec=outer_grid[1], wspace=0.2, hspace=0.2
        )
        value_ax = []
        legend_ax = []

        for idx in range(len(ys)):
            inner_grid = GridSpecFromSubplotSpec(
                2, 1, subplot_spec=value_grid[idx], height_ratios=[7, 1], hspace=0.3
            )
            inner_value_ax = plt.subplot(inner_grid[0])
            fig.add_subplot(inner_value_ax)
            value_ax.append(inner_value_ax)

            cmap_idx = cmaps[idx % len(cmaps)]
            value_min, value_max, percentile_ax, percentile_ay = _plot_interact(
                y=ys[idx], plot_ax=inner_value_ax, cmap=cmap_idx, **interact_params
            )

            # draw legend
            inner_legend_grid = GridSpecFromSubplotSpec(
                1, 4, subplot_spec=inner_grid[1], wspace=0
            )
            inner_legend_ax = [
                plt.subplot(inner_legend_grid[0]),
                plt.subplot(inner_legend_grid[1]),
            ]
            fig.add_subplot(inner_legend_ax[0])
            fig.add_subplot(inner_legend_ax[1])
            _plot_legend_colorbar(
                value_min=value_min,
                value_max=value_max,
                colorbar_ax=inner_legend_ax[0],
                cmap=cmap_idx,
                font_family=font_family,
            )
            _plot_legend_circles(
                count_min=count_min,
                count_max=count_max,
                circle_ax=inner_legend_ax[1],
                cmap=cmap_idx,
                font_family=font_family,
            )
            legend_ax.append(inner_legend_ax)

            if is_target_plot:
                subplot_title = "Average %s" % ys[idx]
            else:
                subplot_title = "target_%s: median prediction" % ys[idx].split("_")[-2]
            if len(percentile_columns[0]) > 0:
                subplot_title += "\n\n\n"
            inner_value_ax.set_title(
                subplot_title, fontdict={"fontsize": 11, "fontname": font_family}
            )

            if idx % ncols != 0:
                inner_value_ax.set_yticklabels([])

            if (idx % ncols + 1 != ncols) and idx != len(value_ax) - 1:
                if percentile_ay is not None:
                    percentile_ay.set_yticklabels([])

        if len(value_ax) > len(ys):
            for idx in range(len(ys), len(value_ax)):
                value_ax[idx].axis("off")

    axes = {"title_ax": title_ax, "value_ax": value_ax, "legend_ax": legend_ax}
    return fig, axes


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
    data_input,
    features,
    feature_types,
    num_grid_points,
    grid_types,
    percentile_ranges,
    grid_ranges,
    cust_grid_points,
    show_percentile,
    show_outliers,
    endpoint,
    agg_dict,
):
    """Prepare data for information interact plots"""
    prepared_results = []
    for i in range(2):
        prepared_result = _prepare_data_x(
            feature=features[i],
            feature_type=feature_types[i],
            data=data_input,
            num_grid_points=num_grid_points[i],
            grid_type=grid_types[i],
            percentile_range=percentile_ranges[i],
            grid_range=grid_ranges[i],
            cust_grid_points=cust_grid_points[i],
            show_percentile=show_percentile,
            show_outliers=show_outliers[i],
            endpoint=endpoint,
        )
        prepared_results.append(prepared_result)
        if i == 0:
            data_input = prepared_result["data"].rename(columns={"x": "x1"})

    data_x = prepared_results[1]["data"].rename(columns={"x": "x2"})
    data_x["fake_count"] = 1
    plot_data = data_x.groupby(["x1", "x2"], as_index=False).agg(agg_dict)

    return data_x, plot_data, prepared_results


def _prepare_info_plot_interact_summary(
    data_x, plot_data, prepared_results, feature_types
):
    """Prepare summary data frame for interact plots"""

    x1_values = []
    x2_values = []
    for x1_value in range(data_x["x1"].min(), data_x["x1"].max() + 1):
        for x2_value in range(data_x["x2"].min(), data_x["x2"].max() + 1):
            x1_values.append(x1_value)
            x2_values.append(x2_value)
    summary_df = pd.DataFrame()
    summary_df["x1"] = x1_values
    summary_df["x2"] = x2_values
    summary_df = summary_df.merge(
        plot_data.rename(columns={"fake_count": "count"}), on=["x1", "x2"], how="left"
    ).fillna(0)

    info_cols = ["x1", "x2", "display_column_1", "display_column_2"]
    display_columns = []
    percentile_columns = []
    for i in range(2):
        display_columns_i, bound_lows_i, bound_ups_i = prepared_results[i][
            "value_display"
        ]
        (
            percentile_columns_i,
            percentile_bound_lows_i,
            percentile_bound_ups_i,
        ) = prepared_results[i]["percentile_display"]
        display_columns.append(display_columns_i)
        percentile_columns.append(percentile_columns_i)

        summary_df["display_column_%d" % (i + 1)] = summary_df["x%d" % (i + 1)].apply(
            lambda x: display_columns_i[int(x)]
        )
        if feature_types[i] == "numeric":
            summary_df["value_lower_%d" % (i + 1)] = summary_df["x%d" % (i + 1)].apply(
                lambda x: bound_lows_i[int(x)]
            )
            summary_df["value_upper_%d" % (i + 1)] = summary_df["x%d" % (i + 1)].apply(
                lambda x: bound_ups_i[int(x)]
            )
            info_cols += ["value_lower_%d" % (i + 1), "value_upper_%d" % (i + 1)]

        if len(percentile_columns_i) != 0:
            summary_df["percentile_column_%d" % (i + 1)] = summary_df[
                "x%d" % (i + 1)
            ].apply(lambda x: percentile_columns_i[int(x)])
            summary_df["percentile_lower_%d" % (i + 1)] = summary_df[
                "x%d" % (i + 1)
            ].apply(lambda x: percentile_bound_lows_i[int(x)])
            summary_df["percentile_upper_%d" % (i + 1)] = summary_df[
                "x%d" % (i + 1)
            ].apply(lambda x: percentile_bound_ups_i[int(x)])
            info_cols += [
                "percentile_column_%d" % (i + 1),
                "percentile_lower_%d" % (i + 1),
                "percentile_upper_%d" % (i + 1),
            ]

    return summary_df, info_cols, display_columns, percentile_columns


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
    if (
        (percentile_range is None)
        and (grid_range is None)
        and (cust_grid_points is None)
    ):
        show_outliers = False

    return feature_type, show_outliers


def _check_info_plot_interact_params(
    num_grid_points,
    grid_types,
    percentile_ranges,
    grid_ranges,
    cust_grid_points,
    show_outliers,
    plot_params,
    features,
    df,
):
    """Check interact information plot parameters"""

    _check_dataset(df=df)
    num_grid_points = _expand_default(num_grid_points, 10)
    grid_types = _expand_default(grid_types, "percentile")
    _check_grid_type(grid_type=grid_types[0])
    _check_grid_type(grid_type=grid_types[1])

    percentile_ranges = _expand_default(percentile_ranges, None)
    _check_percentile_range(percentile_range=percentile_ranges[0])
    _check_percentile_range(percentile_range=percentile_ranges[1])

    grid_ranges = _expand_default(grid_ranges, None)
    cust_grid_points = _expand_default(cust_grid_points, None)

    if not show_outliers:
        show_outliers = [False, False]
    else:
        show_outliers = [True, True]
        for i in range(2):
            if (
                (percentile_ranges[i] is None)
                and (grid_ranges[i] is None)
                and (cust_grid_points[i] is None)
            ):
                show_outliers[i] = False

    # set up graph parameters
    if plot_params is None:
        plot_params = dict()

    # check features
    feature_types = [
        _check_feature(feature=features[0], df=df),
        _check_feature(feature=features[1], df=df),
    ]

    return {
        "num_grid_points": num_grid_points,
        "grid_types": grid_types,
        "percentile_ranges": percentile_ranges,
        "grid_ranges": grid_ranges,
        "cust_grid_points": cust_grid_points,
        "show_outliers": show_outliers,
        "plot_params": plot_params,
        "feature_types": feature_types,
    }
