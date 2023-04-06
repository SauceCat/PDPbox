from .utils import (
    _q2,
    _check_interact_plot_params,
    _display_percentile_inter,
)
from .styles import (
    _prepare_plot_style,
    _axes_modify,
    _display_percentile,
    _display_ticks_plotly,
)

import numpy as np
import pandas as pd
import copy
from itertools import product, cycle
import plotly.express as px

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patheffects as PathEffects

import plotly.graph_objects as go


def _target_plot(plot_obj, plot_params):
    """Internal call for target_plot"""
    feature_info = plot_obj.feature_info
    plot_style = _prepare_plot_style(
        feature_info.name, plot_obj.target, plot_params, plot_obj.plot_type
    )
    fig, inner_grid, title_axes = plot_style.make_subplots()

    axes = {"title_axes": title_axes, "bar_axes": [], "line_axes": []}
    line_colors = cycle(plot_style.line["colors"])

    for i, target in enumerate(plot_obj.target):
        line_color = next(line_colors)
        bar_axes = plt.subplot(inner_grid[i])
        line_axes = bar_axes.twinx()
        line_data = (
            plot_obj.target_lines[i]
            .rename(columns={target: "y"})
            .sort_values("x", ascending=True)
        )
        _draw_barplot(
            plot_obj.count_df, bar_axes, feature_info.display_columns, plot_style
        )
        _draw_lineplot(
            f"Average {target}",
            line_data,
            line_axes,
            line_color,
            plot_style,
        )
        _axes_modify(line_axes, plot_style, right=True, grid=False)

        if len(plot_obj.target) > 1:
            bar_axes.set_title(
                target,
                **plot_style.title["subplot_title"],
            )

        if plot_style.show_percentile:
            _display_percentile(bar_axes, feature_info.percentile_columns, plot_style)

        axes["bar_axes"].append(bar_axes)
        axes["line_axes"].append(line_axes)

    return fig, axes


def _target_plot_plotly(plot_obj, plot_params):
    """Internal call for target_plot, drawing with plotly"""
    feature_info = plot_obj.feature_info
    plot_style = _prepare_plot_style(
        feature_info.name, plot_obj.target, plot_params, plot_obj.plot_type
    )
    nrows, ncols = plot_style.nrows, plot_style.ncols
    plot_args = {
        "rows": nrows,
        "cols": ncols,
        "specs": [[{"secondary_y": True} for _ in range(ncols)] for _ in range(nrows)],
        "horizontal_spacing": 0,
        "vertical_spacing": 0,
    }
    fig = plot_style.make_subplots_plotly(plot_args)
    subplot_titles = []
    line_colors = cycle(plot_style.line["colors"])
    for i, target in enumerate(plot_obj.target):
        grids = {"col": i % ncols + 1, "row": i // ncols + 1}
        nr, nc = grids["row"], grids["col"]
        title = plot_style.update_plot_domains(fig, nr, nc, grids, target)
        subplot_titles.append(title)
        line_color = next(line_colors)
        line_data = (
            plot_obj.target_lines[i]
            .rename(columns={target: "y"})
            .sort_values("x", ascending=True)
        )
        _draw_barplot_plotly(fig, plot_obj.count_df, plot_style, grids)
        _draw_lineplot_plotly(target, fig, line_data, line_color, grids)
        _display_ticks_plotly(
            feature_info.display_columns,
            feature_info.percentile_columns if plot_style.show_percentile else [],
            fig,
            grids,
            is_y=False,
        )
        fig.update_yaxes(title_text="count", secondary_y=False, **grids)
        fig.update_yaxes(
            title_text=f"Average {target}", secondary_y=True, showgrid=False, **grids
        )

    if len(plot_obj.target) > 1:
        fig.update_layout(annotations=subplot_titles)

    return fig, None


def _predict_plot(plot_obj, plot_params):
    """Internal call for actual_plot"""
    feature_info = plot_obj.feature_info
    plot_style = _prepare_plot_style(
        feature_info.name, plot_obj.target, plot_params, plot_obj.plot_type
    )
    fig, inner_grid, title_axes = plot_style.make_subplots()
    axes = {"title_axes": title_axes, "bar_axes": [], "box_axes": []}

    box_colors = cycle(plot_style.box["colors"])
    for i, target in enumerate(plot_obj.target):
        box_color = next(box_colors)
        inner = GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=inner_grid[i],
            wspace=plot_style.gaps["inner_x"],
            hspace=plot_style.gaps["inner_y"],
        )
        box_axes = plt.subplot(inner[0])
        bar_axes = plt.subplot(inner[1])
        box_df = plot_obj.df[["x", target]].rename(columns={target: "y"})
        line_df = plot_obj.target_lines[i].rename(columns={target + "_q2": "y"})
        _draw_boxplot(box_df, line_df, box_axes, box_color, plot_style)
        box_axes.set(ylabel=f"{target} dist", xticklabels=[])
        _draw_barplot(
            plot_obj.count_df, bar_axes, feature_info.display_columns, plot_style
        )
        if len(plot_obj.target) > 1:
            box_axes.set_title(
                target,
                **plot_style.title["subplot_title"],
            )

        if plot_style.show_percentile:
            _display_percentile(box_axes, feature_info.percentile_columns, plot_style)

        axes["bar_axes"].append(bar_axes)
        axes["box_axes"].append(box_axes)

    return fig, axes


def _predict_plot_plotly(plot_obj, plot_params):
    """Internal call for actual_plot"""
    feature_info = plot_obj.feature_info
    plot_style = _prepare_plot_style(
        feature_info.name, plot_obj.target, plot_params, plot_obj.plot_type
    )
    nrows, ncols = plot_style.nrows, plot_style.ncols
    plot_args = {
        "rows": nrows * 2,
        "cols": ncols,
        "shared_xaxes": True,
        "horizontal_spacing": 0,
        "vertical_spacing": 0,
    }
    fig = plot_style.make_subplots_plotly(plot_args)
    subplot_titles = []
    box_colors = cycle(plot_style.box["colors"])
    for i, target in enumerate(plot_obj.target):
        box_color = next(box_colors)
        grids = {"col": i % ncols + 1, "row": i // ncols + 1}
        box_grids = {"col": grids["col"], "row": grids["row"] * 2 - 1}
        bar_grids = {"col": grids["col"], "row": grids["row"] * 2}
        nr, nc = grids["row"], grids["col"]
        title = plot_style.update_plot_domains(
            fig, nr, nc, (box_grids, bar_grids), target
        )
        subplot_titles.append(title)

        box_df = plot_obj.df[["x", target]].rename(columns={target: "y"})
        line_df = plot_obj.target_lines[i].rename(columns={target + "_q2": "y"})

        _draw_boxplot_plotly(
            target, box_df, line_df, fig, box_color, plot_style, box_grids
        )
        _draw_barplot_plotly(fig, plot_obj.count_df, plot_style, bar_grids)
        _display_ticks_plotly(
            feature_info.display_columns,
            feature_info.percentile_columns if plot_style.show_percentile else [],
            fig,
            bar_grids,
            is_y=False,
        )
        fig.update_yaxes(title_text="count", **bar_grids)
        fig.update_yaxes(title_text=f"{target} dist", **box_grids)

    if len(plot_obj.target) > 1:
        fig.update_layout(annotations=subplot_titles)

    return fig, None


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
            2,
            1,
            subplot_spec=value_grid[i],
            wspace=plot_style.gaps["inner_x"],
            hspace=plot_style.gaps["inner_y"],
            height_ratios=plot_style.subplot_ratio["y"],
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
        if plot_style.ncols > 1:
            inner_legend_grid = GridSpecFromSubplotSpec(
                1, 2, subplot_spec=inner_grid[1], wspace=0, width_ratios=[1, 1]
            )
        else:
            inner_legend_grid = GridSpecFromSubplotSpec(
                1, 4, subplot_spec=inner_grid[1], wspace=0, width_ratios=[1, 1, 1, 1]
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

    axes = {
        "title_axes": title_axes,
        "value_axes": value_axes,
        "legend_axes": legend_axes,
    }
    return fig, axes


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
        "horizontal_spacing": 0,
        "vertical_spacing": 0,
        # "specs": specs,
        # "row_heights": row_heights,
    }
    fig = _make_subplots_plotly(plot_args, plot_style)
    cb_texts = []

    cmaps = plot_style.marker["cmaps"]
    for i, t in enumerate(target):
        grids = {"col": i % ncols + 1, "row": i // ncols + 1}
        scatter_grids = {"row": grids["row"] * 2 - 1, "col": grids["col"] * 2 - 1}
        cb_grids = {"row": grids["row"] * 2, "col": grids["col"] * 2 - 1}
        size_grids = {"row": grids["row"] * 2, "col": grids["col"] * 2}
        nc, nr = grids["col"], grids["row"]
        cb_domain_x, cb_domain_y = plot_style.update_plot_domains(
            fig, nc, nr, (scatter_grids, cb_grids, size_grids), t
        )

        cmap = cmaps[i % len(cmaps)]
        sx = plot_data["x1"].values
        sy = plot_data["x2"].values
        sv = plot_data[t].values
        texts = [
            f"{item['fake_count']}<br>{item[t]:.3f}"
            for item in plot_data.to_dict("records")
        ]
        colors = getattr(px.colors.sequential, cmap)
        line_color = colors[len(colors) // 3]
        edge_color = colors[-1]

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

        colorbar = go.Figure(
            data=go.Heatmap(z=[np.arange(1000)], showscale=False, colorscale=cmap),
        )
        fig.add_trace(colorbar.data[0], **cb_grids)
        fig.update_yaxes(showticklabels=False, **cb_grids)
        fig.update_xaxes(showticklabels=False, **cb_grids)

        cb_texts += [
            go.layout.Annotation(
                x=cb_domain_x[0],
                y=np.mean(cb_domain_y),
                xref="paper",
                yref="paper",
                text=str(round(np.min(sv), 3)),
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(color=edge_color),
            ),
            go.layout.Annotation(
                x=cb_domain_x[1],
                y=np.mean(cb_domain_y),
                xref="paper",
                yref="paper",
                text=str(round(np.max(sv), 3)),
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                font=dict(color="#ffffff"),
            ),
        ]

        fig.add_trace(
            go.Scatter(
                x=[0.25, 0.75],
                y=[0.5, 0.5],
                text=[count_min, count_max],
                mode="markers+lines+text",
                marker=dict(
                    size=[15, 30],
                    color=line_color,
                ),
            ),
            **size_grids,
        )
        fig.update_yaxes(visible=False, **size_grids)
        fig.update_xaxes(visible=False, range=[0, 1], **size_grids)

    fig.update_layout(annotations=cb_texts)

    return fig


def _draw_barplot(df, axes, xticklabels, plot_style):
    bar_style = plot_style.bar

    rects = axes.bar(
        x=df["x"],
        height=df["count"],
        width=bar_style["width"],
        color=bar_style["color"],
        alpha=bar_style["alpha"],
    )

    for rect in rects:
        height = rect.get_height()
        bbox_props = {
            "facecolor": "white",
            "edgecolor": bar_style["color"],
            "boxstyle": "square,pad=0.5",
        }
        axes.text(
            rect.get_x() + rect.get_width() / 2.0,
            height,
            str(int(height)),
            ha="center",
            va="center",
            bbox=bbox_props,
            color=bar_style["color"],
            fontdict=bar_style["fontdict"],
        )

    axes.set_ylabel("count", fontdict=plot_style.label["fontdict"])
    axes.set_xticks(range(len(xticklabels)), labels=xticklabels)
    axes.set_xlim(-0.5, len(xticklabels) - 0.5)
    _axes_modify(axes, plot_style)


def _draw_barplot_plotly(fig, df, plot_style, grids):
    bar_style = plot_style.bar
    fig.add_trace(
        go.Bar(
            x=df["x"],
            y=df["count"],
            text=df["count"],
            textposition="outside",
            width=bar_style["width"],
            name="count",
            marker=dict(
                color=bar_style["color"],
                opacity=0.5,
            ),
        ),
        secondary_y=False,
        **grids,
    )
    fig.update_yaxes(range=[0, df["count"].max() * 1.2], secondary_y=False, **grids)


def _draw_lineplot(ylabel, df, axes, line_color, plot_style):
    line_style = plot_style.line

    axes.plot(df["x"], df["y"], linewidth=line_style["width"], c=line_color, marker="o")

    for _, row in df.iterrows():
        x, y = row["x"], row["y"]
        bbox_props = {
            "facecolor": line_color,
            "edgecolor": "none",
            "boxstyle": "square,pad=0.5",
        }

        axes.text(
            x,
            y,
            f"{y:.3f}",
            ha="center",
            va="top",
            bbox=bbox_props,
            color="#ffffff",
            fontdict=line_style["fontdict"],
        )

    axes.set_ylabel(ylabel, fontdict=plot_style.label["fontdict"])


def _draw_lineplot_plotly(target, fig, df, line_color, grids, y2=True):
    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            text=[round(v, 3) for v in df["y"].values],
            textposition="top center",
            mode="lines+markers+text",
            yaxis="y2",
            name=target,
            line=dict(color=line_color),
            marker=dict(color=line_color),
        ),
        secondary_y=y2,
        **grids,
    )


def _prepare_box_data(df):
    xs = sorted(df["x"].unique())
    ys = []
    for x in xs:
        ys.append(df[df["x"] == x]["y"].values)
    return xs, ys


def _draw_boxplot(box_df, line_df, axes, box_color, plot_style):
    """Draw box plot"""
    box_style = plot_style.box
    xs, ys = _prepare_box_data(box_df)
    common_props = dict(
        linewidth=box_style["line_width"], color=box_color, alpha=box_style["alpha"]
    )
    boxplot_args = {
        "whiskerprops": common_props,
        "capprops": common_props,
        "boxprops": common_props,
        "medianprops": dict(linewidth=0),
    }

    axes.boxplot(
        ys,
        positions=xs,
        showfliers=False,
        widths=box_style["width"],
        **boxplot_args,
    )

    _draw_lineplot("", line_df, axes, box_color, plot_style)
    _axes_modify(axes, plot_style, right=False, grid=True)


def _draw_boxplot_plotly(target, box_df, line_df, fig, box_color, plot_style, grids):
    box_style = plot_style.box
    xs, ys = _prepare_box_data(box_df)
    for i, y in enumerate(ys):
        fig.add_trace(
            go.Box(
                x0=i,
                y=y,
                width=box_style["width"],
                marker=dict(
                    color=box_color,
                    opacity=box_style["alpha"],
                ),
                boxpoints=False,
                name=target,
            ),
            **grids,
        )
    _draw_lineplot_plotly(target, fig, line_df, box_color, grids, y2=False)


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


def _plot_interact(
    feat_names,
    target,
    plot_data,
    plot_axes,
    plot_style,
    marker_sizes,
    cmap,
):
    """Interact scatter plot"""
    xs, ys = plot_style.display_columns
    plot_axes.set_xticks(
        range(len(xs)), xs, rotation=plot_style.tick["xticks_rotation"]
    )
    plot_axes.set_xlim(-0.5, len(xs) - 0.5)
    plot_axes.set_yticks(range(len(ys)), ys)
    plot_axes.set_ylim(-0.5, len(ys) - 0.5)

    per_xaxes, per_yaxes = _display_percentile_inter(
        plot_style, plot_axes, x=True, y=True
    )
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
    res = _check_interact_plot_params(
        df,
        features,
        grid_types,
        percentile_ranges,
        grid_ranges,
        num_grid_points,
        cust_grid_points,
    )

    if not show_outliers:
        show_outliers = [False, False]
    else:
        show_outliers = [True, True]
        for i, vs in enumerate(
            zip(res["percentile_ranges"], res["grid_ranges"], res["cust_grid_points"])
        ):
            if all(v is None for v in vs):
                show_outliers[i] = False
    res["show_outliers"] = show_outliers

    return res
