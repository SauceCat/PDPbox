from .styles import (
    _prepare_plot_style,
    _axes_modify,
    _display_percentile,
    _display_ticks_plotly,
    _modify_legend_axes,
)

import numpy as np
from itertools import cycle
import plotly.express as px

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patheffects as PathEffects

import plotly.graph_objects as go


def _target_plot(plot_obj, which_classes, plot_params):
    feature_info = plot_obj.feature_info
    plot_style = _prepare_plot_style(
        feature_info.name, len(which_classes), plot_params, plot_obj.plot_type
    )
    fig, inner_grid, title_axes = plot_style.make_subplots()

    axes = {"title_axes": title_axes, "bar_axes": [], "line_axes": []}
    line_colors = cycle(plot_style.line["colors"])

    for i, class_idx in enumerate(which_classes):
        target = plot_obj.target[class_idx]
        line_color = next(line_colors)
        bar_axes = plt.subplot(inner_grid[i])
        line_axes = bar_axes.twinx()
        line_data = (
            plot_obj.target_lines[class_idx]
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

        xlabel = "value"
        if plot_style.show_percentile and len(feature_info.percentile_columns) > 0:
            _display_percentile(bar_axes, feature_info.percentile_columns, plot_style)
            xlabel += " + percentile"
        bar_axes.set_xlabel(xlabel, fontdict=plot_style.label["fontdict"])

        axes["bar_axes"].append(bar_axes)
        axes["line_axes"].append(line_axes)

    return fig, axes


def _target_plot_plotly(plot_obj, which_classes, plot_params):
    feature_info = plot_obj.feature_info
    plot_style = _prepare_plot_style(
        feature_info.name, len(which_classes), plot_params, plot_obj.plot_type
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

    for i, class_idx in enumerate(which_classes):
        target = plot_obj.target[class_idx]
        grids = {"col": i % ncols + 1, "row": i // ncols + 1}
        nr, nc = grids["row"], grids["col"]
        title = plot_style.update_plot_domains(fig, nr, nc, grids, target)
        subplot_titles.append(title)
        line_color = next(line_colors)
        line_data = (
            plot_obj.target_lines[class_idx]
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


def _predict_plot(plot_obj, which_classes, plot_params):
    feature_info = plot_obj.feature_info
    plot_style = _prepare_plot_style(
        feature_info.name, len(which_classes), plot_params, plot_obj.plot_type
    )
    fig, inner_grid, title_axes = plot_style.make_subplots()
    axes = {"title_axes": title_axes, "bar_axes": [], "box_axes": []}

    box_colors = cycle(plot_style.box["colors"])
    for i, class_idx in enumerate(which_classes):
        target = plot_obj.target[class_idx]
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
        line_df = plot_obj.target_lines[class_idx].rename(columns={target + "_q2": "y"})
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

        xlabel = "value"
        if plot_style.show_percentile and len(feature_info.percentile_columns) > 0:
            _display_percentile(box_axes, feature_info.percentile_columns, plot_style)
            xlabel += " + percentile"
        bar_axes.set_xlabel(xlabel, fontdict=plot_style.label["fontdict"])

        axes["bar_axes"].append(bar_axes)
        axes["box_axes"].append(box_axes)

    return fig, axes


def _predict_plot_plotly(plot_obj, which_classes, plot_params):
    feature_info = plot_obj.feature_info
    plot_style = _prepare_plot_style(
        feature_info.name, len(which_classes), plot_params, plot_obj.plot_type
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
    for i, class_idx in enumerate(which_classes):
        target = plot_obj.target[class_idx]
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
        line_df = plot_obj.target_lines[class_idx].rename(columns={target + "_q2": "y"})
        ymin, ymax = box_df["y"].min(), box_df["y"].max()

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
        fig.update_yaxes(title_text=f"{target} dist", range=[ymin, ymax], **box_grids)

    if len(plot_obj.target) > 1:
        fig.update_layout(annotations=subplot_titles)

    return fig, None


def _get_interact_marker_sizes(plot_obj, plot_style):
    count_min, count_max = (
        plot_obj.plot_df["count"].min(),
        plot_obj.plot_df["count"].max(),
    )
    marker_sizes = []
    for count in plot_obj.plot_df["count"].values:
        size = (
            float(count - count_min)
            / (count_max - count_min)
            * (plot_style.marker["max_size"] - plot_style.marker["min_size"])
            + plot_style.marker["min_size"]
        )
        marker_sizes.append(size)

    return count_min, count_max, marker_sizes


def _interact_info_plot(plot_obj, which_classes, plot_params):
    feat_names = [plot_obj.feature_infos[i].name for i in range(2)]
    plot_style = _prepare_plot_style(
        feat_names, len(which_classes), plot_params, plot_obj.plot_type
    )
    count_min, count_max, marker_sizes = _get_interact_marker_sizes(
        plot_obj, plot_style
    )
    fig, inner_grid, title_axes = plot_style.make_subplots()

    axes = {"title_axes": title_axes, "value_axes": [], "legend_axes": []}
    cmaps = cycle(plot_style.marker["cmaps"])

    for i, class_idx in enumerate(which_classes):
        target = plot_obj.target[class_idx]
        cmap = next(cmaps)
        inner = GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=inner_grid[i],
            wspace=plot_style.gaps["inner_x"],
            hspace=plot_style.gaps["inner_y"],
            height_ratios=plot_style.subplot_ratio["y"],
        )
        value_axes = plt.subplot(inner[0])
        value_min, value_max = _draw_interact(
            target,
            plot_obj,
            marker_sizes,
            value_axes,
            cmap,
            plot_style,
        )
        if len(plot_obj.target) > 1:
            value_axes.set_title(
                target,
                **plot_style.title["subplot_title"],
            )

        # draw legend
        if plot_style.ncols > 1:
            n = 2
            legend_grid = GridSpecFromSubplotSpec(
                1, n, subplot_spec=inner[1], wspace=0, width_ratios=[1] * n
            )
        else:
            n = 4
            legend_grid = GridSpecFromSubplotSpec(
                1, n, subplot_spec=inner[1], wspace=0, width_ratios=[1] * n
            )

        legend_axes = []
        for j in range(2):
            subplot = plt.subplot(legend_grid[j])
            legend_axes.append(subplot)

        _draw_legend_colorbar(
            value_min,
            value_max,
            legend_axes[0],
            cmap,
            plot_style.legend["colorbar"],
        )
        _draw_legend_sizes(
            count_min,
            count_max,
            legend_axes[1],
            cmap,
            plot_style.legend["circles"],
        )

        axes["value_axes"].append(value_axes)
        axes["legend_axes"].append(legend_axes)

    return fig, axes


def _interact_info_plot_plotly(plot_obj, which_classes, plot_params):
    feat_names = [plot_obj.feature_infos[i].name for i in range(2)]
    plot_style = _prepare_plot_style(
        feat_names, len(which_classes), plot_params, plot_obj.plot_type
    )
    count_min, count_max, marker_sizes = _get_interact_marker_sizes(
        plot_obj, plot_style
    )
    nrows, ncols = plot_style.nrows, plot_style.ncols
    plot_args = {
        "rows": nrows * 2,
        "cols": ncols * 2,
        "horizontal_spacing": 0,
        "vertical_spacing": 0,
    }
    fig = plot_style.make_subplots_plotly(plot_args)
    subplot_titles = []
    cb_texts = []
    cmaps = cycle(plot_style.marker["cmaps"])
    for i, class_idx in enumerate(which_classes):
        target = plot_obj.target[class_idx]
        grids = {"col": i % ncols + 1, "row": i // ncols + 1}
        scatter_grids = {"row": grids["row"] * 2 - 1, "col": grids["col"] * 2 - 1}
        cb_grids = {"row": grids["row"] * 2, "col": grids["col"] * 2 - 1}
        size_grids = {"row": grids["row"] * 2, "col": grids["col"] * 2}
        nc, nr = grids["col"], grids["row"]
        cb_domain_x, cb_domain_y, title = plot_style.update_plot_domains(
            fig, nc, nr, (scatter_grids, cb_grids, size_grids), target
        )
        subplot_titles.append(title)

        cmap = next(cmaps)
        value_min, value_max = _draw_interact_plotly(
            target,
            plot_obj,
            marker_sizes,
            fig,
            cmap,
            plot_style,
            scatter_grids,
        )
        cb_texts += _draw_legend_colorbar_plotly(
            value_min, value_max, cb_domain_x, cb_domain_y, fig, cmap, cb_grids
        )

        _draw_legend_sizes_plotly(count_min, count_max, fig, cmap, size_grids)

    if len(plot_obj.target) > 1:
        fig.update_layout(annotations=subplot_titles + cb_texts)
    else:
        fig.update_layout(annotations=cb_texts)

    return fig, None


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
    fig.update_yaxes(range=[0, df["y"].max() * 1.2], secondary_y=y2, **grids)


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


def _draw_interact(
    target,
    plot_obj,
    marker_sizes,
    axes,
    cmap,
    plot_style,
):
    feature_infos = plot_obj.feature_infos
    xs, ys = [feature_infos[i].display_columns for i in range(2)]
    lxs, lys = len(xs), len(ys)
    axes.set_xticks(range(lxs), labels=xs)
    axes.set_xlim(-0.5, lxs - 0.5)
    axes.set_yticks(range(lys), labels=ys)
    axes.set_ylim(-0.5, lys - 0.5)

    plot_df = plot_obj.plot_df
    value_min, value_max = plot_df[target].min(), plot_df[target].max()
    if value_max == value_min:
        colors = [plt.get_cmap(cmap)(0.0)] * len(plot_df)
    else:
        colors = [
            plt.get_cmap(cmap)(float(v - value_min) / (value_max - value_min))
            for v in plot_df[target].values
        ]

    axes.scatter(
        plot_df["x1"].values,
        plot_df["x2"].values,
        s=marker_sizes,
        c=colors,
        linewidth=plot_style.marker["line_width"],
        edgecolors=plt.get_cmap(cmap)(1.0),
    )

    if plot_style.annotate:
        for item in plot_df.to_dict("records"):
            text = axes.text(
                x=item["x1"],
                y=item["x2"],
                s=f"{item['count']}\n{item[target]:.3f}",
                fontdict={
                    "family": plot_style.font_family,
                    "color": plt.get_cmap(cmap)(1.0),
                    "size": plot_style.marker["fontsize"],
                },
                va="center",
                ha="left",
            )
            text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="w")])

    xlabel, ylabel = "value", "value"
    if plot_style.show_percentile:
        if len(feature_infos[0].percentile_columns) > 0:
            _display_percentile(
                axes,
                feature_infos[0].percentile_columns,
                plot_style,
                is_x=True,
                is_y=False,
            )
            xlabel += " + percentile"
        if len(feature_infos[1].percentile_columns) > 0:
            _display_percentile(
                axes,
                feature_infos[1].percentile_columns,
                plot_style,
                is_x=False,
                is_y=True,
            )
            ylabel += " + percentile"
    axes.set_xlabel(
        feature_infos[0].name + f"({xlabel})", fontdict=plot_style.label["fontdict"]
    )
    axes.set_ylabel(
        feature_infos[1].name + f"({ylabel})", fontdict=plot_style.label["fontdict"]
    )
    _axes_modify(axes, plot_style)

    return value_min, value_max


def _draw_interact_plotly(
    target,
    plot_obj,
    marker_sizes,
    fig,
    cmap,
    plot_style,
    grids,
):
    feature_infos = plot_obj.feature_infos
    plot_df = plot_obj.plot_df
    texts = [
        f"{item['count']}<br>{item[target]:.3f}" for item in plot_df.to_dict("records")
    ]
    colors = getattr(px.colors.sequential, cmap)
    value_min, value_max = plot_df[target].min(), plot_df[target].max()

    fig.add_trace(
        go.Scatter(
            x=plot_df["x1"].values,
            y=plot_df["x2"].values,
            text=texts,
            textposition="middle center",
            mode="markers+text" if plot_style.annotate else "markers",
            name="count+"
            + ("target" if plot_obj.plot_type == "interact_target" else "predict"),
            marker=dict(
                size=marker_sizes,
                color=plot_df[target].values,
                colorscale=cmap,
                line=dict(width=plot_style.marker["line_width"], color=colors[-1]),
            ),
        ),
        **grids,
    )

    for i in range(2):
        feat_name = feature_infos[i].name
        title_text = f"<b>{feat_name}</b> (value)"
        ticktext = feature_infos[i].display_columns.copy()
        if plot_style.show_percentile:
            percentile_columns = feature_infos[i].percentile_columns
            if len(percentile_columns) > 0:
                for j, p in enumerate(percentile_columns):
                    ticktext[j] += f"<br><sup><b>{p}</b></sup>"
                title_text = f"<b>{feat_name}</b> (value+percentile)"

        kwargs = dict(
            title_text=title_text,
            ticktext=ticktext,
            tickvals=np.arange(len(ticktext)),
            **grids,
        )
        if i == 0:
            fig.update_xaxes(**kwargs)
        else:
            fig.update_yaxes(**kwargs)

    return value_min, value_max


def _draw_legend_colorbar(value_min, value_max, axes, cmap, style):
    norm = mpl.colors.Normalize(vmin=float(value_min), vmax=float(value_max))
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


def _draw_legend_colorbar_plotly(
    value_min, value_max, cb_domain_x, cb_domain_y, fig, cmap, grids
):
    colorbar = go.Figure(
        data=go.Heatmap(z=[np.arange(1000)], showscale=False, colorscale=cmap),
    )
    fig.add_trace(colorbar.data[0], **grids)
    fig.update_yaxes(showticklabels=False, **grids)
    fig.update_xaxes(showticklabels=False, **grids)
    colors = getattr(px.colors.sequential, cmap)

    anno_kwargs = {
        "y": np.mean(cb_domain_y),
        "xref": "paper",
        "yref": "paper",
        "showarrow": False,
        "yanchor": "middle",
    }

    texts = [
        go.layout.Annotation(
            x=cb_domain_x[0],
            text=str(round(value_min, 3)),
            xanchor="left",
            font=dict(color=colors[-1]),
            **anno_kwargs,
        ),
        go.layout.Annotation(
            x=cb_domain_x[1],
            text=str(round(value_max, 3)),
            xanchor="right",
            font=dict(color="#ffffff"),
            **anno_kwargs,
        ),
    ]
    return texts


def _draw_legend_sizes(count_min, count_max, axes, cmap, style):
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


def _draw_legend_sizes_plotly(count_min, count_max, fig, cmap, grids):
    colors = getattr(px.colors.sequential, cmap)
    fig.add_trace(
        go.Scatter(
            x=[0.25, 0.75],
            y=[0.5, 0.5],
            text=[count_min, count_max],
            mode="markers+lines+text",
            marker=dict(
                size=[15, 30],
                color=colors[2],
            ),
        ),
        **grids,
    )
    fig.update_yaxes(visible=False, **grids)
    fig.update_xaxes(visible=False, range=[0, 1], **grids)
