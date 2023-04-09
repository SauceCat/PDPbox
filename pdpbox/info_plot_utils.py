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


class BaseInfoPlotEngine:
    def __init__(
        self,
        plot_obj,
        which_classes,
        plot_params,
    ):
        if plot_obj.plot_type in ["target", "predict"]:
            feat_name = plot_obj.feature_info.name
        else:
            feat_name = [plot_obj.feature_infos[i].name for i in range(2)]

        self.plot_style = _prepare_plot_style(
            feat_name,
            len(which_classes),
            plot_params,
            plot_obj.plot_type,
        )
        self.which_classes = which_classes
        self.plot_obj = plot_obj

    def set_subplot_title(self, axes, title):
        if len(self.plot_obj.target) > 1:
            axes.set_title(
                title,
                **self.plot_style.title["subplot_title"],
            )

    def plot(self):
        return (
            self.plot_matplotlib()
            if self.plot_style.engine == "matplotlib"
            else self.plot_plotly()
        )

    def plot_matplotlib(self):
        raise NotImplementedError("Subclasses must implement plot_matplotlib method.")

    def plot_plotly(self):
        raise NotImplementedError("Subclasses must implement plot_plotly method.")


class InfoPlotEngine(BaseInfoPlotEngine):
    def __init__(
        self,
        plot_obj,
        which_classes,
        plot_params,
    ):
        super().__init__(
            plot_obj,
            which_classes,
            plot_params,
        )
        self.display_columns = self.plot_obj.feature_info.display_columns
        self.percentile_columns = self.plot_obj.feature_info.percentile_columns

    def _draw_barplot(self, axes):
        df = self.plot_obj.count_df
        bar_style = self.plot_style.bar

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

        axes.set_ylabel("count", fontdict=self.plot_style.label["fontdict"])
        axes.set_xticks(range(len(self.display_columns)), labels=self.display_columns)
        axes.set_xlim(-0.5, len(self.display_columns) - 0.5)
        _axes_modify(axes, self.plot_style)

    def _draw_barplot_plotly(self, fig, grids):
        df = self.plot_obj.count_df
        bar_style = self.plot_style.bar

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

    def _draw_lineplot(self, ylabel, df, axes, line_color):
        line_style = self.plot_style.line
        axes.plot(
            df["x"], df["y"], linewidth=line_style["width"], c=line_color, marker="o"
        )

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

        axes.set_ylabel(ylabel, fontdict=self.plot_style.label["fontdict"])

    def _draw_lineplot_plotly(self, target, df, line_color, fig, grids, y2=True):
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

    def _prepare_box_data(self, df):
        xs = sorted(df["x"].unique())
        ys = []
        for x in xs:
            ys.append(df[df["x"] == x]["y"].values)
        return xs, ys

    def _draw_boxplot(self, box_df, line_df, axes, box_color):
        box_style = self.plot_style.box
        xs, ys = self._prepare_box_data(box_df)
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

        self._draw_lineplot("", line_df, axes, box_color)
        _axes_modify(axes, self.plot_style, right=False, grid=True)

    def _draw_boxplot_plotly(self, target, box_df, line_df, box_color, fig, grids):
        box_style = self.plot_style.box
        xs, ys = self._prepare_box_data(box_df)
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
        self._draw_lineplot_plotly(target, line_df, box_color, fig, grids, y2=False)

    def wrapup_subplot(self, title, bar_axes, box_axes=None):
        title_axes = box_axes or bar_axes
        self.set_subplot_title(title_axes, title)

        xlabel = "value"
        if self.plot_style.show_percentile and len(self.percentile_columns) > 0:
            _display_percentile(title_axes, self.percentile_columns, self.plot_style)
            xlabel += " + percentile"
        bar_axes.set_xlabel(xlabel, fontdict=self.plot_style.label["fontdict"])

    def wrapup_subplot_plotly(
        self, target, fig, bar_grids, box_grids=None, yrange=None
    ):
        _display_ticks_plotly(
            self.display_columns,
            self.percentile_columns,
            fig,
            bar_grids,
            is_y=False,
        )
        fig.update_yaxes(title_text="count", secondary_y=False, **bar_grids)
        if box_grids is None:
            fig.update_yaxes(
                title_text=f"Average {target}",
                secondary_y=True,
                showgrid=False,
                **bar_grids,
            )
        else:
            fig.update_yaxes(title_text=f"{target} dist", range=yrange, **box_grids)


class TargetPlotEngine(InfoPlotEngine):
    def __init__(
        self,
        plot_obj,
        which_classes,
        plot_params,
    ):
        super().__init__(
            plot_obj,
            which_classes,
            plot_params,
        )
        self.line_colors = cycle(self.plot_style.line["colors"])

    def prepare_subplot(self, class_idx):
        target = self.plot_obj.target[class_idx]
        line_color = next(self.line_colors)
        line_df = (
            self.plot_obj.target_lines[class_idx]
            .rename(columns={target: "y"})
            .sort_values("x", ascending=True)
        )
        return target, line_df, line_color

    def plot_matplotlib(self):
        fig, inner_grid, title_axes = self.plot_style.make_subplots()
        axes = {"title_axes": title_axes, "bar_axes": [], "line_axes": []}

        for i, class_idx in enumerate(self.which_classes):
            target, line_df, line_color = self.prepare_subplot(class_idx)
            bar_axes = plt.subplot(inner_grid[i])
            line_axes = bar_axes.twinx()

            self._draw_barplot(bar_axes)
            self._draw_lineplot(
                f"Average {target}",
                line_df,
                line_axes,
                line_color,
            )
            _axes_modify(line_axes, self.plot_style, right=True, grid=False)

            self.wrapup_subplot(target, bar_axes, box_axes=None)
            axes["bar_axes"].append(bar_axes)
            axes["line_axes"].append(line_axes)

        return fig, axes

    def plot_plotly(self):
        nrows, ncols = self.plot_style.nrows, self.plot_style.ncols
        plot_args = {
            "rows": nrows,
            "cols": ncols,
            "specs": [
                [{"secondary_y": True} for _ in range(ncols)] for _ in range(nrows)
            ],
            "horizontal_spacing": 0,
            "vertical_spacing": 0,
        }
        fig = self.plot_style.make_subplots_plotly(plot_args)
        subplot_titles = []

        for i, class_idx in enumerate(self.which_classes):
            target, line_df, line_color = self.prepare_subplot(class_idx)
            grids = {"col": i % ncols + 1, "row": i // ncols + 1}
            nr, nc = grids["row"], grids["col"]
            title = self.plot_style.update_plot_domains(fig, nr, nc, grids, target)
            subplot_titles.append(title)

            self._draw_barplot_plotly(fig, grids)
            self._draw_lineplot_plotly(target, line_df, line_color, fig, grids)
            self.wrapup_subplot_plotly(target, fig, grids, box_grids=None, yrange=None)

        if len(self.plot_obj.target) > 1:
            fig.update_layout(annotations=subplot_titles)

        return fig, None


class PredictPlotEngine(InfoPlotEngine):
    def __init__(
        self,
        plot_obj,
        which_classes,
        plot_params,
    ):
        super().__init__(
            plot_obj,
            which_classes,
            plot_params,
        )
        self.box_colors = cycle(self.plot_style.box["colors"])

    def prepare_subplot(self, class_idx):
        target = self.plot_obj.target[class_idx]
        box_color = next(self.box_colors)
        box_df = self.plot_obj.df[["x", target]].rename(columns={target: "y"})
        line_df = self.plot_obj.target_lines[class_idx].rename(
            columns={target + "_q2": "y"}
        )
        return target, box_df, line_df, box_color

    def plot_matplotlib(self):
        fig, inner_grid, title_axes = self.plot_style.make_subplots()
        axes = {"title_axes": title_axes, "bar_axes": [], "box_axes": []}

        for i, class_idx in enumerate(self.which_classes):
            target, box_df, line_df, box_color = self.prepare_subplot(class_idx)
            inner = GridSpecFromSubplotSpec(
                2,
                1,
                subplot_spec=inner_grid[i],
                wspace=self.plot_style.gaps["inner_x"],
                hspace=self.plot_style.gaps["inner_y"],
            )
            box_axes = plt.subplot(inner[0])
            bar_axes = plt.subplot(inner[1])

            self._draw_boxplot(box_df, line_df, box_axes, box_color)
            box_axes.set(ylabel=f"{target} dist", xticklabels=[])
            self._draw_barplot(bar_axes)

            self.wrapup_subplot(target, bar_axes, box_axes)
            axes["bar_axes"].append(bar_axes)
            axes["box_axes"].append(box_axes)

        return fig, axes

    def plot_plotly(self):
        nrows, ncols = self.plot_style.nrows, self.plot_style.ncols
        plot_args = {
            "rows": nrows * 2,
            "cols": ncols,
            "shared_xaxes": True,
            "horizontal_spacing": 0,
            "vertical_spacing": 0,
        }
        fig = self.plot_style.make_subplots_plotly(plot_args)
        subplot_titles = []

        for i, class_idx in enumerate(self.which_classes):
            target, box_df, line_df, box_color = self.prepare_subplot(class_idx)
            grids = {"col": i % ncols + 1, "row": i // ncols + 1}
            box_grids = {"col": grids["col"], "row": grids["row"] * 2 - 1}
            bar_grids = {"col": grids["col"], "row": grids["row"] * 2}
            nr, nc = grids["row"], grids["col"]
            title = self.plot_style.update_plot_domains(
                fig, nr, nc, (box_grids, bar_grids), target
            )
            subplot_titles.append(title)

            self._draw_boxplot_plotly(
                target, box_df, line_df, box_color, fig, box_grids
            )
            self._draw_barplot_plotly(fig, bar_grids)

            ymin, ymax = box_df["y"].min(), box_df["y"].max()
            self.wrapup_subplot_plotly(
                target, fig, bar_grids, box_grids, yrange=[ymin, ymax]
            )

        if len(self.plot_obj.target) > 1:
            fig.update_layout(annotations=subplot_titles)

        return fig, None


class InteractInfoPlotEngine(BaseInfoPlotEngine):
    def __init__(
        self,
        plot_obj,
        which_classes,
        plot_params,
    ):
        super().__init__(
            plot_obj,
            which_classes,
            plot_params,
        )
        self.feat_names = [self.plot_obj.feature_infos[i].name for i in range(2)]
        self.display_columns = [
            self.plot_obj.feature_infos[i].display_columns for i in range(2)
        ]
        self.percentile_columns = [
            self.plot_obj.feature_infos[i].percentile_columns for i in range(2)
        ]
        self.cmaps = cycle(self.plot_style.marker["cmaps"])
        (
            self.count_min,
            self.count_max,
            self.marker_sizes,
        ) = self._get_interact_marker_sizes()

    def _get_interact_marker_sizes(self):
        df = self.plot_obj.plot_df
        marker_style = self.plot_style.marker
        count_min, count_max = df["count"].min(), df["count"].max()

        if count_min == count_max:
            marker_sizes = [marker_style["min_size"]] * len(df["count"].values)
        else:
            marker_sizes = [
                (
                    (count - count_min)
                    / (count_max - count_min)
                    * (marker_style["max_size"] - marker_style["min_size"])
                    + marker_style["min_size"]
                )
                for count in df["count"].values
            ]

        return count_min, count_max, marker_sizes

    def prepare_subplot(self, class_idx):
        target = self.plot_obj.target[class_idx]
        cmap = next(self.cmaps)
        return target, cmap

    def _draw_interact(self, target, axes, cmap):
        xs, ys = self.display_columns
        lxs, lys = len(xs), len(ys)
        axes.set(
            xticks=range(lxs),
            xticklabels=xs,
            xlim=(-0.5, lxs - 0.5),
            yticks=range(lys),
            yticklabels=ys,
            ylim=(-0.5, lys - 0.5),
        )

        df = self.plot_obj.plot_df
        value_min, value_max = df[target].min(), df[target].max()
        cmap_func = plt.get_cmap(cmap)
        colors = (
            [cmap_func(0.0)] * len(df)
            if value_max == value_min
            else [
                cmap_func((v - value_min) / (value_max - value_min))
                for v in df[target].values
            ]
        )

        marker_style = self.plot_style.marker
        axes.scatter(
            df["x1"].values,
            df["x2"].values,
            s=self.marker_sizes,
            c=colors,
            linewidth=marker_style["line_width"],
            edgecolors=cmap_func(1.0),
        )

        if self.plot_style.annotate:
            for item in df.to_dict("records"):
                text = axes.text(
                    x=item["x1"],
                    y=item["x2"],
                    s=f"{item['count']}\n{item[target]:.3f}",
                    fontdict={
                        "family": self.plot_style.font_family,
                        "color": cmap_func(1.0),
                        "size": marker_style["fontsize"],
                    },
                    va="center",
                    ha="left",
                )
                text.set_path_effects(
                    [PathEffects.withStroke(linewidth=3, foreground="w")]
                )

        xlabel, ylabel = "value", "value"
        if self.plot_style.show_percentile:
            if len(self.percentile_columns[0]) > 0:
                _display_percentile(
                    axes,
                    self.percentile_columns[0],
                    self.plot_style,
                    is_x=True,
                    is_y=False,
                )
                xlabel += " + percentile"
            if len(self.percentile_columns[1]) > 0:
                _display_percentile(
                    axes,
                    self.percentile_columns[1],
                    self.plot_style,
                    is_x=False,
                    is_y=True,
                )
                ylabel += " + percentile"
        axes.set_xlabel(
            self.feat_names[0] + f"({xlabel})",
            fontdict=self.plot_style.label["fontdict"],
        )
        axes.set_ylabel(
            self.feat_names[1] + f"({ylabel})",
            fontdict=self.plot_style.label["fontdict"],
        )
        _axes_modify(axes, self.plot_style)

        return value_min, value_max

    def _draw_interact_plotly(self, target, cmap, fig, grids):
        df = self.plot_obj.plot_df
        texts = [
            f"{item['count']}<br>{item[target]:.3f}" for item in df.to_dict("records")
        ]
        colors = getattr(px.colors.sequential, cmap)
        value_min, value_max = df[target].min(), df[target].max()

        name = "target" if self.plot_obj.plot_type == "interact_target" else "predict"
        fig.add_trace(
            go.Scatter(
                x=df["x1"].values,
                y=df["x2"].values,
                text=texts,
                textposition="middle center",
                mode="markers+text" if self.plot_style.annotate else "markers",
                name="count+" + name,
                marker=dict(
                    size=self.marker_sizes,
                    color=df[target].values,
                    colorscale=cmap,
                    line=dict(
                        width=self.plot_style.marker["line_width"], color=colors[-1]
                    ),
                ),
            ),
            **grids,
        )

        for i, feat_name in enumerate(self.feat_names):
            title_text = f"<b>{feat_name}</b> (value)"
            ticktext = self.display_columns[i].copy()
            percentile_columns = self.percentile_columns[i]

            if self.plot_style.show_percentile and len(percentile_columns) > 0:
                ticktext = [
                    f"{text}<br><sup><b>{p}</b></sup>"
                    for text, p in zip(ticktext, percentile_columns)
                ]
                title_text = f"<b>{feat_name}</b> (value+percentile)"

            kwargs = dict(
                title_text=title_text,
                ticktext=ticktext,
                tickvals=np.arange(len(ticktext)),
                **grids,
            )
            (fig.update_xaxes if i == 0 else fig.update_yaxes)(**kwargs)

        return value_min, value_max

    def _draw_legend_colorbar(self, value_min, value_max, axes, cmap):
        cb_style = self.plot_style.legend["colorbar"]
        norm = mpl.colors.Normalize(vmin=float(value_min), vmax=float(value_max))
        cax = inset_axes(
            axes, height=cb_style["height"], width=cb_style["width"], loc=10
        )
        cb = mpl.colorbar.ColorbarBase(
            cax, cmap=plt.get_cmap(cmap), norm=norm, orientation="horizontal"
        )
        cb.outline.set_linewidth(0)
        cb.set_ticks([])

        width_float = float(cb_style["width"].replace("%", "")) / 100
        text_params = {
            "fontdict": {
                "color": plt.get_cmap(cmap)(1.0),
                "fontsize": cb_style["fontsize"],
                "fontfamily": cb_style["fontfamily"],
            },
            "transform": axes.transAxes,
            "va": "center",
        }

        for pos, value, align in [
            ((1.0 - width_float) / 2, value_min, "left"),
            (1.0 - (1.0 - width_float) / 2, value_max, "right"),
        ]:
            text = cb.ax.text(pos, 0.5, f"{value:.3f}", ha=align, **text_params)
            text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="w")])

        _modify_legend_axes(axes, cb_style["fontfamily"])

    def _draw_legend_colorbar_plotly(
        self, value_min, value_max, cb_domain_x, cb_domain_y, cmap, fig, grids
    ):
        colorbar = go.Figure(
            data=go.Heatmap(z=[np.arange(1000)], showscale=False, colorscale=cmap)
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

        annotations = []
        for val, x, xanchor, color in [
            (value_min, cb_domain_x[0], "left", colors[-1]),
            (value_max, cb_domain_x[1], "right", "#ffffff"),
        ]:
            annotation = go.layout.Annotation(
                x=x,
                text=str(round(val, 3)),
                xanchor=xanchor,
                font=dict(color=color),
                **anno_kwargs,
            )
            annotations.append(annotation)

        return annotations

    def _draw_legend_sizes(self, axes, cmap):
        size_style = self.plot_style.legend["circles"]
        dark_color = plt.get_cmap(cmap)(1.0)
        xs, ys = [0.75, 2], [1] * 2

        axes.plot(xs, ys, color=dark_color, zorder=1, ls="--", lw=1)
        axes.scatter(
            xs, ys, s=[200, 500], edgecolors=dark_color, color="white", zorder=2
        )
        axes.set_xlim(0.0, 3)

        text_params = {
            "fontdict": {
                "color": dark_color,
                "fontsize": size_style["fontsize"],
                "fontfamily": size_style["fontfamily"],
            },
            "ha": "center",
            "va": "center",
        }

        for x, count in zip(xs, [self.count_min, self.count_max]):
            count_text = axes.text(x, 1, count, **text_params)
            count_text.set_path_effects(
                [PathEffects.withStroke(linewidth=3, foreground="w")]
            )

        _modify_legend_axes(axes, size_style["fontfamily"])

    def _draw_legend_sizes_plotly(self, cmap, fig, grids):
        color = getattr(px.colors.sequential, cmap)[2]
        fig.add_trace(
            go.Scatter(
                x=[0.25, 0.75],
                y=[0.5, 0.5],
                text=[self.count_min, self.count_max],
                mode="markers+lines+text",
                marker=dict(
                    size=[15, 30],
                    color=color,
                ),
            ),
            **grids,
        )
        fig.update_yaxes(visible=False, **grids)
        fig.update_xaxes(visible=False, range=[0, 1], **grids)

    def plot_matplotlib(self):
        fig, inner_grid, title_axes = self.plot_style.make_subplots()
        axes = {"title_axes": title_axes, "value_axes": [], "legend_axes": []}

        for i, class_idx in enumerate(self.which_classes):
            target, cmap = self.prepare_subplot(class_idx)
            inner = GridSpecFromSubplotSpec(
                2,
                1,
                subplot_spec=inner_grid[i],
                wspace=self.plot_style.gaps["inner_x"],
                hspace=self.plot_style.gaps["inner_y"],
                height_ratios=self.plot_style.subplot_ratio["y"],
            )
            value_axes = plt.subplot(inner[0])
            value_min, value_max = self._draw_interact(target, value_axes, cmap)

            n = 2 if self.plot_style.ncols > 1 else 4
            legend_grid = GridSpecFromSubplotSpec(
                1, n, subplot_spec=inner[1], wspace=0, width_ratios=[1] * n
            )

            legend_axes = [plt.subplot(legend_grid[j]) for j in range(2)]
            self._draw_legend_colorbar(value_min, value_max, legend_axes[0], cmap)
            self._draw_legend_sizes(legend_axes[1], cmap)

            self.set_subplot_title(value_axes, target)
            axes["value_axes"].append(value_axes)
            axes["legend_axes"].append(legend_axes)

        return fig, axes

    def plot_plotly(self):
        nrows, ncols = self.plot_style.nrows, self.plot_style.ncols
        plot_args = {
            "rows": nrows * 2,
            "cols": ncols * 2,
            "horizontal_spacing": 0,
            "vertical_spacing": 0,
        }
        fig = self.plot_style.make_subplots_plotly(plot_args)
        subplot_titles = []
        cb_texts = []

        for i, class_idx in enumerate(self.which_classes):
            target, cmap = self.prepare_subplot(class_idx)
            grids = {"col": i % ncols + 1, "row": i // ncols + 1}
            scatter_grids = {"row": grids["row"] * 2 - 1, "col": grids["col"] * 2 - 1}
            cb_grids = {"row": grids["row"] * 2, "col": grids["col"] * 2 - 1}
            size_grids = {"row": grids["row"] * 2, "col": grids["col"] * 2}

            cb_domain_x, cb_domain_y, title = self.plot_style.update_plot_domains(
                fig,
                grids["col"],
                grids["row"],
                (scatter_grids, cb_grids, size_grids),
                target,
            )
            subplot_titles.append(title)

            value_min, value_max = self._draw_interact_plotly(
                target, cmap, fig, scatter_grids
            )
            cb_texts += self._draw_legend_colorbar_plotly(
                value_min, value_max, cb_domain_x, cb_domain_y, cmap, fig, cb_grids
            )
            self._draw_legend_sizes_plotly(cmap, fig, size_grids)

        annotations = (
            subplot_titles + cb_texts if len(self.plot_obj.target) > 1 else cb_texts
        )
        fig.update_layout(annotations=annotations)

        return fig, None
