from .utils import (
    _get_string,
    _to_rgba,
)
from .styles import (
    _prepare_plot_style,
    _axes_modify,
    _display_percentile,
    _modify_legend_axes,
    _get_bold_text,
)

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

import copy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import plotly.graph_objects as go

from sklearn.cluster import MiniBatchKMeans, KMeans
from itertools import cycle


class PDPIsolatePlotEngine:
    def __init__(
        self,
        plot_obj,
        which_classes,
        plot_params,
    ):
        self.plot_style = _prepare_plot_style(
            plot_obj.feature_info.name,
            len(which_classes),
            plot_params,
            plot_obj.plot_type,
        )
        self.which_classes = which_classes
        self.plot_obj = plot_obj
        self.cmaps = cycle(self.plot_style.line["cmaps"])
        self.display_columns = self.plot_obj.feature_info.display_columns
        self.percentile_columns = self.plot_obj.feature_info.percentile_columns
        self.grids, self.percentiles = [], []
        if self.plot_obj.feature_info.type == "numeric":
            self.grids = [_get_string(v) for v in self.plot_obj.feature_info.grids]
            self.percentiles = self.plot_obj.feature_info.percentiles

    def plot(self):
        return (
            self.plot_matplotlib()
            if self.plot_style.engine == "matplotlib"
            else self.plot_plotly()
        )

    def prepare_subplot(self, class_idx):
        target = self.plot_obj.target[class_idx]
        cmap = next(self.cmaps)
        colors = [cmap, plt.get_cmap(cmap)(0.1), plt.get_cmap(cmap)(1.0)]
        return target, cmap, colors

    def _cluster_ice_lines(self, ice_lines, feature_grids):
        method = self.plot_style.clustering["method"]
        if method not in ["approx", "accurate"]:
            raise ValueError('cluster method: should be "approx" or "accurate".')

        n_centers = self.plot_style.clustering["n_centers"]
        if method == "approx":
            kmeans = MiniBatchKMeans(n_clusters=n_centers, random_state=0, verbose=0)
        else:
            kmeans = KMeans(n_clusters=n_centers, random_state=0)

        kmeans.fit(ice_lines[np.arange(len(feature_grids))])
        return pd.DataFrame(kmeans.cluster_centers_, columns=feature_grids)

    def _sample_ice_lines(self, ice_lines):
        num_samples = self.plot_style.frac_to_plot
        total = len(ice_lines)
        if num_samples <= 1:
            num_samples *= total
        if num_samples < total:
            ice_lines = ice_lines.sample(num_samples, replace=False).reset_index(
                drop=True
            )
        return ice_lines

    def _prepare_pdp_line_data(self, class_idx):
        pdp_result = self.plot_obj.results[class_idx]
        pdp = copy.deepcopy(pdp_result.pdp)
        ice_lines = copy.deepcopy(pdp_result.ice_lines)
        feature_grids = self.plot_obj.feature_info.grids
        grid_indices = np.arange(len(feature_grids))

        x = grid_indices
        if self.plot_obj.feature_info.type == "numeric" and not self.plot_style.to_bins:
            x = feature_grids

        if self.plot_style.center:
            pdp -= pdp[0]
            for i in np.arange(1, len(feature_grids)):
                ice_lines[i] -= ice_lines[0]
            ice_lines[0] = 0

        line_data = None
        if self.plot_style.plot_lines:
            if self.plot_style.clustering["on"]:
                line_data = self._cluster_ice_lines(ice_lines, feature_grids)
            else:
                line_data = self._sample_ice_lines(ice_lines)

        line_std = ice_lines[grid_indices].std().values

        return x, pdp, line_data, line_std

    def _ice_line_plot(self, x, lines, cmap, axes):
        total = len(lines)
        linewidth = np.max([1.0 / np.log10(total), 0.3])
        linealpha = np.min([np.max([1.0 / np.log10(total), 0.3]), 0.8])
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, 20))[5:15]
        if self.plot_style.engine == "matplotlib":
            for i, line in enumerate(lines.values):
                axes.plot(
                    x, line, linewidth=linewidth, c=colors[i % 10], alpha=linealpha
                )
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

    def _pdp_std_plot(self, x, pdp, std, colors, axes):
        upper = pdp + std
        lower = pdp - std
        light_color, dark_color = colors
        line_style = self.plot_style.line

        if self.plot_style.std_fill:
            axes.fill_between(
                x, upper, lower, alpha=line_style["fill_alpha"], color=light_color
            )

        if self.plot_style.pdp_hl:
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

    def _pdp_std_plot_plotly(self, x, pdp, std, colors):
        upper = pdp + std
        lower = pdp - std
        light_color, dark_color = colors
        line_style = self.plot_style.line
        light_color = _to_rgba(light_color, line_style["fill_alpha"])
        dark_color = _to_rgba(dark_color)
        trace_params = {"x": x, "mode": "lines", "hoverinfo": "none"}

        fill_traces = [None, None]
        if self.plot_style.std_fill:
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
        if self.plot_style.pdp_hl:
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
            hovertemplate="%{y}",
            name="pdp",
        )

        return fill_traces + [pdp_hl_trace, zero_trace, pdp_trace]

    def _pdp_line_plot(self, class_idx, colors, axes=None, fig=None, grids=None):
        cmap, light_color, dark_color = colors
        x, pdp, line_data, line_std = self._prepare_pdp_line_data(class_idx)
        ice_line_traces = []
        if self.plot_style.plot_lines:
            ice_line_traces = self._ice_line_plot(x, line_data, cmap, axes)

        std_params = {
            "x": x,
            "pdp": pdp,
            "std": line_std,
            "colors": (light_color, dark_color),
        }
        if self.plot_style.engine == "matplotlib":
            self._pdp_std_plot(axes=axes, **std_params)
            _axes_modify(axes, self.plot_style)
        else:
            std_traces = self._pdp_std_plot_plotly(**std_params)
            for trace in ice_line_traces + std_traces:
                if trace is not None:
                    fig.add_trace(trace, **grids)

    def _get_pdp_xticks(self):
        dist_xticklabels = line_xticklabels = copy.deepcopy(self.display_columns)
        is_numeric = self.plot_obj.feature_info.type == "numeric"
        if is_numeric:
            if self.plot_style.to_bins:
                line_xticklabels = self.grids
            else:
                dist_xticklabels = line_xticklabels = None
                self.plot_style.show_percentile = False

        if self.plot_style.show_percentile and self.plot_style.engine == "plotly":
            for j, p in enumerate(self.percentile_columns):
                dist_xticklabels[j] += f"<br><sup><b>{p}</b></sup>"
            if self.plot_style.to_bins:
                for j, p in enumerate(self.percentiles):
                    line_xticklabels[j] += f"<br><sup><b>{p}</b></sup>"

        if self.plot_style.plot_pts_dist and line_xticklabels is not None:
            line_xticklabels = [""] * len(line_xticklabels)

        return is_numeric, line_xticklabels, dist_xticklabels

    def _set_pdp_xticks(
        self, xticklabels, axes=None, fig=None, grids=None, is_numeric_line=False
    ):
        if xticklabels is not None:
            xticks = np.arange(len(xticklabels))
            if is_numeric_line:
                x_range = [xticks[0], xticks[-1]]
            else:
                x_range = [xticks[0] - 0.5, xticks[-1] + 0.5]

            if self.plot_style.engine == "matplotlib":
                axes.set_xlim(x_range)
                axes.set_xticks(xticks, labels=xticklabels)
            else:
                fig.update_xaxes(
                    ticktext=xticklabels,
                    tickvals=xticks,
                    range=x_range,
                    **grids,
                )

    def _draw_pdp_distplot(self, data, color, axes):
        axes.plot(
            data,
            [1] * len(data),
            "|",
            color=color,
            markersize=self.plot_style.dist["markersize"],
        )
        _modify_legend_axes(axes, self.plot_style.font_family)

    def _draw_pdp_countplot(self, data, cmap, axes):
        count_norm = data["count_norm"].values
        norm = mpl.colors.Normalize(vmin=0, vmax=np.max(count_norm))
        _modify_legend_axes(axes, self.plot_style.font_family)

        xticks = data["x"].values
        axes.imshow(
            np.expand_dims(count_norm, 0),
            aspect="auto",
            cmap=cmap,
            norm=norm,
            alpha=self.plot_style.dist["fill_alpha"],
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
                    "family": self.plot_style.font_family,
                    "fontsize": self.plot_style.dist["font_size"],
                },
            )

        axes.set_xticks(xticks[:-1] + 0.5, minor=True)
        axes.grid(which="minor", color="w", linestyle="-", linewidth=1.5)
        axes.tick_params(which="minor", bottom=False, left=False)

    def _pdp_dist_plot(self, colors, axes, line_axes):
        cmap, light_color, dark_color = colors

        if self.plot_obj.feature_info.type == "numeric" and not self.plot_style.to_bins:
            dist_df = self.plot_obj.dist_df
            self._draw_pdp_distplot(dist_df, dark_color, axes)
            vmin, vmax = dist_df.min(), dist_df.max()
            axes.set_xlim(vmin, vmax)
            axes.set_xticks([])
            line_axes.set_xlim(vmin, vmax)
        else:
            self._draw_pdp_countplot(self.plot_obj.count_df, cmap, axes)

        axes.set_title(
            "distribution of data points",
            fontdict={
                "family": self.plot_style.font_family,
                "color": self.plot_style.tick["labelcolor"],
            },
            fontsize=self.plot_style.tick["labelsize"],
        )
        axes.tick_params(**self.plot_style.tick)

    def _draw_pdp_countplot_plotly(self, df, cmap):
        count_norm = df["count_norm"].values
        xticks = df["x"].values

        heatmap = go.Heatmap(
            x=xticks,
            y=[0],
            z=[count_norm],
            text=[[round(v, 3) for v in count_norm]],
            texttemplate="%{text}",
            hovertemplate="%{text}",
            colorscale=cmap,
            zmin=0,
            zmax=np.max(count_norm),
            opacity=self.plot_style.dist["fill_alpha"],
            showscale=False,
            xgap=2,
            name="dist",
        )
        return heatmap

    def _pdp_dist_plot_plotly(self, colors, fig, grids):
        cmap, light_color, dark_color = colors

        if self.plot_obj.feature_info.type == "numeric" and not self.plot_style.to_bins:
            dist_df = self.plot_obj.dist_df
            dist_trace = go.Scatter(
                x=dist_df,
                y=[1] * len(dist_df),
                mode="text",
                text=["|"] * len(dist_df),
                textposition="middle center",
                textfont=dict(
                    size=self.plot_style.dist["markersize"], color=_to_rgba(dark_color)
                ),
                showlegend=False,
                name="dist",
                hoverinfo="none",
            )
            fig.update_xaxes(showgrid=False, showticklabels=False, **grids)
            fig.update_yaxes(showgrid=False, **grids)
        else:
            dist_trace = self._draw_pdp_countplot_plotly(self.plot_obj.count_df, cmap)

        fig.add_trace(dist_trace, **grids)

    def wrapup_subplot(self, title, line_axes, dist_axes=None):
        if len(self.plot_obj.target) > 1:
            line_axes.set_title(
                title,
                **self.plot_style.title["subplot_title"],
            )

        xlabel = "value"
        if self.plot_style.show_percentile and len(self.percentile_columns) > 0:
            _display_percentile(line_axes, self.percentile_columns, self.plot_style)
            xlabel += " + percentile"
        label_axes = dist_axes or line_axes
        label_axes.set_xlabel(xlabel, fontdict=self.plot_style.label["fontdict"])

    def wrapup_subplot_plotly(self, fig, line_grids, dist_grids=None):
        xlabel = "value"
        if self.plot_style.show_percentile and len(self.percentile_columns) > 0:
            xlabel += " + percentile"
        label_grids = dist_grids or line_grids
        fig.update_xaxes(title_text=xlabel, **label_grids)

    def plot_matplotlib(self):
        fig, inner_grid, title_axes = self.plot_style.make_subplots()
        axes = {"title_axes": title_axes, "line_axes": [], "dist_axes": []}

        for i, class_idx in enumerate(self.which_classes):
            target, cmap, colors = self.prepare_subplot(class_idx)
            if self.plot_style.plot_pts_dist:
                inner = GridSpecFromSubplotSpec(
                    2,
                    1,
                    subplot_spec=inner_grid[i],
                    wspace=self.plot_style.gaps["inner_x"],
                    hspace=self.plot_style.gaps["inner_y"],
                    height_ratios=self.plot_style.subplot_ratio["y"],
                )
                line_axes = plt.subplot(inner[0])
                dist_axes = plt.subplot(inner[1])
            else:
                line_axes = plt.subplot(inner_grid[i])
                dist_axes = None

            self._pdp_line_plot(class_idx, colors, line_axes)
            is_numeric, line_xticklabels, dist_xticklabels = self._get_pdp_xticks()
            self._set_pdp_xticks(
                line_xticklabels, line_axes, is_numeric_line=is_numeric
            )

            if self.plot_style.plot_pts_dist:
                self._pdp_dist_plot(colors, dist_axes, line_axes)
                self._set_pdp_xticks(dist_xticklabels, dist_axes)
                axes["dist_axes"].append(dist_axes)

            self.wrapup_subplot(f"class_{target}", line_axes, dist_axes)
            axes["line_axes"].append(line_axes)

        return fig, axes

    def plot_plotly(self):
        nrows, ncols = self.plot_style.nrows, self.plot_style.ncols
        plot_args = {
            "rows": nrows * (2 if self.plot_style.plot_pts_dist else 1),
            "cols": ncols,
            "horizontal_spacing": 0,
            "vertical_spacing": 0,
        }
        fig = self.plot_style.make_subplots_plotly(plot_args)
        subplot_titles = []

        for i, class_idx in enumerate(self.which_classes):
            target, cmap, colors = self.prepare_subplot(class_idx)
            grids = {"col": i % ncols + 1, "row": i // ncols + 1}
            line_grids = {"col": grids["col"], "row": grids["row"] * 2 - 1}
            dist_grids = None
            nr, nc = grids["row"], grids["col"]

            self._pdp_line_plot(class_idx, colors, None, fig, line_grids)
            is_numeric, line_xticklabels, dist_xticklabels = self._get_pdp_xticks()
            self._set_pdp_xticks(
                line_xticklabels, None, fig, line_grids, is_numeric_line=is_numeric
            )

            if self.plot_style.plot_pts_dist:
                dist_grids = {"col": grids["col"], "row": grids["row"] * 2}
                self._pdp_dist_plot_plotly(colors, fig, dist_grids)
                self._set_pdp_xticks(dist_xticklabels, None, fig, dist_grids)
                fig.update_yaxes(
                    showticklabels=False,
                    **dist_grids,
                )
            title = self.plot_style.update_plot_domains(
                fig, nr, nc, (line_grids, dist_grids), f"class_{target}"
            )
            subplot_titles.append(title)
            self.wrapup_subplot_plotly(fig, line_grids, dist_grids)

        if len(self.plot_obj.target) > 1:
            fig.update_layout(annotations=subplot_titles)

        return fig, None


class PDPInteractPlotEngine:
    def __init__(
        self,
        plot_obj,
        which_classes,
        plot_params,
    ):
        self.which_classes = which_classes
        self.plot_obj = plot_obj
        feature_infos = [
            self.plot_obj.pdp_isolate_objs[i].feature_info for i in range(2)
        ]
        self.display_columns = [feature_infos[i].display_columns for i in range(2)]
        self.percentile_columns = [
            feature_infos[i].percentile_columns for i in range(2)
        ]
        self.feature_names = [feature_infos[i].name for i in range(2)]
        self.feature_grids = [feature_infos[i].grids for i in range(2)]
        self.feature_types = [feature_infos[i].type for i in range(2)]

        self.grids = []
        for i in range(2):
            if self.feature_types[i] == "onehot":
                self.grids.append(feature_infos[i].grids)
            else:
                self.grids.append([_get_string(v) for v in feature_infos[i].grids])

        self.percentiles = [feature_infos[i].percentiles for i in range(2)]
        self.plot_style = _prepare_plot_style(
            self.feature_names,
            len(which_classes),
            plot_params,
            plot_obj.plot_type,
        )
        self.cmaps = cycle(self.plot_style.interact["cmaps"])

    def prepare_subplot(self, class_idx):
        target = self.plot_obj.target[class_idx]
        cmap = next(self.cmaps)
        pdp_x, pdp_y = [
            copy.deepcopy(self.plot_obj.pdp_isolate_objs[i].results[class_idx].pdp)
            for i in range(2)
        ]
        pdp_xy = copy.deepcopy(self.plot_obj.results[class_idx].pdp)

        return pdp_x, pdp_y, pdp_xy, target, cmap

    def plot(self):
        return (
            self.plot_matplotlib()
            if self.plot_style.engine == "matplotlib"
            else self.plot_plotly()
        )

    def wrapup_subplot(self, title, axes):
        if len(self.plot_obj.target) > 1:
            axes.set_title(
                title,
                **self.plot_style.title["subplot_title"],
            )

    def _pdp_xy_grid(self, pdp_xy, norm, cmap, axes):
        im = axes.imshow(
            pdp_xy,
            cmap=cmap,
            norm=norm,
            origin="lower",
            aspect="auto",
            alpha=self.plot_style.interact["fill_alpha"],
        )
        R, C = pdp_xy.shape

        for r in range(R):
            for c in range(C):
                text_color = "black"
                if pdp_xy[r, c] >= norm.vmin + (norm.vmax - norm.vmin) * 0.5:
                    text_color = "w"
                axes.text(
                    c,
                    r,
                    round(pdp_xy[r, c], 3),
                    ha="center",
                    va="center",
                    color=text_color,
                    size=self.plot_style.interact["font_size"],
                )

        axes.set_xticks(np.arange(C - 1) + 0.5)
        axes.set_yticks(np.arange(R - 1) + 0.5)

        return im

    def _pdp_xy_grid_plotly(self, pdp_xy, plot_params, fig, grids):
        grids_x, grids_y = self.feature_grids
        fig.add_trace(
            go.Heatmap(
                x=np.arange(len(grids_x)),
                y=np.arange(len(grids_y)),
                text=np.array([["{:.3f}".format(v) for v in row] for row in pdp_xy]),
                texttemplate="%{text}",
                **plot_params,
            ),
            **grids,
        )

    def _pdp_contour_plot(self, X, Y, pdp_xy, norm, cmap, axes):
        level = np.min([X.shape[0], X.shape[1]])
        c1 = axes.contourf(
            X,
            Y,
            pdp_xy,
            N=level,
            origin="lower",
            cmap=cmap,
            norm=norm,
            alpha=self.plot_style.interact["fill_alpha"],
        )
        c2 = axes.contour(c1, levels=c1.levels, colors="white", origin="lower")
        axes.clabel(c2, fontsize=self.plot_style.interact["font_size"], inline=1)
        axes.set_aspect("auto")

        return c1

    def _pdp_contour_plot_plotly(self, plot_params, fig, grids):
        grids_x, grids_y = self.feature_grids
        plot_params.update(
            dict(
                line=dict(width=1, color="white"),
                contours=dict(showlabels=True),
            )
        )
        if self.plot_style.to_bins:
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
        fig.add_trace(trace, **grids)

    def _pdp_inter_plot(self, pdp_xy, norm, cmap, axes):
        grids_x, grids_y = self.feature_grids
        pdp_xy = pdp_xy.reshape((len(grids_x), len(grids_y))).T

        if self.plot_style.interact["type"] == "grid":
            im = self._pdp_xy_grid(pdp_xy, norm, cmap, axes)
        else:
            if self.plot_style.to_bins:
                X, Y = np.meshgrid(range(pdp_xy.shape[1]), range(pdp_xy.shape[0]))
            else:
                X, Y = np.meshgrid(grids_x, grids_y)
            im = self._pdp_contour_plot(X, Y, pdp_xy, norm, cmap, axes)
            axes.grid(False)

        plt.setp(axes.get_xticklabels(), visible=False)
        plt.setp(axes.get_yticklabels(), visible=False)
        axes.tick_params(which="minor", bottom=False, left=False)
        axes.tick_params(which="major", bottom=False, left=False)
        axes.set_frame_on(False)
        _axes_modify(axes, self.plot_style)

        return im

    def _pdp_inter_plot_plotly(
        self,
        pdp_xy,
        cmap,
        cb_xyz,
        v_range,
        fig,
        grids,
    ):
        grids_x, grids_y = self.feature_grids
        pdp_xy = pdp_xy.reshape((len(grids_x), len(grids_y))).T
        cb_x, cb_y, cb_z = cb_xyz

        plot_params = dict(
            z=pdp_xy,
            colorscale=cmap,
            opacity=self.plot_style.interact["fill_alpha"],
            showscale=True,
            colorbar=dict(len=cb_z, yanchor="bottom", y=cb_y, x=cb_x, xanchor="left"),
            zmin=v_range[0],
            zmax=v_range[1],
            name="pdp interact",
        )

        if self.plot_style.interact["type"] == "grid":
            self._pdp_xy_grid_plotly(pdp_xy, plot_params, fig, grids)
        else:
            self._pdp_contour_plot_plotly(plot_params, fig, grids)

    def _pdp_iso_plot(
        self,
        pdp,
        vmean,
        norm,
        cmap,
        axes,
        is_y=False,
    ):
        axes.imshow(
            np.expand_dims(pdp, int(is_y)),
            cmap=cmap,
            norm=norm,
            origin="lower",
            alpha=self.plot_style.isolate["fill_alpha"],
        )

        for i in range(len(pdp)):
            text_color = "black"
            if pdp[i] >= vmean:
                text_color = "w"

            text_params = {
                "s": round(pdp[i], 3),
                "ha": "center",
                "va": "center",
                "color": text_color,
                "size": self.plot_style.isolate["font_size"],
                "fontdict": {"family": self.plot_style.font_family},
            }
            if is_y:
                axes.text(x=0, y=i, rotation="vertical", **text_params)
            else:
                axes.text(x=i, y=0, **text_params)

        axes.set_frame_on(False)
        axes.axes.axis("tight")

    def _pdp_iso_plot_plotly(
        self,
        pdp,
        cmap,
        v_range,
        fig,
        grids,
        is_y=False,
    ):
        ticks = np.arange(len(pdp))
        text = np.array([["{:.3f}".format(v) for v in pdp]])
        pdp = pdp.reshape((1, -1))
        x, y = ticks, [0]

        if is_y:
            x, y = [0], ticks
            pdp = pdp.T
            text = text.T

        fig.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=pdp,
                text=text,
                texttemplate="%{text}",
                hovertemplate="%{text}",
                colorscale=cmap,
                opacity=self.plot_style.interact["fill_alpha"],
                zmin=v_range[0],
                zmax=v_range[1],
                showscale=False,
                name="pdp isolate",
            ),
            **grids,
        )

    def _get_pdp_xticks(self, is_xy=False, is_y=False):
        i = int(is_y)
        label = ""
        ticklabels = self.display_columns[i]
        per_ticklabels = self.percentile_columns[i]
        if self.feature_types[i] == "numeric":
            label = "value"
            ticklabels = self.grids[i]
            per_ticklabels = self.percentiles[i]
            if self.plot_style.show_percentile and self.plot_style.engine == "plotly":
                for j, p in enumerate(per_ticklabels):
                    ticklabels[j] += f"<br><sup><b>{p}</b></sup>"
                label = "value+percentile"
        label = _get_bold_text(self.feature_names[i], self.plot_style.engine) + (
            f" ({label})" if label else ""
        )

        return label, ticklabels, per_ticklabels

    def _set_pdp_ticks(self, axes, is_xy=False, is_y=False):
        if is_xy and self.plot_style.plot_pdp:
            return
        label, ticklabels, per_ticklabels = self._get_pdp_xticks(is_xy, is_y)
        if is_y:
            axes.set_ylabel(label, fontdict=self.plot_style.label["fontdict"])
        else:
            axes.set_xlabel(label, fontdict=self.plot_style.label["fontdict"])
            axes.xaxis.set_label_position("top")

        if (
            is_xy
            and self.plot_style.interact["type"] == "contour"
            and not self.plot_style.to_bins
        ):
            return

        if is_y:
            axes.set_yticks(range(len(ticklabels)), ticklabels)
            if self.plot_style.show_percentile and len(per_ticklabels) > 0:
                _display_percentile(axes, per_ticklabels, self.plot_style, is_y=True)
            if not is_xy:
                axes.get_xaxis().set_visible(False)
        else:
            axes.set_xticks(range(len(ticklabels)), ticklabels)
            if self.plot_style.show_percentile and len(per_ticklabels) > 0:
                _display_percentile(
                    axes, per_ticklabels, self.plot_style, is_y=False, top=False
                )
            if not is_xy:
                axes.get_yaxis().set_visible(False)

        _axes_modify(axes, self.plot_style, grid=False, top=True)
        axes.tick_params(axis="both", which="minor", length=0)
        axes.grid(False)

    def _set_pdp_ticks_plotly(self, fig, grids, is_xy=False, is_y=False):
        label, ticklabels, per_ticklabels = self._get_pdp_xticks(is_xy, is_y)
        if (
            is_xy
            and self.plot_style.interact["type"] == "contour"
            and not self.plot_style.to_bins
        ):
            tick_params = dict(
                title_text=label,
                title_standoff=0,
            )
        else:
            tick_params = dict(
                title_text=label,
                title_standoff=0,
                ticktext=ticklabels,
                tickvals=np.arange(len(ticklabels)),
            )
        tick_params.update(grids)

        if is_y:
            if not is_xy:
                fig.update_xaxes(showgrid=False, showticklabels=False, **grids)
            fig.update_yaxes(**tick_params)
        else:
            tick_params.update({"side": "top"})
            fig.update_xaxes(**tick_params)
            if not is_xy:
                fig.update_yaxes(showgrid=False, showticklabels=False, **grids)

    def _insert_colorbar(self, im, norm, axes):
        cax = inset_axes(
            axes,
            width="100%",
            height="100%",
            loc="right",
            bbox_to_anchor=(
                1.08
                if self.plot_style.show_percentile and not self.plot_style.plot_pdp
                else 1.02,
                0.0,
                0.03,
                1,
            ),
            bbox_transform=axes.transAxes,
            borderpad=0,
        )
        grids_x, grids_y = self.feature_grids
        cb_num_grids = np.max([np.min([len(grids_x), len(grids_y), 8]), 8])
        boundaries = [
            round(v, 3) for v in np.linspace(norm.vmin, norm.vmax, cb_num_grids)
        ]
        cb = plt.colorbar(im, cax=cax, boundaries=boundaries)
        cb.ax.tick_params(**self.plot_style.tick)
        cb.outline.set_visible(False)

    def plot_matplotlib(self):
        fig, inner_grid, title_axes = self.plot_style.make_subplots()
        axes = {"title_axes": title_axes, "interact_axes": [], "isolate_axes": []}

        for i, class_idx in enumerate(self.which_classes):
            pdp_x, pdp_y, pdp_xy, target, cmap = self.prepare_subplot(class_idx)

            if self.plot_style.plot_pdp:
                inner = GridSpecFromSubplotSpec(
                    2,
                    2,
                    subplot_spec=inner_grid[i],
                    wspace=self.plot_style.gaps["inner_x"],
                    hspace=self.plot_style.gaps["inner_y"],
                    height_ratios=self.plot_style.subplot_ratio["y"],
                    width_ratios=self.plot_style.subplot_ratio["x"],
                )
                xy_axes = plt.subplot(inner[3])
                if self.plot_style.interact["type"] == "grid":
                    x_axes = plt.subplot(inner[1], sharex=xy_axes)
                    y_axes = plt.subplot(inner[2], sharey=xy_axes)
                else:
                    x_axes = plt.subplot(inner[1])
                    y_axes = plt.subplot(inner[2])
                pdp_values = np.concatenate((pdp_x, pdp_y, pdp_xy))
            else:
                xy_axes = plt.subplot(inner_grid[i])
                x_axes, y_axes = None, None
                pdp_values = pdp_xy

            pdp_min, pdp_max = np.min(pdp_values), np.max(pdp_values)
            norm = mpl.colors.Normalize(vmin=pdp_min, vmax=pdp_max)
            vmean = norm.vmin + (norm.vmax - norm.vmin) * 0.5
            im = self._pdp_inter_plot(pdp_xy, norm, cmap, xy_axes)

            if self.plot_style.plot_pdp:
                plot_params = {
                    "vmean": vmean,
                    "norm": norm,
                    "cmap": cmap,
                }
                self._pdp_iso_plot(pdp_x, axes=x_axes, is_y=False, **plot_params)
                self._pdp_iso_plot(pdp_y, axes=y_axes, is_y=True, **plot_params)
                self._set_pdp_ticks(x_axes, is_xy=False, is_y=False)
                self._set_pdp_ticks(y_axes, is_xy=False, is_y=True)
                xy_axes.get_xaxis().set_visible(False)
                xy_axes.get_yaxis().set_visible(False)

                self.wrapup_subplot(f"class_{target}", x_axes)
            else:
                self._set_pdp_ticks(xy_axes, is_xy=True, is_y=False)
                self._set_pdp_ticks(xy_axes, is_xy=True, is_y=True)
                self.wrapup_subplot(f"class_{target}", xy_axes)

            self._insert_colorbar(im, norm, xy_axes)
            axes["interact_axes"].append(xy_axes)
            axes["isolate_axes"].append([x_axes, y_axes])

        return fig, axes

    def plot_plotly(self):
        nrows, ncols = self.plot_style.nrows, self.plot_style.ncols
        plot_args = {
            "rows": nrows * (2 if self.plot_style.plot_pdp else 1),
            "cols": ncols * (2 if self.plot_style.plot_pdp else 1),
            "horizontal_spacing": 0,
            "vertical_spacing": 0,
        }
        fig = self.plot_style.make_subplots_plotly(plot_args)
        subplot_titles = []

        for i, class_idx in enumerate(self.which_classes):
            pdp_x, pdp_y, pdp_xy, target, cmap = self.prepare_subplot(class_idx)
            grids = {"col": i % ncols + 1, "row": i // ncols + 1}
            nc, nr = grids["col"], grids["row"]

            if self.plot_style.plot_pdp:
                xy_grids = {"col": grids["col"] * 2, "row": grids["row"] * 2}
                x_grids = {"col": grids["col"] * 2, "row": grids["row"] * 2 - 1}
                y_grids = {"col": grids["col"] * 2 - 1, "row": grids["row"] * 2}
                pdp_values = np.concatenate((pdp_x, pdp_y, pdp_xy))
            else:
                xy_grids = copy.deepcopy(grids)
                x_grids, y_grids = None, None
                pdp_values = pdp_xy

            pdp_min, pdp_max = np.min(pdp_values), np.max(pdp_values)
            v_range = [pdp_min, pdp_max]

            title, cb_xyz = self.plot_style.update_plot_domains(
                fig, nc, nr, (x_grids, y_grids, xy_grids), f"class_{target}"
            )
            subplot_titles.append(title)
            self._pdp_inter_plot_plotly(pdp_xy, cmap, cb_xyz, v_range, fig, xy_grids)
            if self.plot_style.plot_pdp:
                self._pdp_iso_plot_plotly(
                    pdp_x, cmap, v_range, fig, x_grids, is_y=False
                )
                self._pdp_iso_plot_plotly(pdp_y, cmap, v_range, fig, y_grids, is_y=True)
                self._set_pdp_ticks_plotly(fig, x_grids, is_xy=False, is_y=False)
                self._set_pdp_ticks_plotly(fig, y_grids, is_xy=False, is_y=True)
                fig.update_xaxes(showgrid=False, showticklabels=False, **xy_grids)
                fig.update_yaxes(showgrid=False, showticklabels=False, **xy_grids)
            else:
                self._set_pdp_ticks_plotly(fig, xy_grids, is_xy=True, is_y=False)
                self._set_pdp_ticks_plotly(fig, xy_grids, is_xy=True, is_y=True)

        if len(self.plot_obj.target) > 1:
            fig.update_layout(annotations=subplot_titles)

        return fig, None
