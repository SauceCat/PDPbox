import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


class defaultColors:
    cmap_inter = "viridis"
    cmaps = ["Blues", "Greens", "Oranges", "Reds", "Purples", "Greys"]
    colors = ["tab20", px.colors.qualitative.Plotly]
    DARK_GREEN = "#1A4E5D"
    LIGHT_GREEN = "#5BB573"
    DARK_GRAY = "#424242"
    LIGHT_GRAY = "#9E9E9E"
    BLACK = "#000000"
    BLUE = "#3288bd"
    DARK_BLUE = "#1A4E5D"
    LIGHT_BLUE = "#66C2D7"
    YELLOW = "#FEDC00"
    RED = "#E75438"


default_font_family = "Arial"
default_figsize_plotly = (1200, 600)
default_figsize = (16, 9)


def _update_style(curr, update):
    for key in update:
        if isinstance(curr[key], dict):
            if isinstance(update[key], dict):
                curr[key].update(update[key])
        else:
            curr[key] = update[key]


def _get_colors(colors, engine):
    if colors is None:
        if engine == "matplotlib":
            colors = plt.get_cmap(defaultColors.colors[0])(np.arange(20))
        else:
            colors = defaultColors.colors[1]
    return colors


def _get_bold_text(text, engine, is_inter=False):
    if is_inter:
        if engine == "matplotlib":
            bold_text = " and ".join([r"$\bf{" + v + "}$" for v in text]).replace(
                "_", "\\_"
            )
        else:
            bold_text = " and ".join([f"<b>{v}</b>" for v in text])
    else:
        if engine == "matplotlib":
            bold_text = r"$\bf{" + text.replace("_", "\\_") + "}$"
        else:
            bold_text = f"<b>{text}</b>"

    return bold_text


def _get_axes_label(feat_name, feat_type, show_percentile, engine):
    label = "value" if feat_type == "numeric" else ""
    if show_percentile and engine == "plotly":
        label += "+percentile"

    label = _get_bold_text(feat_name, engine) + (f" ({label})" if label else "")
    return label


def _prepare_plot_style(feat_name, num_plots, plot_params, plot_type):
    plot_style_classes = {
        "target": InfoPlotStyle,
        "predict": InfoPlotStyle,
        "interact_target": InteractInfoPlotStyle,
        "interact_predict": InteractInfoPlotStyle,
        "pdp_isolate": PDPIsolatePlotStyle,
        "pdp_interact": PDPInteractPlotStyle,
    }

    args = (feat_name, num_plots, plot_params, plot_type)
    plot_style = plot_style_classes[plot_type](*args)
    return plot_style


class PlotStyle:
    def __init__(self, num_plots, plot_params):
        self.plot_params = plot_params or {}

        self.set_default_attributes()
        self.set_figsize(num_plots)
        self.set_tick_style()
        self.set_label_style()
        self.set_title_style()

    def set_default_attributes(self):
        attributes = [
            "dpi",
            "engine",
            "template",
            "show_percentile",
            "num_bins",
        ]
        for attr in attributes:
            setattr(self, attr, self.plot_params.get(attr))
        self.font_family = self.plot_params.get("font_family", default_font_family)

    def set_figsize(self, num_plots):
        figsize = self.plot_params.get("figsize") or (
            default_figsize if self.engine == "matplotlib" else default_figsize_plotly
        )
        width, height = figsize
        ncols = int(min(num_plots, self.plot_params.get("ncols", 1)))
        nrows = int(np.ceil(num_plots / ncols))
        self.figsize = (width, height * nrows)
        self.nrows, self.ncols = nrows, ncols

    def set_tick_style(self):
        self.tick = {
            "axis": "both",
            "which": "major",
            "labelsize": 9,
            "labelcolor": defaultColors.DARK_GRAY,
            "colors": defaultColors.LIGHT_GRAY,
            "labelrotation": 0,
        }

    def set_label_style(self):
        self.label = {
            "fontdict": {
                "fontfamily": self.font_family,
                "fontsize": 10,
            }
        }

    def set_title_style(self):
        self.title = {
            "title": {
                "fontsize": 13,
                "color": defaultColors.BLACK,
                "text": "",
            },
            "subtitle": {
                "fontsize": 11,
                "color": defaultColors.LIGHT_GRAY,
                "text": "",
            },
            "subplot_title": {
                "fontsize": 12,
                "color": defaultColors.BLACK,
                "fontweight": "bold",
            },
        }

    def make_subplots(self):
        def _plot_title(axes):
            title_params = {
                "x": 0,
                "va": "top",
                "ha": "left",
            }
            axes.set_facecolor("white")
            title_text = self.title["title"].pop("text")
            subtitle_text = self.title["subtitle"].pop("text")
            axes.text(y=0.7, s=title_text, **title_params, **self.title["title"])
            axes.text(y=0.5, s=subtitle_text, **title_params, **self.title["subtitle"])
            axes.axis("off")

        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        title_ratio = 2
        outer_grids = GridSpec(
            nrows=2,
            ncols=1,
            wspace=0.0,
            hspace=self.gaps["top"],
            height_ratios=[title_ratio, self.figsize[1] - title_ratio],
        )
        title_axes = plt.subplot(outer_grids[0])
        _plot_title(title_axes)

        inner_grids = GridSpecFromSubplotSpec(
            self.nrows,
            self.ncols,
            subplot_spec=outer_grids[1],
            wspace=self.gaps["outer_x"],
            hspace=self.gaps["outer_y"],
        )

        return fig, inner_grids, title_axes

    def make_subplots_plotly(self, plot_args):
        fig = make_subplots(**plot_args)
        fig.update_layout(
            width=self.figsize[0],
            height=self.figsize[1],
            template=self.template,
            showlegend=False,
            title=go.layout.Title(
                text=f"{self.title['title']['text']} <br><sup>{self.title['subtitle']['text']}</sup>",
                xref="paper",
                x=0,
            ),
        )

        return fig

    def get_plot_sizes(self):
        def calc_dim(outer_gap, inner_gap, length, ratios, is_y=False):
            adjusted_gap = self.gaps["top"] if is_y else 0
            group_dim = (1.0 - outer_gap * (length - 1) - adjusted_gap) / length
            subplot_dim = group_dim + outer_gap if length > 1 else group_dim
            unit_dim = (
                (group_dim - inner_gap * (len(ratios) - 1)) / sum(ratios)
                if ratios
                else 0
            )
            return group_dim, subplot_dim, unit_dim

        dimensions = [
            "group_w",
            "group_h",
            "subplot_w",
            "subplot_h",
            "unit_w",
            "unit_h",
        ]
        group_w, subplot_w, unit_w = calc_dim(
            self.gaps["outer_x"],
            self.gaps["inner_x"],
            self.ncols,
            self.subplot_ratio.get("x"),
        )
        group_h, subplot_h, unit_h = calc_dim(
            self.gaps["outer_y"],
            self.gaps["inner_y"],
            self.nrows,
            self.subplot_ratio.get("y"),
            True,
        )
        return dict(
            zip(dimensions, [group_w, group_h, subplot_w, subplot_h, unit_w, unit_h])
        )

    def get_subplot_title(self, x, y, title_text):
        return go.layout.Annotation(
            x=x,
            y=y,
            xref="paper",
            yref="paper",
            text=f"<b>{title_text}</b>",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=self.title["subplot_title"]["fontsize"]),
        )

    def _update_plot_domains(self, fig, nr, nc, grids, title_text, plot_domain_func):
        subplot_w, subplot_h = (
            self.plot_sizes["subplot_w"],
            self.plot_sizes["subplot_h"],
        )
        x0, y0 = subplot_w * (nc - 1), subplot_h * (nr - 1) + self.gaps["top"]

        tx, ty, domains = plot_domain_func(x0, y0)
        for (domain_x, domain_y), grid in zip(domains, grids):
            if grid is not None:
                fig.update_xaxes(domain=[np.clip(v, 0, 1) for v in domain_x], **grid)
                fig.update_yaxes(domain=[np.clip(v, 0, 1) for v in domain_y], **grid)

        title = self.get_subplot_title(tx, ty, title_text)
        return title, domains

    def update_styles(self):
        for style_name in self.plot_params:
            if hasattr(self, style_name):
                if isinstance(self.plot_params[style_name], dict):
                    _update_style(
                        getattr(self, style_name), self.plot_params[style_name]
                    )


class InfoPlotStyle(PlotStyle):
    """
    Manges the style of an information plot, including `TargetPlot` and `PredictPlot`.

    Attributes
    ----------
    plot_params : dict or None
        Custom plot parameters that control the style and aesthetics of the plot.
    dpi : int
        The resolution of the plot, measured in dots per inch. Only applicable when
        `engine` is 'matplotlib'.
    engine : {'matplotlib', 'plotly'}
        The plotting engine to use.
    template : str
        The template to use for plotly plots. Only applicable when `engine` is
        'plotly'. Reference: https://plotly.com/python/templates/
    show_percentile : bool
        If True, percentiles are shown in the plot.
    num_bins : int
        The number of feature bins. For binary and one-hot encoded features, it is
        the same as number of grids. For numeric features, it is the number of
        grids minus 1.
    font_family : str
        The font family to use for the plot.
    figsize : tuple or None
        The figure size for matplotlib or plotly figure. If None, the default
        figure size is used.
    nrows : int
        The number of rows of subplots in the figure.
    ncols : int
        The number of columns of subplots in the figure.
    tick : dict
        The tick style. Configurable through `plot_params['tick']`.
    label : dict
        The label style. Configurable through `plot_params['label']`.
    title : dict
        The title style. You can config three types of titles: 'title', 'subtitle',
        'subplot_title'. Configurable through `plot_params['title']`.
    plot_type : str
        The type of the plot to be generated.
    plot_type_to_title : dict
        Dictionary containing title and subtitle text for each plot type.
    line : dict
        The style config for the line plot. Configurable through
        `plot_params['line']`.
    bar : dict
        The style config for the bar plot. Configurable through
        `plot_params['bar']`.
    box : dict
        The style config for the box plot. Only applicable for `PredictPlot`.
        Configurable through `plot_params['box']`.
    subplot_ratio : dict
        The ratio for the subplots. Configurable through
        `plot_params['subplot_ratio']`.
    gaps : dict
        The gaps between subplots, as well as the inner gaps for each subplot.
        Configurable through `plot_params['gaps']`.
    plot_sizes : dict
        The calculated plot sizes.
    """

    def __init__(self, feat_name, num_plots, plot_params, plot_type="target"):
        super().__init__(num_plots, plot_params)
        self.plot_type = plot_type

        self.set_plot_title(feat_name)
        self.set_line_style()
        self.set_bar_style()
        self.set_box_style()
        self.set_subplot_ratio()
        self.set_gaps()
        self.update_styles()
        self.set_plot_sizes()

    def set_plot_title(self, feat_name):
        bold_text = _get_bold_text(feat_name, self.engine)
        self.plot_type_to_title = {
            "title": {
                "target": "Target plot for feature " + bold_text,
                "predict": "Prediction plot for feature " + bold_text,
            },
            "subtitle": {
                "target": "Average target value by different feature values.",
                "predict": "Distribution of predict prediction by different feature values.",
            },
        }

        for key in ["title", "subtitle"]:
            self.title[key]["text"] = self.plot_type_to_title[key][self.plot_type]

    def set_line_style(self):
        self.line = {
            "colors": None,
            "width": 1,
            "fontdict": self.label["fontdict"],
        }

    def set_bar_style(self):
        self.bar = {
            "width": None,
            "color": defaultColors.LIGHT_GREEN,
            "fontdict": self.label["fontdict"],
            "alpha": 0.5,
        }

    def set_box_style(self):
        self.box = {
            "width": self.bar["width"],
            "line_width": 1.5,
            "colors": None,
            "fontdict": self.label["fontdict"],
            "alpha": 0.8,
        }

    def set_subplot_ratio(self):
        self.subplot_ratio = {
            "y": [1, 1],
        }

    def set_gaps(self):
        self.gaps = {
            "top": 0.02,
            "outer_x": 0.08 if self.plot_type == "predict" else 0.15,
            "outer_y": 0.04,
            "inner_x": 0.02,
            "inner_y": 0.01,
        }
        if self.engine == "matplotlib":
            self.gaps = {
                "top": 0.02,
                "outer_x": 0.15 if self.plot_type == "predict" else 0.3,
                "outer_y": 0.15,
                "inner_x": 0.02,
                "inner_y": 0.2,
            }

    def set_plot_sizes(self):
        self.plot_sizes = self.get_plot_sizes()
        group_w = self.plot_sizes["group_w"]
        group_h = self.plot_sizes["group_h"]
        unit_h = self.plot_sizes["unit_h"]
        y1, y2 = self.subplot_ratio["y"]

        self.plot_sizes.update(
            {
                "box_w": group_w,
                "bar_w": group_w,
                "box_h": unit_h * y1,
                "bar_h": unit_h * y2 if self.plot_type == "predict" else group_h,
            }
        )

    def update_styles(self):
        super().update_styles()
        self.line["colors"] = _get_colors(self.line["colors"], self.engine)
        if self.bar["width"] is None:
            self.box["width"] = self.bar["width"] = np.min(
                [0.4, 0.4 / (10.0 / self.num_bins)]
            )
        self.box["colors"] = _get_colors(
            self.box["colors"],
            self.engine,
        )

    def update_plot_domains(self, fig, nr, nc, grids, title_text):
        title, _ = self._update_plot_domains(
            fig, nr, nc, grids, title_text, self._info_plot_domain
        )
        return title

    def _info_plot_domain(self, x0, y0):
        domain_x = [x0, x0 + self.plot_sizes["group_w"]]

        if self.plot_type == "target":
            box_domain_y = None
            bar_domain_y = [1.0 - (y0 + self.plot_sizes["group_h"]), 1.0 - y0]
        else:
            box_domain_y = [1.0 - (y0 + self.plot_sizes["box_h"]), 1.0 - y0]
            bar_y0 = y0 + self.plot_sizes["box_h"] + self.gaps["inner_y"]
            bar_domain_y = [1.0 - (bar_y0 + self.plot_sizes["bar_h"]), 1.0 - bar_y0]

        # girds: box_grids, bar_grids
        domains = [(domain_x, box_domain_y), (domain_x, bar_domain_y)]

        tx, ty = sum(domain_x) / 2, 1.0 - y0
        return tx, ty, domains


class InteractInfoPlotStyle(PlotStyle):
    """
    Manages the style of an interactive information plot, including `InteractTargetPlot` and `InteractPredictPlot`.

    Attributes
    ----------
    plot_params : dict or None
        Custom plot parameters that control the style and aesthetics of the plot.
    dpi : int
        The resolution of the plot, measured in dots per inch. Only applicable when
        `engine` is 'matplotlib'.
    engine : {'matplotlib', 'plotly'}
        The plotting engine to use.
    template : str
        The template to use for plotly plots. Only applicable when `engine` is
        'plotly'. Reference: https://plotly.com/python/templates/
    show_percentile : bool
        If True, percentiles are shown in the plot.
    num_bins : int
        The number of feature bins. For binary and one-hot encoded features, it is
        the same as number of grids. For numeric features, it is the number of
        grids minus 1.
    font_family : str
        The font family to use for the plot.
    figsize : tuple or None
        The figure size for matplotlib or plotly figure. If None, the default
        figure size is used.
    nrows : int
        The number of rows of subplots in the figure.
    ncols : int
        The number of columns of subplots in the figure.
    tick : dict
        The tick style. Configurable through `plot_params['tick']`.
    label : dict
        The label style. Configurable through `plot_params['label']`.
    title : dict
        The title style. You can config three types of titles: 'title', 'subtitle',
        'subplot_title'. Configurable through `plot_params['title']`.
    plot_type : str
        The type of the plot to be generated.
    annotate : bool
        If it is True, the circles on the plot will be annotated with detailed
        information.
    plot_type_to_title : dict
        Dictionary containing title and subtitle text for each plot type.
    marker : dict
        The style config for the scatter plot marker. Configurable through
        `plot_params['marker']`.
    legend : dict
        The style config for the legend, including color bar and marker.
        Configurable through `plot_params['legend']`.
    subplot_ratio : dict
        The ratio for the subplots. Configurable through
        `plot_params['subplot_ratio']`.
    gaps : dict
        The gaps between subplots, as well as the inner gaps for each subplot.
        Configurable through `plot_params['gaps']`.
    plot_sizes : dict
        The calculated plot sizes.
    """

    def __init__(self, feat_names, num_plots, plot_params, plot_type="interact_target"):
        super().__init__(num_plots, plot_params)
        self.plot_type = plot_type
        self.annotate = self.plot_params["annotate"]

        self.set_plot_title(feat_names)
        self.set_marker_style()
        self.set_legend_style()
        self.set_subplot_ratio()
        self.set_gaps()
        self.update_styles()
        self.set_plot_sizes()

    def set_plot_title(self, feat_names):
        bold_text = _get_bold_text(feat_names, self.engine, is_inter=True)
        self.plot_type_to_title = {
            "title": {
                "interact_target": "Target plot for features " + bold_text,
                "interact_predict": "Actual prediction plot for features " + bold_text,
            },
            "subtitle": {
                "interact_target": "Average target value by different feature value combinations.",
                "interact_predict": "Medium value of predict prediction by different feature value combinations.",
            },
        }
        for key in ["title", "subtitle"]:
            self.title[key]["text"] = self.plot_type_to_title[key][self.plot_type]

    def set_marker_style(self):
        self.marker = {
            "line_width": 1,
            "cmaps": defaultColors.cmaps,
            "min_size": 50 if self.engine == "matplotlib" else 10,
            "max_size": 1500 if self.engine == "matplotlib" else 50,
            "fontsize": 10,
        }

    def set_legend_style(self):
        self.legend = {
            "colorbar": {
                "height": "70%",
                "width": "100%",
                "fontsize": 10,
                "fontfamily": self.font_family,
            },
            "circles": {
                "fontsize": 10,
                "fontfamily": self.font_family,
            },
        }

    def set_subplot_ratio(self):
        self.subplot_ratio = {"y": [7, 0.5], "x": [1, 1]}

    def set_gaps(self):
        self.gaps = {
            "top": 0.02,
            "outer_x": 0.15,
            "outer_y": 0.05,
            "inner_x": 0.02,
            "inner_y": 0.12,
        }
        if self.engine == "matplotlib":
            self.gaps = {
                "top": 0.02,
                "outer_x": 0.3,
                "outer_y": 0.1,
                "inner_x": 0.02,
                "inner_y": 0.2,
            }

    def set_plot_sizes(self):
        self.plot_sizes = self.get_plot_sizes()
        group_w = self.plot_sizes["group_w"]
        unit_h = self.plot_sizes["unit_h"]
        unit_w = self.plot_sizes["unit_w"]
        y1, y2 = self.subplot_ratio["y"]
        x1, x2 = self.subplot_ratio["x"]
        div = 2 if self.ncols == 1 else 1

        self.plot_sizes.update(
            {
                "scatter_w": group_w,
                "scatter_h": unit_h * y1,
                "cb_h": unit_h * y2,
                "size_h": unit_h * y2,
                "cb_w": unit_w * x1 / div,
                "size_w": unit_w * x2 / div,
            }
        )

    def update_plot_domains(self, fig, nr, nc, grids, title_text):
        title, domains = self._update_plot_domains(
            fig, nr, nc, grids, title_text, self._interact_info_plot_domain
        )
        return domains[1], title

    def _interact_info_plot_domain(self, x0, y0):
        scatter_domain_x = [x0, x0 + self.plot_sizes["scatter_w"]]
        scatter_domain_y = [1.0 - (y0 + self.plot_sizes["scatter_h"]), 1.0 - y0]

        legend_y0 = y0 + self.plot_sizes["scatter_h"] + self.gaps["inner_y"]
        cb_x0, cb_y0 = x0, legend_y0
        cb_domain_x = [cb_x0, cb_x0 + self.plot_sizes["cb_w"]]
        cb_domain_y = [1.0 - (cb_y0 + self.plot_sizes["cb_h"]), 1.0 - cb_y0]

        size_x0, size_y0 = (
            cb_x0 + self.plot_sizes["cb_w"] + self.gaps["inner_x"],
            cb_y0,
        )
        size_domain_x = [size_x0, size_x0 + self.plot_sizes["size_w"]]
        size_domain_y = [1.0 - (size_y0 + self.plot_sizes["size_h"]), 1.0 - size_y0]

        tx, ty = sum(scatter_domain_x) / 2, 1.0 - y0
        # grids: scatter_grids, cb_grids, size_grids
        domains = [
            (scatter_domain_x, scatter_domain_y),
            (cb_domain_x, cb_domain_y),
            (size_domain_x, size_domain_y),
        ]
        return tx, ty, domains


class PDPIsolatePlotStyle(PlotStyle):
    """
    Manages the style of the `PDPIsolate` plot.

    Attributes
    ----------
    plot_params : dict or None
        Custom plot parameters that control the style and aesthetics of the plot.
    dpi : int
        The resolution of the plot, measured in dots per inch. Only applicable when
        `engine` is 'matplotlib'.
    engine : {'matplotlib', 'plotly'}
        The plotting engine to use.
    template : str
        The template to use for plotly plots. Only applicable when `engine` is
        'plotly'. Reference: https://plotly.com/python/templates/
    show_percentile : bool
        If True, percentiles are shown in the plot.
    num_bins : int
        The number of feature bins. For binary and one-hot encoded features, it is
        the same as number of grids. For numeric features, it is the number of
        grids minus 1.
    font_family : str
        The font family to use for the plot.
    figsize : tuple or None
        The figure size for matplotlib or plotly figure. If None, the default
        figure size is used.
    nrows : int
        The number of rows of subplots in the figure.
    ncols : int
        The number of columns of subplots in the figure.
    tick : dict
        The tick style. Configurable through `plot_params['tick']`.
    label : dict
        The label style. Configurable through `plot_params['label']`.
    title : dict
        The title style. You can config three types of titles: 'title', 'subtitle',
        'subplot_title'. Configurable through `plot_params['title']`.
    plot_type : str
        The type of the plot to be generated.
    plot_lines : bool
        If True, ICE lines will be plotted.
    frac_to_plot : int or float
        Fraction of ICE lines to plot.
    center : bool
        If True, the PDP will be centered by deducting the values of `grids[0]`.
    clustering : dict
        The clustering config.
    plot_pts_dist : bool
        If True, distribution of points will be plotted.
    to_bins : bool
        If True, the axis will be converted to bins. Only applicable for numeric
        feature.
    std_fill : bool
        If it is True, std of pdp will be colored. Configurable through
        `plot_params['std_fill']`.
    pdp_hl : bool
        If it is True, pdp will be highlighted. Configurable through
        `plot_params['pdp_hl']`.
    plot_type_to_title : dict
        Dictionary containing title and subtitle text for each plot type.
    line : dict
        The style config for the line plot. Configurable through
        `plot_params['line']`.
    dist : dict
        The style config for the distribution plot. Configurable through
        `plot_params['dist']`.
    subplot_ratio : dict
        The ratio for the subplots. Configurable through
        `plot_params['subplot_ratio']`.
    gaps : dict
        The gaps between subplots, as well as the inner gaps for each subplot.
        Configurable through `plot_params['gaps']`.
    plot_sizes : dict
        The calculated plot sizes.
    """

    def __init__(self, feat_name, num_plots, plot_params, plot_type):
        super().__init__(num_plots, plot_params)
        self.plot_type = plot_type

        self.set_plot_attributes()
        self.set_plot_title(feat_name)
        self.set_line_style()
        self.set_dist_style()
        self.set_subplot_ratio()
        self.set_gaps()
        self.update_styles()
        self.set_plot_sizes()

    def set_plot_attributes(self):
        attributes = [
            "plot_lines",
            "frac_to_plot",
            "center",
            "clustering",
            "plot_pts_dist",
            "to_bins",
        ]
        for attr in attributes:
            setattr(self, attr, self.plot_params[attr])

        self.std_fill = self.plot_params.get("std_fill", True)
        if self.plot_lines:
            self.std_fill = False
        self.pdp_hl = self.plot_params.get("pdp_hl", False)

    def set_plot_title(self, feat_name):
        bold_text = _get_bold_text(feat_name, self.engine)
        n_grids = self.plot_params["n_grids"]
        self.plot_type_to_title = {
            "title": {
                "pdp_isolate": "PDP for feature " + bold_text,
            },
            "subtitle": {
                "pdp_isolate": f"Number of unique grid points: {n_grids}",
            },
        }

        for key in ["title", "subtitle"]:
            self.title[key]["text"] = self.plot_type_to_title[key][self.plot_type]

    def set_line_style(self):
        self.line = {
            "hl_color": defaultColors.YELLOW,
            "zero_color": defaultColors.RED,
            "cmaps": defaultColors.cmaps,
            "width": 1 if self.engine == "matplotlib" else 1.5,
            "fontdict": self.label["fontdict"],
            "fill_alpha": 0.8,
            "hl_alpha": 0.8,
            "markersize": 3 if self.engine == "matplotlib" else 5,
        }

    def set_dist_style(self):
        self.dist = {
            "markersize": 20,
            "fill_alpha": 0.8,
            "font_size": 10,
        }

    def set_subplot_ratio(self):
        self.subplot_ratio = {
            "y": [7, 0.5],
        }

    def set_gaps(self):
        self.gaps = {
            "top": 0.01,
            "outer_x": 0.04,
            "outer_y": 0.08,
            "inner_x": 0.02,
            "inner_y": 0.05,
        }
        if self.engine == "matplotlib":
            self.gaps = {
                "top": 0.02,
                "outer_x": 0.2,
                "outer_y": 0.2,
                "inner_x": 0.02,
                "inner_y": 0.3,
            }

    def set_plot_sizes(self):
        self.plot_sizes = self.get_plot_sizes()
        group_w = self.plot_sizes["group_w"]
        group_h = self.plot_sizes["group_h"]
        unit_h = self.plot_sizes["unit_h"]

        self.plot_sizes.update(
            {
                "line_w": group_w,
                "line_h": group_h,
            }
        )

        if self.plot_pts_dist:
            y1, y2 = self.subplot_ratio["y"]

            self.plot_sizes.update(
                {
                    "line_h": unit_h * y1,
                    "dist_h": unit_h * y2,
                    "dist_w": group_w,
                }
            )

    def update_plot_domains(self, fig, nc, nr, grids, title_text):
        title, _ = self._update_plot_domains(
            fig, nr, nc, grids, title_text, self._pdp_isolate_plot_domain
        )
        return title

    def _pdp_isolate_plot_domain(self, x0, y0):
        domain_x = [x0, x0 + self.plot_sizes["line_w"]]
        line_domain_y = [1.0 - (y0 + self.plot_sizes["line_h"]), 1.0 - y0]

        if self.plot_pts_dist:
            dist_y0 = y0 + self.plot_sizes["line_h"] + self.gaps["inner_y"]
            dist_domain_y = [1.0 - (dist_y0 + self.plot_sizes["dist_h"]), 1.0 - dist_y0]
        else:
            dist_domain_y = None

        tx, ty = sum(domain_x) / 2, 1.0 - y0
        # grids: line_grids, dist_grids
        domains = [(domain_x, line_domain_y), (domain_x, dist_domain_y)]
        return tx, ty, domains


class PDPInteractPlotStyle(PlotStyle):
    """
    Manages the style of the `PDPInteract` plot.

    Attributes
    ----------
    plot_params : dict or None
        Custom plot parameters that control the style and aesthetics of the plot.
    dpi : int
        The resolution of the plot, measured in dots per inch. Only applicable when
        `engine` is 'matplotlib'.
    engine : {'matplotlib', 'plotly'}
        The plotting engine to use.
    template : str
        The template to use for plotly plots. Only applicable when `engine` is
        'plotly'. Reference: https://plotly.com/python/templates/
    show_percentile : bool
        If True, percentiles are shown in the plot.
    num_bins : int
        The number of feature bins. For binary and one-hot encoded features, it is
        the same as number of grids. For numeric features, it is the number of
        grids minus 1.
    font_family : str
        The font family to use for the plot.
    figsize : tuple or None
        The figure size for matplotlib or plotly figure. If None, the default
        figure size is used.
    nrows : int
        The number of rows of subplots in the figure.
    ncols : int
        The number of columns of subplots in the figure.
    tick : dict
        The tick style. Configurable through `plot_params['tick']`.
    label : dict
        The label style. Configurable through `plot_params['label']`.
    title : dict
        The title style. You can config three types of titles: 'title', 'subtitle',
        'subplot_title'. Configurable through `plot_params['title']`.
    plot_type : str
        The type of the plot to be generated.
    plot_pdp : bool
        If it is True, pdp for each feature will be plotted.
    to_bins : bool
        If True, the axis will be converted to bins. Only applicable for numeric
        feature.
    plot_type_to_title : dict
        Dictionary containing title and subtitle text for each plot type.
    interact : dict
        The style config for the interaction pdp. Configurable through
        `plot_params['interact']`.
    isolate : dict
        The style config for the isolation pdp. Configurable through
        `plot_params['isolate']`.
    subplot_ratio : dict
        The ratio for the subplots. Configurable through
        `plot_params['subplot_ratio']`.
    gaps : dict
        The gaps between subplots, as well as the inner gaps for each subplot.
        Configurable through `plot_params['gaps']`.
    plot_sizes : dict
        The calculated plot sizes.
    """

    def __init__(self, feat_names, num_plots, plot_params, plot_type):
        super().__init__(num_plots, plot_params)
        self.plot_type = plot_type

        self.set_plot_attributes()
        self.set_plot_title(feat_names)
        self.set_interact_style()
        self.set_isolate_style()
        self.set_subplot_ratio()
        self.set_gaps()
        self.update_styles()
        self.set_plot_sizes()

    def set_plot_attributes(self):
        attributes = [
            "plot_pdp",
            "to_bins",
        ]
        for attr in attributes:
            setattr(self, attr, self.plot_params[attr])

    def set_plot_title(self, feat_names):
        feat1, feat2 = [_get_bold_text(v, self.engine) for v in feat_names]
        grids1, grids2 = self.plot_params["n_grids"]
        self.plot_type_to_title = {
            "title": {
                "pdp_interact": f"PDP interact for features {feat1} and {feat2}",
            },
            "subtitle": {
                "pdp_interact": f"Number of unique grid points: ({feat1}: {grids1}, {feat2}: {grids2})",
            },
        }

        for key in ["title", "subtitle"]:
            self.title[key]["text"] = self.plot_type_to_title[key][self.plot_type]

    def set_interact_style(self):
        self.interact = {
            "cmaps": defaultColors.cmaps,
            "fill_alpha": 0.8,
            "type": self.plot_params["plot_type"],
            "font_size": 10,
        }

    def set_isolate_style(self):
        self.isolate = {
            "fill_alpha": 0.8,
            "font_size": 10,
        }

    def set_subplot_ratio(self):
        self.subplot_ratio = {
            "x": [0.5, 7],
            "y": [0.5, 7],
        }

    def set_gaps(self):
        self.gaps = {
            "top": 0.05,
            "outer_x": 0.15,
            "outer_y": 0.08,
            "inner_x": 0.02,
            "inner_y": 0.05 if self.show_percentile else 0.02,
        }
        if self.engine == "matplotlib":
            self.gaps = {
                "top": 0.05,
                "outer_x": 0.3,
                "outer_y": 0.15,
                "inner_x": 0.15 if self.show_percentile else 0.04,
                "inner_y": 0.2 if self.show_percentile else 0.04,
            }

    def set_plot_sizes(self):
        self.plot_sizes = self.get_plot_sizes()
        unit_w = self.plot_sizes["unit_w"]
        unit_h = self.plot_sizes["unit_h"]
        x1, x2 = self.subplot_ratio["x"]
        y1, y2 = self.subplot_ratio["y"]

        if self.plot_pdp:
            self.plot_sizes.update(
                {
                    "xy_w": unit_w * x2,
                    "xy_h": unit_h * y2,
                    "y_w": unit_w * x1,
                    "y_h": unit_h * y2,
                    "x_w": unit_w * x2,
                    "x_h": unit_h * y1,
                }
            )
        else:
            group_w = self.plot_sizes["group_w"]
            group_h = self.plot_sizes["group_h"]

            self.plot_sizes.update(
                {
                    "xy_w": group_w,
                    "xy_h": group_h,
                }
            )

    def update_plot_domains(self, fig, nc, nr, grids, title_text):
        title, domains = self._update_plot_domains(
            fig, nr, nc, grids, title_text, self._pdp_interact_plot_domain
        )

        xy_domain_x, xy_domain_y = domains[-1]
        cb_xyz = (
            xy_domain_x[1] + self.gaps["inner_x"] / 2,
            xy_domain_y[0],
            self.plot_sizes["xy_h"],
        )
        return title, cb_xyz

    def _pdp_interact_plot_domain(self, x0, y0):
        if self.plot_pdp:
            xy_x0 = x0 + self.plot_sizes["y_w"] + self.gaps["inner_x"]
            xy_y0 = y0 + self.plot_sizes["x_h"] + self.gaps["inner_y"]
        else:
            xy_x0, xy_y0 = x0, y0
        xy_domain_x = [xy_x0, xy_x0 + self.plot_sizes["xy_w"]]
        xy_domain_y = [1.0 - (xy_y0 + self.plot_sizes["xy_h"]), 1.0 - xy_y0]

        if self.plot_pdp:
            x_y0, y_x0 = y0, x0
            x_domain_y = [1.0 - (x_y0 + self.plot_sizes["x_h"]), 1.0 - x_y0]
            y_domain_x = [y_x0, y_x0 + self.plot_sizes["y_w"]]
        else:
            x_domain_y, y_domain_x = None, None

        tx, ty = sum(xy_domain_x) / 2, xy_domain_y[0] - self.gaps["inner_y"]
        # grids: x_grids, y_grids, xy_grids
        domains = [
            (xy_domain_x, x_domain_y),
            (y_domain_x, xy_domain_y),
            (xy_domain_x, xy_domain_y),
        ]
        return tx, ty, domains


def _axes_modify(axes, plot_style, top=False, right=False, grid=True):
    """
    Modify axes settings.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The axes object to modify.
    plot_style : :class:`InfoPlotStyle`
        An instance of the `InfoPlotStyle` class.
    top : bool, optional
        If True, set the target x-axis ticks position at the top of the plot.
        Default is False.
    right : bool, optional
        If True, set the target y-axis ticks position at the right side of the
        plot. Default is False.
    grid : bool, optional
        If True, display the grid on the plot. Default is True.
    """
    axes.set_facecolor("white")
    axes.tick_params(**plot_style.tick)
    axes.set_frame_on(False)
    axes.xaxis.set_ticks_position("top" if top else "bottom")
    axes.yaxis.set_ticks_position("right" if right else "left")
    axes.xaxis.set_label_position("top" if top else "bottom")
    axes.yaxis.set_label_position("right" if right else "left")

    if grid:
        axes.grid(True, "major", "both", ls="--", lw=0.5, c="k", alpha=0.3)


def _modify_legend_axes(axes, font_family):
    """
    Modify the legend axes settings.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The axes object to modify.
    font_family : str
        The font family to use for the plot.
    """
    axes.set_frame_on(False)

    for tick in axes.get_xticklabels():
        tick.set_fontname(font_family)
    for tick in axes.get_yticklabels():
        tick.set_fontname(font_family)

    axes.set_facecolor("white")
    axes.set_xticks([])
    axes.set_yticks([])


def _display_percentile(
    axes, label, percentile_columns, plot_style, is_y=False, right=True, top=True
):
    """
    Display percentile values on the axes.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The axes object to modify.
    label : str
        The axis label.
    percentile_columns : list of str
        percentile_columns from :class:`FeatureInfo`
    plot_style : :class:`InfoPlotStyle`
        An instance of the `InfoPlotStyle` class.
    is_y : bool, optional
        If True, the percentile values will be displayed on the y-axis, otherwise
        they will be displayed on the x-axis. Default is False.
    right : bool, optional
        If True, set the target y-axis ticks position at the right side of the
        plot. Default is True.
    top : bool, optional
        If True, set the target x-axis ticks position at the top of the plot.
        Default is True.
    """
    label = _get_bold_text(label, plot_style.engine)
    if is_y:
        per_axes = axes.twinx()
        ticks = axes.get_yticks()

        if len(ticks) > len(percentile_columns):
            per_axes.set_yticks(ticks[:-1] + 0.5, labels=percentile_columns)
        else:
            per_axes.set_yticks(axes.get_yticks(), labels=percentile_columns)
        per_axes.set_ybound(axes.get_ybound())
        per_axes.set_ylabel(
            f"{label} (percentile)", fontdict=plot_style.label["fontdict"]
        )
        _axes_modify(per_axes, plot_style, right=right, grid=False)
    else:
        per_axes = axes.twiny()
        ticks = axes.get_xticks()

        if len(ticks) > len(percentile_columns):
            per_axes.set_xticks(ticks[:-1] + 0.5, labels=percentile_columns)
        else:
            per_axes.set_xticks(axes.get_xticks(), labels=percentile_columns)
        per_axes.set_xbound(axes.get_xbound())
        per_axes.set_xlabel(
            f"{label} (percentile)", fontdict=plot_style.label["fontdict"]
        )
        _axes_modify(per_axes, plot_style, top=top, grid=False)
