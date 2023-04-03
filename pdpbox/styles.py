import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


class defaultColors:
    cmap_inter = "viridis"
    cmaps = ["Blues", "Greens", "Oranges", "Reds", "Purples", "Greys"]
    colors = ["tab20", px.colors.qualitative.Plotly]
    darkGreen = "#1A4E5D"
    lightGreen = "#5BB573"
    darkGray = "#424242"
    lightGray = "#9E9E9E"
    black = "#000000"
    blue = "#3288bd"
    darkBlue = "#1A4E5D"
    lightBlue = "#66C2D7"
    yellow = "#FEDC00"
    red = "#E75438"


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


def _prepare_plot_style(feat_name, target, plot_params, plot_type):
    plot_style_classes = {
        "target": infoPlotStyle,
        "actual": infoPlotStyle,
        "target_interact": infoPlotInterStyle,
        "actual_interact": infoPlotInterStyle,
        "pdp_isolate": PDPIsolatePlotStyle,
        "pdp_interact": PDPInteractPlotStyle,
    }

    args = (feat_name, target, plot_params, plot_type)
    plot_style = plot_style_classes[plot_type](*args)
    return plot_style


def _get_plot_sizes(subplot_ratio, gaps, ncols, nrows):
    def calc_dim(outer_gap, inner_gap, length, ratios, is_y=False):
        adjusted_gap = gaps["top"] if is_y else 0
        group_dim = (1.0 - outer_gap * (length - 1) - adjusted_gap) / length
        subplot_dim = group_dim + outer_gap if length > 1 else group_dim
        unit_dim = (
            (group_dim - inner_gap * (len(ratios) - 1)) / sum(ratios) if ratios else 0
        )
        return group_dim, subplot_dim, unit_dim

    dimensions = ["group_w", "group_h", "subplot_w", "subplot_h", "unit_w", "unit_h"]
    group_w, subplot_w, unit_w = calc_dim(
        gaps["outer_x"], gaps["inner_x"], ncols, subplot_ratio.get("x")
    )
    group_h, subplot_h, unit_h = calc_dim(
        gaps["outer_y"], gaps["inner_y"], nrows, subplot_ratio.get("y"), True
    )

    return dict(
        zip(dimensions, [group_w, group_h, subplot_w, subplot_h, unit_w, unit_h])
    )


def _get_subplot_title(x, y, title_text):
    return go.layout.Annotation(
        x=x,
        y=y,
        xref="paper",
        yref="paper",
        text=f"<b>{title_text}</b>",
        showarrow=False,
        xanchor="center",
        yanchor="middle",
    )


class plotStyle:
    def __init__(self, target, plot_params):
        if plot_params is None:
            plot_params = {}
        self.plot_params = plot_params

        self.set_default_attributes()
        self.set_figsize(target)
        self.set_tick_style()
        self.set_label_style()
        self.set_title_style()

    def set_default_attributes(self):
        attributes = [
            "dpi",
            "engine",
            "template",
            "display_columns",
            "percentile_columns",
            "show_percentile",
        ]
        for attr in attributes:
            setattr(self, attr, self.plot_params[attr])
        self.font_family = self.plot_params.get("font_family", default_font_family)

    def set_figsize(self, target):
        if self.plot_params["figsize"] is None:
            figsize = (
                default_figsize
                if self.engine == "matplotlib"
                else default_figsize_plotly
            )
        else:
            figsize = self.plot_params["figsize"]
        nrows, ncols = 1, self.plot_params.get("ncols", 1)
        width, height = figsize

        if len(target) > 1:
            ncols = np.min([len(target), ncols])
            nrows = int(np.ceil(len(target) * 1.0 / ncols))
            height = height * nrows
        else:
            ncols = 1

        self.figsize = (width, height)
        self.nrows, self.ncols = int(nrows), int(ncols)

    def set_tick_style(self):
        self.tick = {
            "xticks_rotation": 0,
            "tick_params": {
                "axis": "both",
                "which": "major",
                "labelsize": 9,
                "labelcolor": defaultColors.darkGray,
                "colors": defaultColors.lightGray,
            },
        }
        _update_style(self.tick, self.plot_params.get("tick", {}))

    def set_label_style(self):
        self.label = {
            "fontdict": {
                "fontfamily": self.font_family,
                "fontsize": 10,
            }
        }
        _update_style(self.label, self.plot_params.get("label", {}))

    def set_title_style(self):
        self.title = {
            "title": {
                "fontsize": 13,
                "color": defaultColors.black,
                "text": "",
            },
            "subtitle": {
                "fontsize": 11,
                "color": defaultColors.lightGray,
                "text": "",
            },
            "subplot_title": {
                "fontsize": 11,
                "color": defaultColors.black,
                "fontweight": "bold",
            },
        }


class infoPlotStyle(plotStyle):
    def __init__(self, feat_name, target, plot_params, plot_type="target"):
        super().__init__(target, plot_params)
        self.plot_type = plot_type

        self.set_plot_title(feat_name)
        self.set_line_style()
        self.set_bar_style()
        self.set_box_style()
        self.set_subplot_ratio()
        self.set_gaps()

        self.plot_sizes = _get_plot_sizes(
            self.subplot_ratio, self.gaps, self.ncols, self.nrows
        )
        self.set_plot_sizes()
        self.update_styles()

    def set_plot_title(self, feat_name):
        bold_text = _get_bold_text(feat_name, self.engine)
        self.plot_type_to_title = {
            "title": {
                "target": "Target plot for feature " + bold_text,
                "actual": "Actual prediction plot for feature " + bold_text,
            },
            "subtitle": {
                "target": "Average target value by different feature values.",
                "actual": "Distribution of actual prediction by different feature values.",
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
        _update_style(self.line, self.plot_params.get("line", {}))
        self.line["colors"] = _get_colors(self.line["colors"], self.engine)

    def set_bar_style(self):
        self.bar = {
            "width": None,
            "color": defaultColors.lightGreen,
            "fontdict": self.label["fontdict"],
        }
        _update_style(self.bar, self.plot_params.get("bar", {}))
        if self.bar["width"] is None:
            self.bar["width"] = np.min([0.4, 0.4 / (10.0 / len(self.display_columns))])

    def set_box_style(self):
        self.box = {
            "width": self.bar["width"],
            "line_width": 2,
            "colors": None,
            "fontdict": self.label["fontdict"],
        }
        _update_style(self.box, self.plot_params.get("box", {}))
        self.box["colors"] = _get_colors(
            self.box["colors"],
            self.engine,
        )

    def set_subplot_ratio(self):
        self.subplot_ratio = {
            "y": [1, 1],
        }

    def set_gaps(self):
        self.gaps = {
            "top": 0.02,
            "outer_x": 0.08 if self.plot_type == "actual" else 0.12,
            "outer_y": 0.15 if self.show_percentile else 0.05,
            "inner_x": 0.02,
            "inner_y": 0.02,
        }
        if self.engine == "matplotlib":
            self.gaps = {
                "top": 0.02,
                "outer_x": 0.1 if self.plot_type == "actual" else 0.3,
                "outer_y": 0.2,
                "inner_x": 0.02,
                "inner_y": 0.2,
            }

    def set_plot_sizes(self):
        group_w = self.plot_sizes["group_w"]
        group_h = self.plot_sizes["group_h"]
        unit_h = self.plot_sizes["unit_h"]
        y1, y2 = self.subplot_ratio["y"]

        self.plot_sizes.update(
            {
                "box_w": group_w,
                "bar_w": group_w,
                "box_h": unit_h * y1,
                "bar_h": unit_h * y2 if self.plot_type == "actual" else group_h,
            }
        )

    def update_styles(self):
        styles = [
            ("title", self.title),
            ("subplot_ratio", self.subplot_ratio),
            ("gaps", self.gaps),
        ]
        for style_name, style_dict in styles:
            _update_style(style_dict, self.plot_params.get(style_name, {}))

    def update_plot_domains(self, fig, nr, nc, grids, title_text):
        subplot_w, subplot_h = (
            self.plot_sizes["subplot_w"],
            self.plot_sizes["subplot_h"],
        )
        group_w, group_h = self.plot_sizes["group_w"], self.plot_sizes["group_h"]
        x0, y0 = subplot_w * (nc - 1), subplot_h * (nr - 1) + self.gaps["top"]

        common_domain_x = [x0, x0 + group_w]

        if self.plot_type == "target":
            box_grids, box_domain_y = None, None
            bar_grids, bar_domain_y = grids, [1.0 - (y0 + group_h), 1.0 - y0]
        else:
            box_grids, bar_grids = grids
            box_domain_y = [1.0 - (y0 + self.plot_sizes["box_h"]), 1.0 - y0]
            bar_y0 = y0 + self.plot_sizes["box_h"] + self.gaps["inner_y"]
            bar_domain_y = [1.0 - (bar_y0 + self.plot_sizes["bar_h"]), 1.0 - bar_y0]

        fig.update_xaxes(domain=common_domain_x, **bar_grids)
        fig.update_yaxes(domain=bar_domain_y, **bar_grids)

        if box_grids is not None:
            fig.update_xaxes(domain=common_domain_x, **box_grids)
            fig.update_yaxes(domain=box_domain_y, **box_grids)

        tx, ty = sum(common_domain_x) / 2, 1.0 - y0 + self.gaps["inner_y"]
        return _get_subplot_title(tx, ty, title_text)


class infoPlotInterStyle(plotStyle):
    def __init__(self, feat_names, target, plot_params, plot_type="target"):
        super().__init__(target, plot_params)
        self.plot_type = plot_type
        self.annotate = self.plot_params["annotate"]

        self.set_plot_title(feat_names)
        self.set_marker_style()
        self.set_legend_style()
        self.set_subplot_ratio()
        self.set_gaps()

        self.plot_sizes = _get_plot_sizes(
            self.subplot_ratio, self.gaps, self.ncols, self.nrows
        )
        self.set_plot_sizes()
        self.update_styles()

    def set_plot_title(self, feat_names):
        bold_text = _get_bold_text(feat_names, self.engine, is_inter=True)
        self.plot_type_to_title = {
            "title": {
                "target_interact": "Target plot for features " + bold_text,
                "actual_interact": "Actual prediction plot for features " + bold_text,
            },
            "subtitle": {
                "target_interact": "Average target value by different feature value combinations.",
                "actual_interact": "Medium value of actual prediction by different feature value combinations.",
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
                "height": "50%",
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
        self.subplot_ratio = {"y": [7, 1], "x": [1, 1]}

    def set_gaps(self):
        self.gaps = {
            "top": 0.02,
            "outer_x": 0.15,
            "outer_y": 0.15 if self.show_percentile else 0.05,
            "inner_x": 0.02,
            "inner_y": 0.1,
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

    def update_styles(self):
        styles = [
            ("title", self.title),
            ("marker", self.marker),
            ("legend", self.legend),
            ("subplot_ratio", self.subplot_ratio),
            ("gaps", self.gaps),
        ]
        for style_name, style_dict in styles:
            _update_style(style_dict, self.plot_params.get(style_name, {}))


class PDPIsolatePlotStyle(plotStyle):
    def __init__(self, feat_name, target, plot_params, plot_type):
        super().__init__(target, plot_params)
        self.plot_type = plot_type

        self.set_plot_attributes()
        self.set_plot_title(feat_name)
        self.set_line_style()
        self.set_dist_style()
        self.set_subplot_ratio()
        self.set_gaps()

        self.plot_sizes = _get_plot_sizes(
            self.subplot_ratio, self.gaps, self.ncols, self.nrows
        )
        self.set_plot_sizes()
        self.update_styles()

    def set_plot_attributes(self):
        attributes = [
            "plot_lines",
            "frac_to_plot",
            "center",
            "clustering",
            "plot_pts_dist",
            "x_quantile",
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
            "hl_color": defaultColors.yellow,
            "zero_color": defaultColors.red,
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

    def update_styles(self):
        styles = [
            ("title", self.title),
            ("line", self.line),
            ("dist", self.dist),
            ("subplot_ratio", self.subplot_ratio),
            ("gaps", self.gaps),
        ]
        for style_name, style_dict in styles:
            _update_style(style_dict, self.plot_params.get(style_name, {}))


class PDPInteractPlotStyle(plotStyle):
    def __init__(self, feat_names, target, plot_params, plot_type):
        super().__init__(target, plot_params)
        self.plot_type = plot_type

        self.set_plot_attributes()
        self.set_plot_title(feat_names)
        self.set_interact_style()
        self.set_isolate_style()
        self.set_subplot_ratio()
        self.set_gaps()

        self.plot_sizes = _get_plot_sizes(
            self.subplot_ratio, self.gaps, self.ncols, self.nrows
        )
        self.set_plot_sizes()
        self.update_styles()

    def set_plot_attributes(self):
        self.plot_pdp = self.plot_params["plot_pdp"]
        self.x_quantile = self.plot_params["x_quantile"]

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
            "cmap": defaultColors.cmap_inter,
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
        unit_w = self.plot_sizes["unit_w"]
        unit_h = self.plot_sizes["unit_h"]
        x1, x2 = self.subplot_ratio["x"]
        y1, y2 = self.subplot_ratio["y"]

        if self.plot_pdp:
            self.plot_sizes.update(
                {
                    "inter_w": unit_w * x2,
                    "inter_h": unit_h * y2,
                    "iso_y_w": unit_w * x1,
                    "iso_y_h": unit_h * y2,
                    "iso_x_w": unit_w * x2,
                    "iso_x_h": unit_h * y1,
                }
            )
        else:
            group_w = self.plot_sizes["group_w"]
            group_h = self.plot_sizes["group_h"]

            self.plot_sizes.update(
                {
                    "inter_w": group_w,
                    "inter_h": group_h,
                }
            )

    def update_styles(self):
        styles = [
            ("title", self.title),
            ("interact", self.interact),
            ("isolate", self.isolate),
            ("subplot_ratio", self.subplot_ratio),
            ("gaps", self.gaps),
        ]
        for style_name, style_dict in styles:
            _update_style(style_dict, self.plot_params.get(style_name, {}))
