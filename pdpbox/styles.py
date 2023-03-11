import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


class defaultColors:
    cmap = "tab20"
    cmap_seq = "Blues"
    cmaps_seq = ["Blues", "Greens", "Oranges", "Reds", "Purples", "Greys"]
    colors = px.colors.qualitative.Plotly
    darkGreen = "#1A4E5D"
    lightGreen = "#5BB573"
    darkGray = "#424242"
    lightGray = "#9E9E9E"
    black = "#000000"
    blue = "#3288bd"


default_font_family = "Arial"
default_figsize = (16, 9)
default_figsize_plotly = (1200, 600)


def update_style(curr, update):
    for key in update:
        if isinstance(curr[key], dict):
            if isinstance(update[key], dict):
                curr[key].update(update[key])
        else:
            curr[key] = update[key]


def _get_colors(colors, cmap, engine, is_seq=False):
    if colors is None:
        if engine == "matplotlib":
            colors = plt.get_cmap(cmap)(np.arange(20))
        else:
            if is_seq:
                colors = defaultColors.colors_seq
            else:
                colors = defaultColors.colors
    return colors


class plotStyle:
    def __init__(self, target, plot_params):
        if plot_params is None:
            plot_params = {}

        self.dpi = plot_params["dpi"]
        self.engine = plot_params.get("engine", "plotly")
        self.fontdict_keys = {
            "family": "family" if self.engine == "plotly" else "fontfamily",
            "size": "size" if self.engine == "plotly" else "fontsize",
        }
        self.template = plot_params.get("template", "plotly_white")
        self.font_family = plot_params.get("font_family", default_font_family)
        figsize = plot_params.get(
            "figsize",
            default_figsize_plotly if self.engine == "plotly" else default_figsize,
        )
        self.display_columns = plot_params["display_columns"]
        self.percentile_columns = plot_params["percentile_columns"]
        self.horizontal_spacing = plot_params.get("horizontal_spacing", 0.1)
        self.vertical_spacing = plot_params.get("vertical_spacing", 0.05)

        nrows = plot_params.get("nrows", 1)
        ncols = plot_params.get("ncols", 1)
        width, height = figsize
        if len(target) > 1:
            nrows = int(np.ceil(len(target) * 1.0 / ncols))
            ncols = np.min([len(target), ncols])
            height = width * 1.0 / ncols * nrows * 0.8
        else:
            ncols = 1
        self.figsize = (width, height)
        self.nrows, self.ncols = int(nrows), int(ncols)

        self.tick = {
            "xticks_rotation": 0,
            "tick_params": {
                "axis": "both",
                "which": "major",
                "labelsize": 8,
                "labelcolor": defaultColors.darkGray,
                "colors": defaultColors.lightGray,
            },
        }
        update_style(self.tick, plot_params.get("tick", {}))

        self.label = {
            "fontdict": {
                self.fontdict_keys["family"]: self.font_family,
                self.fontdict_keys["size"]: 10,
            }
        }
        update_style(self.label, plot_params.get("label", {}))

        self.title = {
            "title": {
                "font_size": 15,
                "color": defaultColors.black,
                "text": "",
            },
            "subtitle": {
                "font_size": 12,
                "color": defaultColors.lightGray,
                "text": "",
            },
            "subplot_title": {
                "font_size": 11,
                "color": defaultColors.black,
            },
        }


class infoPlotStyle(plotStyle):
    def __init__(self, feat_name, target, plot_params, plot_type="target"):
        """
        Style configuration for info plots

        Parameters
        ----------
        feat_name: string
            name of the feature, not necessary a column name
        plot_params: dict
            detailed style configuration
        plot_type: "target" or "actual"

        Note
        ----
        plot_params configurable values:
            figure:
                - font_family
                - nrows
                - ncols
            line:
                - color
                - colors: for multiple lines
                - cmap: used when colors are not provided
                - width
                - fontdict: https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text
            bar:
                - width
                - color
                - fontdict: https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text
            box:
                - width
                - line_width
                - color
                - colors: for multiple boxes
                - cmap: used when colors are not provided
            tick:
                - xticks_rotation
                - tick_params: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html#matplotlib.axes.Axes.tick_params
            label:
                - fontdict: https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text
            title:
                title:
                    - font_size
                    - color
                    - text
                subtitle:
                    - font_size
                    - color
                    - text
        """

        super().__init__(target, plot_params)
        self.plot_type = plot_type

        if self.engine == "matplotlib":
            feat_name_bold = r"$\bf{" + feat_name.replace("_", "\\_") + "}$"
        else:
            feat_name_bold = f"<b>{feat_name}</b>"

        self.plot_type_to_title = {
            "title": {
                "target": "Target plot for feature " + feat_name_bold,
                "actual": "Actual prediction plot for feature " + feat_name_bold,
            },
            "subtitle": {
                "target": "Average target value by different feature values.",
                "actual": "Distribution of actual prediction by different feature values.",
            },
        }

        for key in ["title", "subtitle"]:
            self.title[key]["text"] = self.plot_type_to_title[key][self.plot_type]
        update_style(self.title, plot_params.get("title", {}))

        self.line = {
            "color": defaultColors.darkGreen,
            "colors": None,
            "cmap": defaultColors.cmap,
            "width": 1,
            "fontdict": {
                self.fontdict_keys["family"]: self.font_family,
                self.fontdict_keys["size"]: 9,
            },
        }
        update_style(self.line, plot_params.get("line", {}))
        self.line["colors"] = _get_colors(
            self.line["colors"], self.line["cmap"], self.engine, is_seq=False
        )

        self.bar = {
            "width": None,
            "color": defaultColors.lightGreen,
            "fontdict": {
                self.fontdict_keys["family"]: self.font_family,
                self.fontdict_keys["size"]: 9,
            },
        }
        update_style(self.bar, plot_params.get("bar", {}))
        if self.bar["width"] is None:
            self.bar["width"] = np.min([0.4, 0.4 / (10.0 / len(self.display_columns))])

        self.box = {
            "width": None,
            "line_width": 2,
            "color": defaultColors.blue,
            "colors": None,
            "cmap": defaultColors.cmap,
        }
        update_style(self.box, plot_params.get("box", {}))
        self.box["colors"] = _get_colors(
            self.box["colors"], self.box["cmap"], self.engine, is_seq=False
        )

        if self.box["width"] is None:
            self.box["width"] = self.bar["width"]

        if plot_type in ["target", "actual"]:
            self.bar["width"] *= self.ncols
            self.box["width"] *= self.ncols


class infoPlotInterStyle(plotStyle):
    def __init__(self, feat_names, target, plot_params, plot_type="target"):
        """
        Style configuration for info interaction plots

        Parameters
        ----------
        feat_names: list
            name of the features, not necessary column names
        plot_params: dict
            detailed style configuration
        plot_type: "target_interact" or "actual_interact"

        Note
        ----
        plot_params configurable values:
            figure:
                - font_family
                - nrows
                - ncols

            tick:
                - xticks_rotation
                - tick_params: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html#matplotlib.axes.Axes.tick_params
            label:
                - fontdict: https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text
            title:
                title:
                    - font_size
                    - color
                    - text
                subtitle:
                    - font_size
                    - color
                    - text
        """

        super().__init__(target, plot_params)
        self.plot_type = plot_type
        self.vertical_spacing = 0.1
        self.annotate = plot_params.get("annotate", False)

        if self.engine == "matplotlib":
            feat_name_bold = " & ".join(
                [r"$\bf{" + v + "}$" for v in feat_names]
            ).replace("_", "\\_")
        else:
            feat_name_bold = " & ".join([f"<b>{v}</b>" for v in feat_names])

        self.plot_type_to_title = {
            "title": {
                "target_interact": "Target plot for features " + feat_name_bold,
                "actual_interact": "Actual prediction plot for features "
                + feat_name_bold,
            },
            "subtitle": {
                "target_interact": "Average target value by different feature value combinations.",
                "actual_interact": "Medium value of actual prediction by different feature value combinations.",
            },
        }

        for key in ["title", "subtitle"]:
            self.title[key]["text"] = self.plot_type_to_title[key][self.plot_type]
        update_style(self.title, plot_params.get("title", {}))

        self.marker = {
            "line_width": 1,
            "cmap": defaultColors.cmap_seq,
            "cmaps": defaultColors.cmaps_seq,
            "min_size": 50 if self.engine == "matplotlib" else 10,
            "max_size": 1500 if self.engine == "matplotlib" else 50,
            "fontsize": 10,
        }
        update_style(self.marker, plot_params.get("marker", {}))

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
