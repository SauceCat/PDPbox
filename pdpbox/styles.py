import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


class defaultColors:
    cmap = "tab20"
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


class infoPlotStyle:
    def __init__(self, feat_name, plot_params, plot_type="target"):
        """
        Style configuration for info plots

        Parameters
        ----------
        feat_name: string
            name of the feature, not necessary a column name
        plot_params: dict
            detailed style configuration

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
                - cmap: used when colors is not provided
                - width
                - fontdict: https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text
            bar:
                - width
                - color
                - fontdict: https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text
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
        if plot_params is None:
            plot_params = {}

        self.plot_type = plot_type
        self.engine = plot_params.get("engine", "plotly")
        fontdict_keys = {
            "family": "family" if self.engine == "plotly" else "fontfamily",
            "size": "size" if self.engine == "plotly" else "fontsize",
        }
        self.template = plot_params.get("template", "plotly_white")
        self.font_family = plot_params.get("font_family", default_font_family)
        self.figsize = plot_params.get(
            "figsize",
            default_figsize_plotly if self.engine == "plotly" else default_figsize,
        )
        self.nrows = plot_params.get("nrows", 1)
        self.ncols = plot_params.get("ncols", 1)
        self.display_columns = plot_params["display_columns"]
        self.percentile_columns = plot_params["percentile_columns"]

        self.line = {
            "color": defaultColors.darkGreen,
            "colors": None,
            "cmap": defaultColors.cmap,
            "width": 1,
            "fontdict": {
                fontdict_keys["family"]: self.font_family,
                fontdict_keys["size"]: 9,
            },
        }
        update_style(self.line, plot_params.get("line", {}))
        if self.line["colors"] is None:
            if self.engine == "matplotlib":
                self.line["colors"] = plt.get_cmap(self.line["cmap"])(np.arange(20))
            else:
                self.line["colors"] = px.colors.qualitative.Plotly

        self.bar = {
            "width": None,
            "color": defaultColors.lightGreen,
            "fontdict": {
                fontdict_keys["family"]: self.font_family,
                fontdict_keys["size"]: 9,
            },
        }
        update_style(self.bar, plot_params.get("bar", {}))
        if self.bar["width"] is None:
            self.bar["width"] = np.min([0.4, 0.4 / (10.0 / len(self.display_columns))])

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
                fontdict_keys["family"]: self.font_family,
                fontdict_keys["size"]: 10,
            }
        }
        update_style(self.label, plot_params.get("label", {}))

        self.box = {
            "width": None,
            "line_width": 2,
            "color": defaultColors.blue,
            "colors": None,
            "cmap": defaultColors.cmap,
        }
        update_style(self.box, plot_params.get("box", {}))
        if self.box["colors"] is None:
            if self.engine == "matplotlib":
                self.box["colors"] = plt.get_cmap(self.box["cmap"])(np.arange(20))
            else:
                self.box["colors"] = px.colors.qualitative.Plotly
        if self.box["width"] is None:
            self.box["width"] = self.bar["width"]

        self.title = {
            "title": {
                "font_size": 15,
                "color": defaultColors.black,
                "text": (
                    "Target plot for feature "
                    if self.plot_type == "target"
                    else "Actual predictions plot for feature "
                )
                + (
                    f"<b>{feat_name}</b>" if self.engine == "plotly" else f"{feat_name}"
                ),
            },
            "subtitle": {
                "font_size": 12,
                "color": defaultColors.lightGray,
                "text": (
                    "Average target value by different feature values."
                    if self.plot_type == "target"
                    else "Distribution of actual prediction through different feature values."
                ),
            },
        }
        update_style(self.title, plot_params.get("title", {}))
