class defaultColors:
    cmap = "tab20"
    darkGreen = "#1A4E5D"
    lightGreen = "#5BB573"
    darkGray = "#424242"
    lightGray = "#9E9E9E"
    black = "#000000"


default_font_family = "Arial"
default_figsize = (12, 8)


def update_style(curr, update):
    for key in update:
        if isinstance(curr[key], dict):
            if isinstance(update[key], dict):
                curr[key].update(update[key])
        else:
            curr[key] = update[key]


class infoPlotStyle:
    def __init__(self, feat_name, plot_params):
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

        self.font_family = plot_params.get("font_family", default_font_family)
        self.figsize = plot_params.get("figsize", default_figsize)
        self.nrows = plot_params.get("nrows", 1)
        self.ncols = plot_params.get("ncols", 1)
        self.display_columns = plot_params["display_columns"]
        self.percentile_columns = plot_params["percentile_columns"]

        self.line = {
            "color": defaultColors.darkGreen,
            "colors": None,
            "cmap": defaultColors.cmap,
            "width": 1,
            "fontdict": {"fontfamily": self.font_family, "fontsize": 9},
        }
        update_style(self.line, plot_params.get("line", {}))

        self.bar = {
            "width": None,
            "color": defaultColors.lightGreen,
            "fontdict": {"fontfamily": self.font_family, "fontsize": 9},
        }
        update_style(self.bar, plot_params.get("bar", {}))

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

        self.label = {"fontdict": {"fontfamily": self.font_family, "fontsize": 10}}
        update_style(self.label, plot_params.get("label", {}))

        self.title = {
            "title": {
                "font_size": 15,
                "color": defaultColors.black,
                "text": f'Target plot for feature "{feat_name}"',
            },
            "subtitle": {
                "font_size": 12,
                "color": defaultColors.lightGray,
                "text": "Average target value by different feature values.",
            },
        }
        update_style(self.title, plot_params.get("title", {}))
