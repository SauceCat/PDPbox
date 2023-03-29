from .pdp_calc_utils import (
    _calc_ice_lines,
    _calc_ice_lines_inter,
    _prepare_pdp_count_data,
)
from .pdp_plot_utils import (
    _pdp_plot,
    _pdp_plot_plotly,
    _pdp_inter_plot,
    _pdp_inter_plot_plotly,
)
from .utils import (
    _check_model,
    _check_dataset,
    _check_percentile_range,
    _check_feature,
    _check_grid_type,
    _check_memory_limit,
    _check_frac_to_plot,
    _make_list,
    _expand_default,
    _plot_title,
    _calc_memory_usage,
    _get_grids,
    _get_grid_combos,
    _check_classes,
    _calc_figsize,
    _get_string,
    _check_plot_params,
    _get_grids_and_cols,
    _calc_n_jobs,
    _prepare_plot_params,
)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pqdm.processes import pqdm
import copy

import warnings

warnings.filterwarnings("ignore")


def _collect_pdp_results(pdp_obj, calc_args, calc_func):
    grid_results = pqdm(
        calc_args,
        calc_func,
        n_jobs=pdp_obj.n_jobs,
        argument_type="kwargs",
        desc="calculate per feature grid",
    )
    grid_indices = np.arange(pdp_obj.n_grids)
    pdp_obj.results = []
    if pdp_obj.n_classes > 2:
        for cls_idx in range(pdp_obj.n_classes):
            ice_lines = pd.concat([res[cls_idx] for res in grid_results], axis=1)
            pdp = ice_lines[grid_indices].mean().values
            pdp_obj.results.append(PDPResult(cls_idx, ice_lines, pdp))
    else:
        ice_lines = pd.concat(grid_results, axis=1)
        pdp = ice_lines[grid_indices].mean().values
        pdp_obj.results.append(PDPResult(None, ice_lines, pdp))


def _get_which_classes(n_classes, which_classes):
    if which_classes is not None:
        _check_classes(which_classes, n_classes)
    else:
        if n_classes <= 2:
            which_classes = [0]
        else:
            which_classes = np.arange(n_classes)
    return which_classes


class PDPResult:
    def __init__(
        self,
        class_id,
        ice_lines,
        pdp,
    ):
        self.class_id = class_id
        self.ice_lines = ice_lines
        self.pdp = pdp


class PDPIsolate:
    def __init__(
        self,
        model,
        dataset,
        model_features,
        feature,
        pred_func=None,
        n_classes=None,
        num_grid_points=10,
        grid_type="percentile",
        percentile_range=None,
        grid_range=None,
        cust_grid_points=None,
        memory_limit=0.5,
        chunk_size=-1,
        n_jobs=1,
        predict_kwds={},
        data_transformer=None,
    ):

        self.model = model
        self.dataset = dataset
        self.model_features = model_features
        self.feature = feature
        self.feature_type = _check_plot_params(
            dataset, feature, grid_type, percentile_range
        )
        self.n_classes, self.pred_func, self.from_model = _check_model(
            model, n_classes, pred_func
        )
        self.num_grid_points = num_grid_points
        self.grid_type = grid_type
        self.percentile_range = percentile_range
        self.grid_range = grid_range
        self.cust_grid_points = cust_grid_points
        self.memory_limit = memory_limit
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs
        self.predict_kwds = predict_kwds
        self.data_transformer = data_transformer

        self.prepare()
        self.calculate()

    def prepare(self):
        prepared_results = _prepare_pdp_count_data(
            self.feature,
            self.feature_type,
            self.dataset[_make_list(self.feature)],
            self.num_grid_points,
            self.grid_type,
            self.percentile_range,
            self.grid_range,
            self.cust_grid_points,
        )
        self.feature_grids = prepared_results["feature_grids"]
        self.n_grids = len(self.feature_grids)
        self.percentile_grids = prepared_results["percentile_grids"]
        self.count_data = prepared_results["count"]
        self.dist_data = prepared_results["dist"]
        self.display_columns = prepared_results["value_display"][0]
        self.percentile_columns = prepared_results["percentile_display"][0]
        self.n_jobs = _calc_n_jobs(
            self.dataset, len(self.feature_grids), self.memory_limit, self.n_jobs
        )

    def calculate(self):
        args = {
            "model": self.model,
            "data": self.dataset[self.model_features],
            "feat": self.feature,
            "feat_type": self.feature_type,
            "n_classes": self.n_classes,
            "pred_func": self.pred_func,
            "from_model": self.from_model,
            "predict_kwds": self.predict_kwds,
            "data_trans": self.data_transformer,
            "chunk_size": self.chunk_size,
        }
        calc_args = []
        for i, grid in enumerate(self.feature_grids):
            args_ = copy.deepcopy(args)
            args_.update({"feat_grid": grid, "grid_idx": i})
            calc_args.append(args_)

        _collect_pdp_results(self, calc_args, _calc_ice_lines)

    def pdp_plot(
        self,
        feature_name,
        center=True,
        plot_lines=False,
        frac_to_plot=1,
        cluster=False,
        n_cluster_centers=None,
        cluster_method="accurate",
        plot_pts_dist=False,
        x_quantile=False,
        show_percentile=False,
        which_classes=None,
        figsize=None,
        dpi=300,
        ncols=2,
        plot_params=None,
        engine="plotly",
        template="plotly_white",
    ):
        """Plot partial dependent plot

        Parameters
        ----------

        feature_name: string
            name of the feature, not necessary a column name
        center: bool, default=True
            whether to center the plot
        plot_pts_dist: bool, default=False
            whether to show data points distribution
        plot_lines: bool, default=False
            whether to plot out the individual lines
        frac_to_plot: float or integer, default=1
            how many lines to plot, can be a integer or a float
        cluster: bool, default=False
            whether to cluster the individual lines and only plot out the cluster centers
        n_cluster_centers: integer, default=None
            number of cluster centers
        cluster_method: string, default='accurate'
            cluster method to use, default is KMeans, if 'approx' is passed, MiniBatchKMeans is used
        plot_pts_dist: bool, default=False
            whether to show data points distribution
        x_quantile: bool, default=False
            whether to construct x axis ticks using quantiles
        show_percentile: bool, optional, default=False
            whether to display the percentile buckets,
            for numeric feature when grid_type='percentile'
        figsize: tuple or None, optional, default=None
            size of the figure, (width, height)
        ncols: integer, optional, default=2
            number subplot columns, used when it is multi-class problem
        plot_params:  dict or None, optional, default=None
            parameters for the plot, possible parameters as well as default as below:

            .. highlight:: python
            .. code-block:: python

                plot_params = {
                    # plot title and subtitle
                    'title': 'PDP for feature "%s"' % feature_name,
                    'subtitle': "Number of unique grid points: %d" % n_grids,
                    'title_fontsize': 15,
                    'subtitle_fontsize': 12,
                    'font_family': 'Arial',
                    # matplotlib color map for ICE lines
                    'line_cmap': 'Blues',
                    'xticks_rotation': 0,
                    # pdp line color, highlight color and line width
                    'pdp_color': '#1A4E5D',
                    'pdp_hl_color': '#FEDC00',
                    'pdp_linewidth': 1.5,
                    # horizon zero line color and with
                    'zero_color': '#E75438',
                    'zero_linewidth': 1,
                    # pdp std fill color and alpha
                    'fill_color': '#66C2D7',
                    'fill_alpha': 0.2,
                    # marker size for pdp line
                    'markersize': 3.5,
                }

        which_classes: list, optional, default=None
            which classes to plot, only use when it is a multi-class problem

        Returns
        -------
        fig: matplotlib Figure
        axes: a dictionary of matplotlib Axes
            Returns the Axes objects for further tweaking

        """

        _check_frac_to_plot(frac_to_plot)
        which_classes = _get_which_classes(self.n_classes, which_classes)

        if plot_params is None:
            plot_params = {}

        plot_params.update(
            {
                "ncols": ncols,
                "figsize": figsize,
                "dpi": dpi,
                "template": template,
                "engine": engine,
                "display_columns": self.display_columns,
                "percentile_columns": self.percentile_columns,
                "n_grids": self.n_grids,
                "plot_lines": plot_lines,
                "frac_to_plot": frac_to_plot,
                "plot_pts_dist": plot_pts_dist,
                "x_quantile": x_quantile,
                "show_percentile": show_percentile,
                "center": center,
                "clustering": {
                    "on": cluster,
                    "n_centers": n_cluster_centers,
                    "method": cluster_method,
                },
            }
        )

        if engine == "matplotlib":
            fig, axes = _pdp_plot(self, feature_name, which_classes, plot_params)
        else:
            fig = _pdp_plot_plotly(self, feature_name, which_classes, plot_params)
            axes = None

        return fig, axes


class PDPInteract:
    def __init__(
        self,
        model,
        dataset,
        model_features,
        features,
        pred_func=None,
        n_classes=None,
        num_grid_points=None,
        grid_types=None,
        percentile_ranges=None,
        grid_ranges=None,
        cust_grid_points=None,
        memory_limit=0.5,
        chunk_size=-1,
        n_jobs=1,
        predict_kwds={},
        data_transformer=None,
    ):

        self.model = model
        self.dataset = dataset
        self.model_features = model_features
        self.features = features
        self.n_classes = n_classes
        self.pred_func = pred_func
        self.num_grid_points = _expand_default(num_grid_points, 10)
        self.grid_types = _expand_default(grid_types, "percentile")
        self.percentile_ranges = _expand_default(percentile_ranges, None)
        self.grid_ranges = _expand_default(grid_ranges, None)
        self.cust_grid_points = _expand_default(cust_grid_points, None)
        self.memory_limit = memory_limit
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs
        self.predict_kwds = predict_kwds
        self.data_transformer = data_transformer

        self.prepare()
        self.calculate()

    def prepare(self):
        self.pdp_isolate_objs = []
        self.feature_grids = []
        self.feature_types = []
        self.n_grids = 1

        for i in range(2):
            obj = PDPIsolate(
                self.model,
                self.dataset,
                self.model_features,
                self.features[i],
                pred_func=self.pred_func,
                n_classes=self.n_classes,
                num_grid_points=self.num_grid_points[i],
                grid_type=self.grid_types[i],
                percentile_range=self.percentile_ranges[i],
                grid_range=self.grid_ranges[i],
                cust_grid_points=self.cust_grid_points[i],
                memory_limit=self.memory_limit,
                chunk_size=self.chunk_size,
                n_jobs=self.n_jobs,
                predict_kwds=self.predict_kwds,
                data_transformer=self.data_transformer,
            )
            obj.dataset = None
            if obj.feature_type == "numeric":
                obj.display_columns = [_get_string(v) for v in obj.feature_grids]
            if len(obj.percentile_columns) > 0:
                obj.percentile_columns = [str(v) for v in obj.percentile_grids]
            self.n_grids *= obj.n_grids
            self.pdp_isolate_objs.append(obj)
            self.feature_grids.append(obj.feature_grids)
            self.feature_types.append(obj.feature_type)

        self.n_classes, self.pred_func, self.from_model = _check_model(
            self.model, self.n_classes, self.pred_func
        )
        self.feature_grid_combos = _get_grid_combos(
            self.feature_grids, self.feature_types
        )
        self.n_jobs = _calc_n_jobs(
            self.dataset, len(self.feature_grid_combos), self.memory_limit, self.n_jobs
        )

    def calculate(self):
        args = {
            "model": self.model,
            "data": self.dataset[self.model_features],
            "feats": self.features,
            "n_classes": self.n_classes,
            "pred_func": self.pred_func,
            "from_model": self.from_model,
            "predict_kwds": self.predict_kwds,
            "data_trans": self.data_transformer,
            "chunk_size": self.chunk_size,
        }

        calc_args = []
        for i, grid_combo in enumerate(self.feature_grid_combos):
            args_ = copy.deepcopy(args)
            args_.update({"feat_grid_combo": grid_combo, "grid_idx": i})
            calc_args.append(args_)

        _collect_pdp_results(self, calc_args, _calc_ice_lines_inter)

    def pdp_interact_plot(
        self,
        feature_names,
        plot_type="contour",
        plot_pdp=False,
        x_quantile=True,
        show_percentile=False,
        which_classes=None,
        figsize=None,
        dpi=300,
        ncols=2,
        plot_params=None,
        engine="plotly",
        template="plotly_white",
    ):
        """PDP interact

        Parameters
        ----------

        pdp_interact_out: (list of) instance of PDPInteract
            for multi-class, it is a list
        feature_names: list
            [feature_name1, feature_name2]
        plot_type: str, optional, default='contour'
            type of the interact plot, can be 'contour' or 'grid'
        x_quantile: bool, default=False
            whether to construct x axis ticks using quantiles
        plot_pdp: bool, default=False
            whether to plot pdp for each feature
        which_classes: list, optional, default=None
            which classes to plot, only use when it is a multi-class problem
        figsize: tuple or None, optional, default=None
            size of the figure, (width, height)
        ncols: integer, optional, default=2
            number subplot columns, used when it is multi-class problem
        plot_params: dict or None, optional, default=None
            parameters for the plot, possible parameters as well as default as below:

            .. highlight:: python
            .. code-block:: python

                plot_params = {
                    # plot title and subtitle
                    'title': 'PDP interact for "%s" and "%s"',
                    'subtitle': 'Number of unique grid points: (%s: %d, %s: %d)',
                    'title_fontsize': 15,
                    'subtitle_fontsize': 12,
                    # color for contour line
                    'contour_color':  'white',
                    'font_family': 'Arial',
                    # matplotlib color map for interact plot
                    'cmap': 'viridis',
                    # fill alpha for interact plot
                    'inter_fill_alpha': 0.8,
                    # fontsize for interact plot text
                    'inter_fontsize': 9,
                }

        Returns
        -------
        fig: matplotlib Figure
        axes: a dictionary of matplotlib Axes
            Returns the Axes objects for further tweaking
        """

        which_classes = _get_which_classes(self.n_classes, which_classes)

        if plot_params is None:
            plot_params = {}

        if (
            not all(v == "numeric" for v in self.feature_types)
            or plot_pdp
            or plot_type == "grid"
        ):
            x_quantile = True

        if any(v == "numeric" for v in self.feature_types) and not x_quantile:
            show_percentile = False

        plot_params.update(
            {
                "ncols": ncols,
                "figsize": figsize,
                "dpi": dpi,
                "template": template,
                "engine": engine,
                "n_grids": [self.pdp_isolate_objs[i].n_grids for i in range(2)],
                "display_columns": [
                    self.pdp_isolate_objs[i].display_columns for i in range(2)
                ],
                "percentile_columns": [
                    self.pdp_isolate_objs[i].percentile_columns for i in range(2)
                ],
                "plot_pdp": plot_pdp,
                "x_quantile": x_quantile,
                "plot_type": plot_type,
                "show_percentile": show_percentile,
            }
        )

        if engine == "matplotlib":
            fig, axes = _pdp_inter_plot(self, feature_names, which_classes, plot_params)
        else:
            fig = _pdp_inter_plot_plotly(
                self, feature_names, which_classes, plot_params
            )
            axes = None

        return fig, axes
