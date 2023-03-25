from .pdp_calc_utils import (
    _calc_ice_lines,
    _calc_ice_lines_inter,
    _prepare_pdp_count_data,
)
from .pdp_plot_utils import (
    _pdp_plot,
    _pdp_plot_plotly,
    _pdp_inter_three,
    _pdp_inter_one,
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

from joblib import Parallel, delayed
from pqdm.processes import pqdm
import copy

import warnings

warnings.filterwarnings("ignore")


class PDPIsolateResult:
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
        self.percentile_grids = prepared_results["percentile_grids"]
        self.count_data = prepared_results["count"]
        self.dist_data = prepared_results["dist"]
        self.display_columns = prepared_results["value_display"][0]
        self.percentile_columns = prepared_results["percentile_display"][0]
        if len(self.percentile_columns) > 0:
            self.percentile_columns = [str(v) for v in self.percentile_grids]
        self.n_jobs = _calc_n_jobs(
            self.dataset, self.feature_grids, self.memory_limit, self.n_jobs
        )

    def calculate(self):
        args = {
            "model": self.model,
            "data": self.dataset.copy(),
            "features": self.model_features,
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
        for grid in self.feature_grids:
            args_ = copy.deepcopy(args)
            args_.update({"feat_grid": grid})
            calc_args.append(args_)
        grid_results = pqdm(
            calc_args,
            _calc_ice_lines,
            n_jobs=self.n_jobs,
            argument_type="kwargs",
            desc="calculate per feature grid",
        )

        self.results = []
        if self.n_classes > 2:
            for cls_idx in range(self.n_classes):
                ice_lines = pd.concat([res[cls_idx] for res in grid_results], axis=1)
                pdp = ice_lines[self.feature_grids].mean().values
                self.results.append(PDPIsolateResult(cls_idx, ice_lines, pdp))
        else:
            ice_lines = pd.concat(grid_results, axis=1)
            pdp = ice_lines[self.feature_grids].mean().values
            self.results.append(PDPIsolateResult(None, ice_lines, pdp))

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
        n_grids = len(self.feature_grids)
        if which_classes is not None:
            _check_classes(which_classes, self.n_classes)
        else:
            if self.n_classes <= 2:
                which_classes = [0]
            else:
                which_classes = np.arange(self.n_classes)

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
                "n_grids": n_grids,
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
    """Save pdp_interact results

    Parameters
    ----------

    n_classes: integer or None
        number of classes for classifier, None when it is a regressor
    which_class: integer or None
        for multi-class classifier, indicate which class the result belongs to
    features: list
        [feature1, feature2]
    feature_types: list
        [feature1 type, feature2 type]
    feature_grids: list
        [feature1 grid points, feature2 grid points]
    pdp_isolate_outs: list
        [feature1 pdp_isolate result, feature2 pdp_isolate result]
    pdp: pandas DataFrame
        calculated PDP values for each gird combination
    """

    def __init__(
        self,
        n_classes,
        which_class,
        features,
        feature_types,
        feature_grids,
        pdp_isolate_outs,
        pdp,
    ):

        self._type = "PDPInteract_instance"
        self.n_classes = n_classes
        self.which_class = which_class
        self.features = features
        self.feature_types = feature_types
        self.feature_grids = feature_grids
        self.pdp_isolate_outs = pdp_isolate_outs
        self.pdp = pdp


def pdp_interact(
    model,
    dataset,
    model_features,
    features,
    num_grid_points=None,
    grid_types=None,
    percentile_ranges=None,
    grid_ranges=None,
    cust_grid_points=None,
    memory_limit=0.5,
    n_jobs=1,
    predict_kwds={},
    data_transformer=None,
):
    """Calculate PDP interaction plot

    Parameters
    ----------
    model: a fitted sklearn model
    dataset: pandas DataFrame
        data set on which the model is trained
    model_features: list or 1-d array
        list of model features
    features: list
        [feature1, feature2]
    num_grid_points: list, default=None
        [feature1 num_grid_points, feature2 num_grid_points]
    grid_types: list, default=None
        [feature1 grid_type, feature2 grid_type]
    percentile_ranges: list, default=None
        [feature1 percentile_range, feature2 percentile_range]
    grid_ranges: list, default=None
        [feature1 grid_range, feature2 grid_range]
    cust_grid_points: list, default=None
        [feature1 cust_grid_points, feature2 cust_grid_points]
    memory_limit: float, (0, 1)
        fraction of memory to use
    n_jobs: integer, default=1
        number of jobs to run in parallel.
        make sure n_jobs=1 when you are using XGBoost model.
        check:
        1. https://pythonhosted.org/joblib/parallel.html#bad-interaction-of-multiprocessing-and-third-party-libraries
        2. https://github.com/scikit-learn/scikit-learn/issues/6627
    predict_kwds: dict, optional, default={}
        keywords to be passed to the model's predict function
    data_transformer: function or None, optional, default=None
        function to transform the data set as some features changing values

    Returns
    -------
    pdp_interact_out: instance of PDPInteract
    """

    # check function inputs
    n_classes, predict = _check_model(model=model)
    _check_dataset(df=dataset)
    _dataset = dataset.copy()

    num_grid_points = _expand_default(x=num_grid_points, default=10)
    grid_types = _expand_default(x=grid_types, default="percentile")
    _check_grid_type(grid_type=grid_types[0])
    _check_grid_type(grid_type=grid_types[1])

    percentile_ranges = _expand_default(x=percentile_ranges, default=None)
    _check_percentile_range(percentile_range=percentile_ranges[0])
    _check_percentile_range(percentile_range=percentile_ranges[1])

    grid_ranges = _expand_default(x=grid_ranges, default=None)
    cust_grid_points = _expand_default(x=cust_grid_points, default=None)

    _check_memory_limit(memory_limit=memory_limit)

    # calculate pdp_isolate for each feature
    pdp_isolate_outs = []
    for idx in range(2):
        pdp_isolate_out = pdp_isolate(
            model=model,
            dataset=_dataset,
            model_features=model_features,
            feature=features[idx],
            num_grid_points=num_grid_points[idx],
            grid_type=grid_types[idx],
            percentile_range=percentile_ranges[idx],
            grid_range=grid_ranges[idx],
            cust_grid_points=cust_grid_points[idx],
            memory_limit=memory_limit,
            n_jobs=n_jobs,
            predict_kwds=predict_kwds,
            data_transformer=data_transformer,
        )
        pdp_isolate_outs.append(pdp_isolate_out)

    if n_classes > 2:
        feature_grids = [
            pdp_isolate_outs[0][0].feature_grids,
            pdp_isolate_outs[1][0].feature_grids,
        ]
        feature_types = [
            pdp_isolate_outs[0][0].feature_type,
            pdp_isolate_outs[1][0].feature_type,
        ]
    else:
        feature_grids = [
            pdp_isolate_outs[0].feature_grids,
            pdp_isolate_outs[1].feature_grids,
        ]
        feature_types = [
            pdp_isolate_outs[0].feature_type,
            pdp_isolate_outs[1].feature_type,
        ]

    # make features into list
    feature_list = _make_list(features[0]) + _make_list(features[1])

    # create grid combination
    grid_combos = _get_grid_combos(feature_grids, feature_types)

    # Parallel calculate ICE lines
    true_n_jobs = _calc_memory_usage(
        df=_dataset,
        total_units=len(grid_combos),
        n_jobs=n_jobs,
        memory_limit=memory_limit,
    )

    grid_results = Parallel(n_jobs=true_n_jobs)(
        delayed(_calc_ice_lines_inter)(
            grid_combo,
            data=_dataset,
            model=model,
            model_features=model_features,
            n_classes=n_classes,
            feature_list=feature_list,
            predict_kwds=predict_kwds,
            data_transformer=data_transformer,
        )
        for grid_combo in grid_combos
    )

    ice_lines = pd.concat(grid_results, axis=0).reset_index(drop=True)
    pdp = ice_lines.groupby(feature_list, as_index=False).mean()

    # combine the final results
    pdp_interact_params = {
        "n_classes": n_classes,
        "features": features,
        "feature_types": feature_types,
        "feature_grids": feature_grids,
    }
    if n_classes > 2:
        pdp_interact_out = []
        for n_class in range(n_classes):
            _pdp = pdp[feature_list + ["class_%d_preds" % n_class]].rename(
                columns={"class_%d_preds" % n_class: "preds"}
            )
            pdp_interact_out.append(
                PDPInteract(
                    which_class=n_class,
                    pdp_isolate_outs=[
                        pdp_isolate_outs[0][n_class],
                        pdp_isolate_outs[1][n_class],
                    ],
                    pdp=_pdp,
                    **pdp_interact_params
                )
            )
    else:
        pdp_interact_out = PDPInteract(
            which_class=None,
            pdp_isolate_outs=pdp_isolate_outs,
            pdp=pdp,
            **pdp_interact_params
        )

    return pdp_interact_out


def pdp_interact_plot(
    pdp_interact_out,
    feature_names,
    plot_type="contour",
    x_quantile=False,
    plot_pdp=False,
    which_classes=None,
    figsize=None,
    ncols=2,
    plot_params=None,
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

    Examples
    --------

    Quick start with pdp_interact_plot

    .. highlight:: python
    .. code-block:: python

        from pdpbox import pdp, get_dataset

        test_titanic = get_dataset.titanic()
        titanic_data = test_titanic['data']
        titanic_target = test_titanic['target']
        titanic_features = test_titanic['features']
        titanic_model = test_titanic['xgb_model']

        inter1 = pdp.pdp_interact(model=titanic_model,
                                  dataset=titanic_data,
                                  model_features=titanic_features,
                                  features=['Age', 'Fare'],
                                  num_grid_points=[10, 10],
                                  percentile_ranges=[(5, 95), (5, 95)])
        fig, axes = pdp.pdp_interact_plot(pdp_interact_out=inter1,
                                          feature_names=['age', 'fare'],
                                          plot_type='contour',
                                          x_quantile=True,
                                          plot_pdp=True)


    With multi-class

    .. highlight:: python
    .. code-block:: python

        from pdpbox import pdp, get_dataset

        test_otto = get_dataset.otto()
        otto_data = test_otto['data']
        otto_features = test_otto['features']
        otto_model = test_otto['rf_model']
        otto_target = test_otto['target']

        pdp_67_24_rf = pdp.pdp_interact(model=otto_model,
                                        dataset=otto_data,
                                        model_features=otto_features,
                                        features=['feat_67', 'feat_24'],
                                        num_grid_points=[10, 10],
                                        percentile_ranges=[None, None],
                                        n_jobs=4)
        fig, axes = pdp.pdp_interact_plot(pdp_interact_out=pdp_67_24_rf,
                                          feature_names=['feat_67', 'feat_24'],
                                          plot_type='grid',
                                          x_quantile=True,
                                          ncols=2,
                                          plot_pdp=False,
                                          which_classes=[1, 2, 3])

    """

    pdp_interact_plot_data = _make_list(x=pdp_interact_out)
    if which_classes is not None:
        _check_classes(
            classes_list=which_classes, n_classes=pdp_interact_plot_data[0].n_classes
        )

    # multi-class problem
    if len(pdp_interact_plot_data) > 1 and which_classes is not None:
        pdp_interact_plot_data = []
        for n_class in which_classes:
            pdp_interact_plot_data.append(pdp_interact_out[n_class])
    num_charts = len(pdp_interact_plot_data)

    inner_hspace = inner_wspace = 0
    if plot_type == "grid" or plot_pdp:
        x_quantile = True
        if plot_type == "grid":
            inner_hspace = inner_wspace = 0.1

    # calculate figure size
    title_height = 2
    if plot_pdp:
        unit_figsize = (10.5, 10.5)
    else:
        unit_figsize = (7.5, 7.5)

    width, height, nrows, ncols = _calc_figsize(
        num_charts=num_charts,
        ncols=ncols,
        title_height=title_height,
        unit_figsize=unit_figsize,
    )
    if figsize is not None:
        width, height = figsize

    if plot_params is None:
        plot_params = dict()

    fig = plt.figure(figsize=(width, height))
    outer_grid = GridSpec(
        2,
        1,
        wspace=0.0,
        hspace=0.1,
        height_ratios=[title_height, height - title_height],
    )
    title_ax = plt.subplot(outer_grid[0])
    fig.add_subplot(title_ax)

    n_grids = [
        len(pdp_interact_plot_data[0].feature_grids[0]),
        len(pdp_interact_plot_data[0].feature_grids[1]),
    ]
    title = plot_params.get(
        "title", 'PDP interact for "%s" and "%s"' % (feature_names[0], feature_names[1])
    )
    subtitle = plot_params.get(
        "subtitle",
        "Number of unique grid points: (%s: %d, %s: %d)"
        % (feature_names[0], n_grids[0], feature_names[1], n_grids[1]),
    )

    _plot_title(
        title=title, subtitle=subtitle, title_ax=title_ax, plot_params=plot_params
    )

    inter_params = {
        "plot_type": plot_type,
        "x_quantile": x_quantile,
        "plot_params": plot_params,
    }
    if num_charts == 1:
        feature_names_adj = feature_names
        if pdp_interact_plot_data[0].which_class is not None:
            feature_names_adj = [
                "%s (class %d)"
                % (feature_names[0], pdp_interact_plot_data[0].which_class),
                feature_names[1],
            ]
        if plot_pdp:
            inner_grid = GridSpecFromSubplotSpec(
                2,
                2,
                subplot_spec=outer_grid[1],
                height_ratios=[0.5, 7],
                width_ratios=[0.5, 7],
                hspace=inner_hspace,
                wspace=inner_wspace,
            )
            inter_ax = _pdp_inter_three(
                pdp_interact_out=pdp_interact_plot_data[0],
                chart_grids=inner_grid,
                fig=fig,
                feature_names=feature_names_adj,
                **inter_params
            )
        else:
            inter_ax = plt.subplot(outer_grid[1])
            fig.add_subplot(inter_ax)
            _pdp_inter_one(
                pdp_interact_out=pdp_interact_plot_data[0],
                inter_ax=inter_ax,
                norm=None,
                feature_names=feature_names_adj,
                **inter_params
            )
    else:
        wspace = 0.3
        if plot_pdp and plot_type == "grid":
            wspace = 0.35
        inner_grid = GridSpecFromSubplotSpec(
            nrows, ncols, subplot_spec=outer_grid[1], wspace=wspace, hspace=0.2
        )
        inter_ax = []
        for inner_idx in range(num_charts):
            feature_names_adj = [
                "%s (class %d)"
                % (feature_names[0], pdp_interact_plot_data[inner_idx].which_class),
                feature_names[1],
            ]
            if plot_pdp:
                inner_inner_grid = GridSpecFromSubplotSpec(
                    2,
                    2,
                    subplot_spec=inner_grid[inner_idx],
                    height_ratios=[0.5, 7],
                    width_ratios=[0.5, 7],
                    hspace=inner_hspace,
                    wspace=inner_wspace,
                )
                inner_inter_ax = _pdp_inter_three(
                    pdp_interact_out=pdp_interact_plot_data[inner_idx],
                    chart_grids=inner_inner_grid,
                    fig=fig,
                    feature_names=feature_names_adj,
                    **inter_params
                )
            else:
                inner_inter_ax = plt.subplot(inner_grid[inner_idx])
                fig.add_subplot(inner_inter_ax)
                _pdp_inter_one(
                    pdp_interact_out=pdp_interact_plot_data[inner_idx],
                    inter_ax=inner_inter_ax,
                    norm=None,
                    feature_names=feature_names_adj,
                    **inter_params
                )
            inter_ax.append(inner_inter_ax)

    axes = {"title_ax": title_ax, "pdp_inter_ax": inter_ax}

    return fig, axes
