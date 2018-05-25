
from .pdp_calc_utils import _get_grids, _calc_ice_lines, _calc_ice_lines_inter, _prepare_pdp_count_data, _get_grid_combos
from .pdp_plot_utils import (_pdp_plot_title, _pdp_plot, _pdp_interact_plot_title,
                             _pdp_contour_plot, _pdp_interact_plot, _pdp_inter_three, _pdp_inter_one)
from .other_utils import (_check_model, _check_dataset, _check_percentile_range, _check_feature,
                          _check_grid_type, _check_memory_limit, _check_frac_to_plot, _make_list, _expand_default)
from .info_plot_utils import _calc_figsize

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from joblib import Parallel, delayed
import psutil

import warnings
warnings.filterwarnings('ignore')


class PDPIsolate(object):
    """Save pdp_isolate results

    Parameters
    ----------

    n_classes: integer or None
        number of classes for classifier, None when it is a regressor
    classifier: bool
        whether the model is a classifier
    which_class: integer or None
        for multi-class classifier, indicate which class the result belongs to
    feature: string or list
        which feature is calculated on, list for one-hot encoding features
    feature_type: string
        type of the feature
    feature_grids: list
        feature grids
    percentile_info: list
        percentile information for feature grids
    display_columns: list
        columns to display as xticklabels
    ice_lines: pandas DataFrame
        ICE lines
    pdp: 1-d numpy array
        calculated PDP values
    count_data: pandas DataFrame
        data points distribution
    hist_data: 1-d numpy array
        data points distribution for numeric features
    """

    def __init__(self, n_classes, classifier, which_class, feature, feature_type, feature_grids,
                 percentile_info, display_columns, ice_lines, pdp, count_data, hist_data):

        self._type = 'PDPIsolate_instance'
        self.n_classes = n_classes
        self.classifier = classifier
        self.which_class = which_class
        self.feature = feature
        self.feature_type = feature_type
        self.feature_grids = feature_grids
        self.percentile_info = percentile_info
        self.display_columns = display_columns
        self.ice_lines = ice_lines
        self.pdp = pdp
        self.count_data = count_data
        self.hist_data = hist_data


def pdp_isolate(model, train_X, feature, num_grid_points=10, grid_type='percentile',
                percentile_range=None, grid_range=None, cust_grid_points=None,
                memory_limit=0.5, n_jobs=1, predict_kwds=None, data_transformer=None):
    """Calculate PDP isolation plot

    Parameters
    ----------
    model: a fitted sklearn model
    train_X: pandas DataFrame
        data set on which the model is trained
    feature: string or list
        feature or feature list to investigate,
        for one-hot encoding features, feature list is required
    num_grid_points: integer, optional, default=10
        number of grid points for numeric feature
    grid_type: string, optional, default='percentile'
        'percentile' or 'equal',
        type of grid points for numeric feature
    percentile_range: tuple or None, optional, default=None
        percentile range to investigate,
        for numeric feature when grid_type='percentile'
    grid_range: tuple or None, optional, default=None
        value range to investigate,
        for numeric feature when grid_type='equal'
    cust_grid_points: Series, 1d-array, list or None, optional, default=None
        customized list of grid points for numeric feature
    memory_limit: float, (0, 1)
        fraction of memory to use
    n_jobs: integer, default=1
        number of jobs to run in parallel
    predict_kwds: dict or None, optional, default=None
        keywords to be passed to the model's predict function
    data_transformer: function or None, optional, default=None
        function to transform the data set as some features changing values

    Returns
    -------
    pdp_isolate_out: instance of PDPIsolate
    """

    # check function inputs
    n_classes, classifier, predict = _check_model(model=model)

    # avoid polluting the original dataset
    # copy training data set and get the model features
    # it's extremely important to keep the original feature order
    _check_dataset(df=train_X)
    _train_X = train_X.copy()
    model_features = _train_X.columns.values

    feature_type = _check_feature(feature=feature, df=_train_X)
    _check_grid_type(grid_type=grid_type)
    _check_percentile_range(percentile_range=percentile_range)
    _check_memory_limit(memory_limit=memory_limit)

    if predict_kwds is None:
        predict_kwds = dict()

    # create feature grids
    percentile_info = []
    if feature_type == 'binary':
        feature_grids = np.array([0, 1])
        display_columns = ['%s_0' % feature, '%s_1' % feature]
    elif feature_type == 'onehot':
        feature_grids = np.array(feature)
        display_columns = feature
    else:
        # calculate grid points for numeric features
        if cust_grid_points is None:
            feature_grids, percentile_info = _get_grids(
                x=_train_X[feature].values, num_grid_points=num_grid_points, grid_type=grid_type,
                percentile_range=percentile_range, grid_range=grid_range)
        else:
            feature_grids = np.array(sorted(cust_grid_points))

        display_columns = [round(v, 3) for v in feature_grids]

    # calculate memory usage
    unit_memory = _train_X.memory_usage(deep=True).sum()
    free_memory = psutil.virtual_memory()[1] * memory_limit
    num_units = int(np.floor(free_memory / unit_memory))
    true_n_jobs = np.min([num_units, n_jobs])

    grid_results = Parallel(n_jobs=true_n_jobs)(
        delayed(_calc_ice_lines)(
            _train_X, model, classifier, model_features, n_classes, feature, feature_type,
            feature_grid, predict_kwds, data_transformer) for feature_grid in feature_grids)

    if n_classes > 2:
        ice_lines = []
        for n_class in range(n_classes):
            ice_line_n_class = pd.concat([grid_result[n_class] for grid_result in grid_results], axis=1)
            ice_lines.append(ice_line_n_class)
    else:
        ice_lines = pd.concat(grid_results, axis=1)

    # calculate the counts
    count_data = _prepare_pdp_count_data(
        feature=feature, feature_type=feature_type, data=_train_X[_make_list(feature)], feature_grids=feature_grids)

    hist_data = None
    if feature_type == 'numeric':
        hist_data = np.histogram(_train_X[feature].values,
                                 bins=np.min([100, _train_X[feature].nunique()]), normed=True)[0]

    # combine the final results
    if n_classes > 2:
        pdp_isolate_out = []
        for n_class in range(n_classes):
            pdp = ice_lines[n_class][feature_grids].mean().values
            pdp_isolate_out.append(PDPIsolate(
                n_classes=n_classes, classifier=classifier, which_class=n_class, feature=feature,
                feature_type=feature_type, feature_grids=feature_grids, percentile_info=percentile_info,
                display_columns=display_columns, ice_lines=ice_lines[n_class], pdp=pdp,
                count_data=count_data, hist_data=hist_data))
    else:
        pdp = ice_lines[feature_grids].mean().values
        pdp_isolate_out = PDPIsolate(
            n_classes=n_classes, classifier=classifier, which_class=None, feature=feature,
            feature_type=feature_type, feature_grids=feature_grids, percentile_info=percentile_info,
            display_columns=display_columns, ice_lines=ice_lines, pdp=pdp, count_data=count_data, hist_data=hist_data)

    return pdp_isolate_out


def pdp_plot(pdp_isolate_out, feature_name, center=True, plot_pts_dist=False, plot_lines=False, frac_to_plot=1,
             cluster=False, n_cluster_centers=None, cluster_method='accurate', x_quantile=False,
             show_percentile=False, figsize=None, ncols=None, plot_params=None, which_classes=None):
    """Plot partial dependent plot

    Parameters
    ----------

    pdp_isolate_out: (list of) instance of PDPIsolate
        for multi-class, it is a list
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
        parameters for the plot
    which_classes: list, optional, default=None
        which classes to plot, only use when it is a multi-class problem

    Returns
    -------
    fig: matplotlib Figure
    axes: a dictionary of matplotlib Axes
        Returns the Axes objects for further tweaking

    Examples
    --------

    PDP for binary features

    .. highlight:: python
    .. code-block:: python

        from pdpbox import pdp, get_dataset

        test_titanic = get_dataset.titanic()
        titanic_data = test_titanic['data']
        titanic_target = test_titanic['target']

        pdp_sex = pdp.pdp_isolate(model=titanic_model,
                                  train_X=titanic_data[titanic_features],
                                  feature='Sex')
        fig, axes = pdp.pdp_plot(pdp_sex, 'sex')


    PDP for one-hot encoding features

    .. highlight:: python
    .. code-block:: python

        pdp_embark = pdp.pdp_isolate(titanic_model,
                                     titanic_data[titanic_features],
                                     ['Embarked_C', 'Embarked_S', 'Embarked_Q'])
        fig, axes = pdp.pdp_plot(pdp_embark, 'Embark', center=True, plot_lines=True, frac_to_plot=100)


    PDP for numeric features

    .. highlight:: python
    .. code-block:: python

        pdp_fare = pdp.pdp_isolate(titanic_model, titanic_data[titanic_features], 'Fare')
        fig, axes = pdp.pdp_plot(pdp_fare, 'Fare')


    PDP for multi-class problem

    .. highlight:: python
    .. code-block:: python

        from pdpbox import pdp, get_dataset

        test_otto = get_dataset.otto()
        otto_data = test_otto['data']
        otto_target = test_otto['target']

        pdp_feat_67_rf = pdp.pdp_isolate(otto_model, otto_data[otto_features], 'feat_67', n_jobs=2)
        fig, axes = pdp.pdp_plot(pdp_feat_67_rf, 'feat_67', center=True, x_quantile=True, ncols=2,
                                 plot_lines=True, frac_to_plot=100, which_classes=[0, 3, 7])
    """

    # check function inputs
    _check_frac_to_plot(frac_to_plot=frac_to_plot)
    pdp_plot_data = _make_list(x=pdp_isolate_out)
    n_grids = len(pdp_plot_data[0].feature_grids)

    if len(pdp_plot_data) > 1 and which_classes is not None:
        pdp_plot_data = []
        for n_class in which_classes:
            pdp_plot_data.append(pdp_isolate_out[n_class])

    # set up graph parameters
    width, height = 15, 9.5
    nrows = 1

    if len(pdp_plot_data) > 1:
        nrows = int(np.ceil(len(pdp_plot_data) * 1.0 / ncols))
        ncols = np.min([len(pdp_plot_data), ncols])
        width = np.min([7.5 * len(pdp_plot_data), 15])
        height = np.min([width * 1.0 / ncols, 8]) * nrows

    if figsize is not None:
        width, height = figsize

    if plot_params is None:
        plot_params = dict()

    fig = plt.figure(figsize=(width, height))
    outer_grid = GridSpec(2, 1, wspace=0.0, hspace=0.1, height_ratios=[2, height - 2])
    title_ax = plt.subplot(outer_grid[0])
    fig.add_subplot(title_ax)
    _pdp_plot_title(n_grids=n_grids, feature_name=feature_name, ax=title_ax, plot_params=plot_params)

    # prepare count data when plot_pts_dist=True
    # adjust when it is numeric feature
    feature_type = pdp_plot_data[0].feature_type
    pdp_count_hspace = 0.15
    count_data = pdp_plot_data[0].count_data.copy()
    if feature_type == 'numeric':
        pdp_count_hspace = 0.25

    if len(pdp_plot_data) == 1:
        feature_name_adj = feature_name
        if pdp_plot_data[0].which_class is not None:
            feature_name_adj = '%s (class %d)' % (feature_name, pdp_plot_data[0].which_class)

        if plot_pts_dist:
            inner_grid = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[1],
                                                 height_ratios=[7, 0.5], hspace=pdp_count_hspace)
            _pdp_ax = plt.subplot(inner_grid[0])
            fig.add_subplot(_pdp_ax)
            _count_ax = plt.subplot(inner_grid[1])
            fig.add_subplot(_count_ax, sharex=_pdp_ax)
            pdp_ax = {'_pdp_ax': _pdp_ax, '_count_ax': _count_ax}

            _pdp_plot(
                pdp_isolate_out=pdp_plot_data[0], feature_name=feature_name_adj, center=center, plot_lines=plot_lines,
                frac_to_plot=frac_to_plot, cluster=cluster, n_cluster_centers=n_cluster_centers,
                cluster_method=cluster_method, x_quantile=x_quantile, show_percentile=show_percentile,
                pdp_ax=_pdp_ax, count_data=count_data, count_ax=_count_ax, plot_params=plot_params)
        else:
            pdp_ax = plt.subplot(outer_grid[1])
            fig.add_subplot(pdp_ax)
            _pdp_plot(
                pdp_isolate_out=pdp_plot_data[0], feature_name=feature_name_adj, center=center, plot_lines=plot_lines,
                frac_to_plot=frac_to_plot, cluster=cluster, n_cluster_centers=n_cluster_centers,
                cluster_method=cluster_method, x_quantile=x_quantile, show_percentile=show_percentile,
                pdp_ax=pdp_ax, count_data=None, count_ax=None, plot_params=plot_params)

    else:
        inner_grid = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=outer_grid[1], wspace=0.1, hspace=0.2)
        if plot_pts_dist:
            pdp_ax = []
            for inner_idx in range(len(pdp_plot_data)):
                inner_inner_grid = GridSpecFromSubplotSpec(2, 1, subplot_spec=inner_grid[inner_idx],
                                                           height_ratios=[7, 0.5], hspace=pdp_count_hspace)
                _pdp_ax = plt.subplot(inner_inner_grid[0])
                _count_ax = plt.subplot(inner_inner_grid[1])
                pdp_ax.append({'_pdp_ax': _pdp_ax, '_count_ax': _count_ax})
                fig.add_subplot(_pdp_ax)
                fig.add_subplot(_count_ax, sharex=_pdp_ax)

                feature_name_adj = '%s (class %d)' % (feature_name, pdp_plot_data[inner_idx].which_class)
                _pdp_plot(pdp_isolate_out=pdp_plot_data[inner_idx], feature_name=feature_name_adj, center=center,
                          plot_lines=plot_lines, frac_to_plot=frac_to_plot, cluster=cluster,
                          n_cluster_centers=n_cluster_centers, cluster_method=cluster_method, x_quantile=x_quantile,
                          show_percentile=show_percentile, pdp_ax=_pdp_ax,
                          count_data=count_data, count_ax=_count_ax, plot_params=plot_params)
        else:
            pdp_ax = []
            for inner_idx in range(len(pdp_plot_data)):
                ax = plt.subplot(inner_grid[inner_idx])
                pdp_ax.append(ax)
                fig.add_subplot(ax)

                feature_name_adj = '%s (class %d)' % (feature_name, pdp_plot_data[inner_idx].which_class)
                _pdp_plot(pdp_isolate_out=pdp_plot_data[inner_idx], feature_name=feature_name_adj, center=center,
                          plot_lines=plot_lines, frac_to_plot=frac_to_plot, cluster=cluster,
                          n_cluster_centers=n_cluster_centers, cluster_method=cluster_method, x_quantile=x_quantile,
                          show_percentile=show_percentile, pdp_ax=ax, count_data=None, count_ax=None,
                          plot_params=plot_params)

    axes = {'title_ax': title_ax, 'pdp_ax': pdp_ax}
    return fig, axes


class PDPInteract:
    def __init__(self, n_classes, classifier, model_features, features, feature_types, feature_grids,
                 pdp_isolate_out1, pdp_isolate_out2, pdp):

        self._type = 'PDPInteract_instance'
        self.n_classes = n_classes
        self.classifier = classifier
        self.model_features = model_features
        self.features = features
        self.feature_types = feature_types
        self.feature_grids = feature_grids
        self.pdp_isolate_out1 = pdp_isolate_out1
        self.pdp_isolate_out2 = pdp_isolate_out2
        self.pdp = pdp


def pdp_interact(model, train_X, features, num_grid_points=None, grid_types=None, percentile_ranges=None, grid_ranges=None,
                 cust_grid_points=None, memory_limit=0.5, n_jobs=1, predict_kwds={}, data_transformer=None):
    """
    Calculate PDP interaction plot

    :param model: sklearn model
        a fitted model
    :param train_X: pandas DataFrame
        dataset on which the model is trained
    :param features: list
        a list containing two features
    :param num_grid_points: list, default=[10, 10]
        a list of number of grid points for each feature
    :param grid_types: list, default=['percentile', 'percentile']
        a list of grid types for each feature
    :param percentile_ranges: list, default=[None, None]
        a list of percentile range to consider for each feature
    :param grid_ranges: list, default=[None, None]
        a list of grid range to consider for each feature
    :param cust_grid_points: list, default=None
        a list of customized grid points
    :param memory_limit: float, default=0.5
        fraction of RAM can be used to do the calculation
    :param n_jobs: integer, default=1
        the number of jobs to run in parallel
    :param predict_kwds: dict, default={}
        keywords to be passed to the model's predict function
    :param data_transformer: function
        function to transform the data set as some features changing values

    :return:
        instance of pdp_interact_obj
    """

    # check function inputs
    _check_dataset(df=train_X)
    _train_X = train_X.copy()
    model_features = _train_X.columns.values

    num_grid_points = _expand_default(num_grid_points, 10)
    grid_types = _expand_default(grid_types, 'percentile')
    _check_grid_type(grid_type=grid_types[0])
    _check_grid_type(grid_type=grid_types[1])

    percentile_ranges = _expand_default(percentile_ranges, None)
    _check_percentile_range(percentile_range=percentile_ranges[0])
    _check_percentile_range(percentile_range=percentile_ranges[1])

    grid_ranges = _expand_default(grid_ranges, None)
    cust_grid_points = _expand_default(cust_grid_points, None)

    _check_memory_limit(memory_limit=memory_limit)

    # calculate pdp_isolate for each feature
    pdp_isolate_out1 = pdp_isolate(model, _train_X, features[0], num_grid_points=num_grid_points[0],
                                   grid_type=grid_types[0], percentile_range=percentile_ranges[0],
                                   grid_range=grid_ranges[0], cust_grid_points=cust_grid_points[0],
                                   memory_limit=memory_limit, n_jobs=n_jobs, predict_kwds=predict_kwds,
                                   data_transformer=data_transformer)
    pdp_isolate_out2 = pdp_isolate(model, _train_X, features[1], num_grid_points=num_grid_points[1],
                                   grid_type=grid_types[1], percentile_range=percentile_ranges[1],
                                   grid_range=grid_ranges[1], cust_grid_points=cust_grid_points[1],
                                   memory_limit=memory_limit, n_jobs=n_jobs, predict_kwds=predict_kwds,
                                   data_transformer=data_transformer)

    # whether it is for multi-classes
    if type(pdp_isolate_out1) == list:
        feature_grids = [pdp_isolate_out1[0].feature_grids, pdp_isolate_out2[0].feature_grids]
        feature_types = [pdp_isolate_out1[0].feature_type, pdp_isolate_out2[0].feature_type]
        classifier = pdp_isolate_out1[0].classifier
        n_classes = pdp_isolate_out1[0].n_classes
    else:
        feature_grids = [pdp_isolate_out1.feature_grids, pdp_isolate_out2.feature_grids]
        feature_types = [pdp_isolate_out1.feature_type, pdp_isolate_out2.feature_type]
        classifier = pdp_isolate_out1.classifier
        n_classes = pdp_isolate_out1.n_classes

    # make features into list
    feature_list = _make_list(features[0]) + _make_list(features[1])

    # calculate memory usage
    unit_memory = _train_X.memory_usage(deep=True).sum()
    free_memory = psutil.virtual_memory()[1] * memory_limit
    num_units = int(np.floor(free_memory / unit_memory))
    true_n_jobs = np.min([num_units, n_jobs])

    # create grid combination
    grid_combos = _get_grid_combos(feature_grids, feature_types)

    grid_results = Parallel(n_jobs=true_n_jobs)(
        delayed(_calc_ice_lines_inter)(
            _train_X, model, classifier, model_features, n_classes, feature_list, grid_combo, predict_kwds,
            data_transformer) for grid_combo in grid_combos)

    ice_lines = pd.concat(grid_results, axis=0).reset_index(drop=True)
    pdp = ice_lines.groupby(feature_list, as_index=False).mean()

    # combine the final results
    if n_classes > 2:
        pdp_interact_out = []
        for n_class in range(n_classes):
            _pdp = pdp[feature_list + ['class_%d_preds' % n_class]].rename(columns={'class_%d_preds' % n_class: 'preds'})
            pdp_interact_out.append(PDPInteract(
                n_classes=n_classes, classifier=classifier, model_features=model_features, features=features,
                feature_types=feature_types, feature_grids=feature_grids, pdp_isolate_out1=pdp_isolate_out1[n_class],
                pdp_isolate_out2=pdp_isolate_out2[n_class], pdp=_pdp))
    else:
        pdp_interact_out = PDPInteract(
            n_classes=n_classes, classifier=classifier, model_features=model_features, features=features,
            feature_types=feature_types, feature_grids=feature_grids, pdp_isolate_out1=pdp_isolate_out1,
            pdp_isolate_out2=pdp_isolate_out2, pdp=pdp)

    return pdp_interact_out


def pdp_interact_plot(pdp_interact_out, feature_names, plot_type='contour', x_quantile=False, plot_pdp=False,
                      which_classes=None, figsize=None, ncols=2, plot_params=None):

    pdp_interact_plot_data = _make_list(x=pdp_interact_out)

    # multi-class problem
    if len(pdp_interact_plot_data) > 1 and which_classes is not None:
        pdp_interact_plot_data = []
        for n_class in which_classes:
            pdp_interact_plot_data.append(pdp_interact_out[n_class])
    num_charts = len(pdp_interact_plot_data)

    # plot_pdp and plot_type grid is only True for numeric features when x_quantile is True
    if 'numeric' in pdp_interact_plot_data[0].feature_types and not x_quantile:
        plot_pdp = False
        plot_type = 'contour'

    # set up graph parameters
    width, height = 7, 9.5
    nrows = 1

    if num_charts > 1:
        nrows = int(np.ceil(num_charts * 1.0 / ncols))
        ncols = np.min([num_charts, ncols])
        width = np.min([7.5 * ncols, 15])
        height = np.min([width * 1.0 / ncols, 7.5]) * nrows

    if figsize is not None:
        width, height = figsize

    if plot_params is None:
        plot_params = dict()

    fig = plt.figure(figsize=(width, height))
    outer_grid = GridSpec(2, 1, wspace=0.0, hspace=0.1, height_ratios=[2, height - 2])
    title_ax = plt.subplot(outer_grid[0])
    fig.add_subplot(title_ax)
    n_grids = len(pdp_interact_plot_data[0].feature_grids[0]) * len(pdp_interact_plot_data[0].feature_grids[1])
    _pdp_plot_title(n_grids=n_grids, feature_name="%s and %s" % (feature_names[0], feature_names[1]),
                    ax=title_ax, plot_params=plot_params)

    if num_charts == 1:
        if plot_pdp:
            inner_grid = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_grid[1], height_ratios=[0.5, 7],
                                                 width_ratios=[0.5, 7], hspace=0.1, wspace=0)
            _pdp_inter_three(pdp_interact_out=pdp_interact_plot_data[0], feature_names=feature_names, plot_type=plot_type,
                             chart_grids=inner_grid, x_quantile=x_quantile, fig=fig, plot_params=plot_params)
        else:
            inter_ax = plt.subplot(outer_grid[1])
            fig.add_subplot(inter_ax)
            _pdp_inter_one(pdp_interact_out=pdp_interact_plot_data[0], feature_names=feature_names, plot_type=plot_type,
                           inter_ax=inter_ax, x_quantile=x_quantile, fig=fig, plot_params=plot_params, norm=None)
    else:
        inner_grid = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=outer_grid[1], wspace=0.1, hspace=0.2)
        inter_ax = []
        for inner_idx in range(num_charts):
            if plot_pdp:
                inner_inner_grid = GridSpecFromSubplotSpec(2, 2, subplot_spec=inner_grid[inner_idx],
                                                           height_ratios=[0.5, 7], width_ratios=[0.5, 7],
                                                           hspace=0.1, wspace=0)
                _pdp_inter_three(pdp_interact_out=pdp_interact_plot_data[inner_idx], feature_names=feature_names, plot_type=plot_type,
                                 chart_grids=inner_inner_grid, x_quantile=x_quantile, fig=fig, plot_params=plot_params)
            else:
                inner_inter_ax = plt.subplot(inner_grid[inner_idx])
                fig.add_subplot(inner_inter_ax)
                _pdp_inter_one(pdp_interact_out=pdp_interact_plot_data[inner_idx], feature_names=feature_names, plot_type=plot_type,
                               inter_ax=inner_inter_ax, x_quantile=x_quantile, fig=fig, plot_params=plot_params, norm=None)

