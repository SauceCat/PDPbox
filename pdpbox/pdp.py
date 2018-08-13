
from .pdp_calc_utils import _calc_ice_lines, _calc_ice_lines_inter, _prepare_pdp_count_data
from .pdp_plot_utils import (_pdp_plot, _pdp_inter_three, _pdp_inter_one)
from .utils import (_check_model, _check_dataset, _check_percentile_range, _check_feature,
                    _check_grid_type, _check_memory_limit, _check_frac_to_plot, _make_list, _expand_default,
                    _plot_title, _calc_memory_usage, _get_grids, _get_grid_combos, _check_classes, _calc_figsize,
                    _get_string)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore')


class PDPIsolate(object):
    """Save pdp_isolate results

    Parameters
    ----------

    n_classes: integer or None
        number of classes for classifier, None when it is a regressor
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

    def __init__(self, n_classes, which_class, feature, feature_type, feature_grids,
                 percentile_info, display_columns, ice_lines, pdp, count_data, hist_data):

        self._type = 'PDPIsolate_instance'
        self.n_classes = n_classes
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


def pdp_isolate(model, dataset, model_features, feature, num_grid_points=10, grid_type='percentile',
                percentile_range=None, grid_range=None, cust_grid_points=None,
                memory_limit=0.5, n_jobs=1, predict_kwds={}, data_transformer=None):
    """Calculate PDP isolation plot

    Parameters
    ----------
    model: a fitted sklearn model
    dataset: pandas DataFrame
        data set on which the model is trained
    model_features: list or 1-d array
        list of model features
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
    pdp_isolate_out: instance of PDPIsolate

    """

    # check function inputs
    n_classes, predict = _check_model(model=model)

    # avoid polluting the original dataset
    # copy training data set and get the model features
    # it's extremely important to keep the original feature order
    _check_dataset(df=dataset)
    _dataset = dataset.copy()

    feature_type = _check_feature(feature=feature, df=_dataset)
    _check_grid_type(grid_type=grid_type)
    _check_percentile_range(percentile_range=percentile_range)
    _check_memory_limit(memory_limit=memory_limit)

    # feature_grids: grid points to calculate on
    # display_columns: xticklabels for grid points
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
                feature_values=_dataset[feature].values, num_grid_points=num_grid_points, grid_type=grid_type,
                percentile_range=percentile_range, grid_range=grid_range)
        else:
            # make sure grid points are unique and in ascending order
            feature_grids = np.array(sorted(np.unique(cust_grid_points)))
        display_columns = [_get_string(v) for v in feature_grids]

    # Parallel calculate ICE lines
    true_n_jobs = _calc_memory_usage(
        df=_dataset, total_units=len(feature_grids), n_jobs=n_jobs, memory_limit=memory_limit)
    grid_results = Parallel(n_jobs=true_n_jobs)(
        delayed(_calc_ice_lines)(
            feature_grid, data=_dataset, model=model, model_features=model_features, n_classes=n_classes,
            feature=feature, feature_type=feature_type, predict_kwds=predict_kwds, data_transformer=data_transformer)
        for feature_grid in feature_grids)

    if n_classes > 2:
        ice_lines = []
        for n_class in range(n_classes):
            ice_line_n_class = pd.concat([grid_result[n_class] for grid_result in grid_results], axis=1)
            ice_lines.append(ice_line_n_class)
    else:
        ice_lines = pd.concat(grid_results, axis=1)

    # calculate the counts
    count_data = _prepare_pdp_count_data(
        feature=feature, feature_type=feature_type, data=_dataset[_make_list(feature)], feature_grids=feature_grids)

    # prepare histogram information for numeric feature
    hist_data = None
    if feature_type == 'numeric':
        hist_data = _dataset[feature].values

    # combine the final results
    pdp_params = {'n_classes': n_classes, 'feature': feature, 'feature_type': feature_type,
                  'feature_grids': feature_grids, 'percentile_info': percentile_info,
                  'display_columns': display_columns, 'count_data': count_data, 'hist_data': hist_data}
    if n_classes > 2:
        pdp_isolate_out = []
        for n_class in range(n_classes):
            pdp = ice_lines[n_class][feature_grids].mean().values
            pdp_isolate_out.append(
                PDPIsolate(which_class=n_class, ice_lines=ice_lines[n_class], pdp=pdp, **pdp_params))
    else:
        pdp = ice_lines[feature_grids].mean().values
        pdp_isolate_out = PDPIsolate(which_class=None, ice_lines=ice_lines, pdp=pdp, **pdp_params)

    return pdp_isolate_out


def pdp_plot(pdp_isolate_out, feature_name, center=True, plot_pts_dist=False, plot_lines=False, frac_to_plot=1,
             cluster=False, n_cluster_centers=None, cluster_method='accurate', x_quantile=False,
             show_percentile=False, figsize=None, ncols=2, plot_params=None, which_classes=None):
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

    Examples
    --------

    Quick start with pdp_plot

    .. highlight:: python
    .. code-block:: python

        from pdpbox import pdp, get_dataset

        test_titanic = get_dataset.titanic()
        titanic_data = test_titanic['data']
        titanic_target = test_titanic['target']
        titanic_features = test_titanic['features']
        titanic_model = test_titanic['xgb_model']

        pdp_sex = pdp.pdp_isolate(model=titanic_model,
                                  dataset=titanic_data,
                                  model_features=titanic_features,
                                  feature='Sex')
        fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_sex, feature_name='sex')


    With One-hot encoding features

    .. highlight:: python
    .. code-block:: python

        pdp_embark = pdp.pdp_isolate(model=titanic_model, dataset=titanic_data,
                                     model_features=titanic_features,
                                     feature=['Embarked_C', 'Embarked_S', 'Embarked_Q'])
        fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_embark,
                                 feature_name='Embark',
                                 center=True,
                                 plot_lines=True,
                                 frac_to_plot=100,
                                 plot_pts_dist=True)

    With numeric features

    .. highlight:: python
    .. code-block:: python

        pdp_fare = pdp.pdp_isolate(model=titanic_model,
                                   dataset=titanic_data,
                                   model_features=titanic_features,
                                   feature='Fare')
        fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_fare,
                                 feature_name='Fare',
                                 plot_pts_dist=True)


    With multi-class

    .. highlight:: python
    .. code-block:: python

        from pdpbox import pdp, get_dataset

        test_otto = get_dataset.otto()
        otto_data = test_otto['data']
        otto_features = test_otto['features']
        otto_model = test_otto['rf_model']
        otto_target = test_otto['target']

        pdp_feat_67_rf = pdp.pdp_isolate(model=otto_model,
                                         dataset=otto_data,
                                         model_features=otto_features,
                                         feature='feat_67')
        fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_feat_67_rf,
                                 feature_name='feat_67',
                                 center=True,
                                 x_quantile=True,
                                 ncols=3,
                                 plot_lines=True,
                                 frac_to_plot=100)

    """

    # check function inputs
    _check_frac_to_plot(frac_to_plot=frac_to_plot)
    pdp_plot_data = _make_list(x=pdp_isolate_out)
    n_grids = len(pdp_plot_data[0].feature_grids)
    if which_classes is not None:
        _check_classes(classes_list=which_classes, n_classes=pdp_plot_data[0].n_classes)

    # select the subset to plot
    if len(pdp_plot_data) > 1 and which_classes is not None:
        pdp_plot_data = []
        which_classes = sorted(which_classes)
        for n_class in which_classes:
            pdp_plot_data.append(pdp_isolate_out[n_class])

    # set up graph parameters
    width, height = 15, 9.5
    nrows = 1
    if len(pdp_plot_data) > 1:
        nrows = int(np.ceil(len(pdp_plot_data) * 1.0 / ncols))
        ncols = np.min([len(pdp_plot_data), ncols])
        width = np.min([7.5 * len(pdp_plot_data), 15])
        height = np.min([width * 1.0 / ncols, 7.5]) * nrows

    if figsize is not None:
        width, height = figsize

    if plot_params is None:
        plot_params = dict()

    # construct the chart
    fig = plt.figure(figsize=(width, height))
    outer_grid = GridSpec(2, 1, wspace=0.0, hspace=0.1, height_ratios=[2, height - 2])

    # plot title
    title_ax = plt.subplot(outer_grid[0])
    fig.add_subplot(title_ax)
    title = plot_params.get('title', 'PDP for feature "%s"' % feature_name)
    subtitle = plot_params.get('subtitle', "Number of unique grid points: %d" % n_grids)
    _plot_title(title=title, subtitle=subtitle, title_ax=title_ax, plot_params=plot_params)

    # plot pdp
    feature_type = pdp_plot_data[0].feature_type
    pdp_count_hspace = 0.15
    count_data = pdp_plot_data[0].count_data.copy()
    if feature_type == 'numeric':
        pdp_count_hspace = 0.25

    pdp_plot_params = {'center': center, 'plot_lines': plot_lines, 'frac_to_plot': frac_to_plot, 'cluster': cluster,
                       'n_cluster_centers': n_cluster_centers, 'cluster_method': cluster_method,
                       'x_quantile': x_quantile, 'show_percentile': show_percentile, 'count_data': count_data,
                       'plot_params': plot_params}

    if len(pdp_plot_data) == 1:
        # add class information if need
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

            _pdp_plot(pdp_isolate_out=pdp_plot_data[0], feature_name=feature_name_adj, pdp_ax=_pdp_ax,
                      count_ax=_count_ax, **pdp_plot_params)
        else:
            pdp_ax = plt.subplot(outer_grid[1])
            fig.add_subplot(pdp_ax)
            _pdp_plot(pdp_isolate_out=pdp_plot_data[0], feature_name=feature_name_adj, pdp_ax=pdp_ax,
                      count_ax=None, **pdp_plot_params)

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
                _pdp_plot(pdp_isolate_out=pdp_plot_data[inner_idx], feature_name=feature_name_adj, pdp_ax=_pdp_ax,
                          count_ax=_count_ax, **pdp_plot_params)
        else:
            pdp_ax = []
            for inner_idx in range(len(pdp_plot_data)):
                ax = plt.subplot(inner_grid[inner_idx])
                pdp_ax.append(ax)
                fig.add_subplot(ax)

                feature_name_adj = '%s (class %d)' % (feature_name, pdp_plot_data[inner_idx].which_class)
                _pdp_plot(pdp_isolate_out=pdp_plot_data[inner_idx], feature_name=feature_name_adj, pdp_ax=ax,
                          count_ax=None, **pdp_plot_params)

    axes = {'title_ax': title_ax, 'pdp_ax': pdp_ax}
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
    def __init__(self, n_classes, which_class, features, feature_types, feature_grids,
                 pdp_isolate_outs, pdp):

        self._type = 'PDPInteract_instance'
        self.n_classes = n_classes
        self.which_class = which_class
        self.features = features
        self.feature_types = feature_types
        self.feature_grids = feature_grids
        self.pdp_isolate_outs = pdp_isolate_outs
        self.pdp = pdp


def pdp_interact(model, dataset, model_features, features, num_grid_points=None, grid_types=None,
                 percentile_ranges=None, grid_ranges=None, cust_grid_points=None, memory_limit=0.5,
                 n_jobs=1, predict_kwds={}, data_transformer=None):
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
    grid_types = _expand_default(x=grid_types, default='percentile')
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
            model=model, dataset=_dataset, model_features=model_features, feature=features[idx],
            num_grid_points=num_grid_points[idx], grid_type=grid_types[idx], percentile_range=percentile_ranges[idx],
            grid_range=grid_ranges[idx], cust_grid_points=cust_grid_points[idx], memory_limit=memory_limit,
            n_jobs=n_jobs, predict_kwds=predict_kwds, data_transformer=data_transformer)
        pdp_isolate_outs.append(pdp_isolate_out)

    if n_classes > 2:
        feature_grids = [pdp_isolate_outs[0][0].feature_grids, pdp_isolate_outs[1][0].feature_grids]
        feature_types = [pdp_isolate_outs[0][0].feature_type, pdp_isolate_outs[1][0].feature_type]
    else:
        feature_grids = [pdp_isolate_outs[0].feature_grids, pdp_isolate_outs[1].feature_grids]
        feature_types = [pdp_isolate_outs[0].feature_type, pdp_isolate_outs[1].feature_type]

    # make features into list
    feature_list = _make_list(features[0]) + _make_list(features[1])

    # create grid combination
    grid_combos = _get_grid_combos(feature_grids, feature_types)

    # Parallel calculate ICE lines
    true_n_jobs = _calc_memory_usage(
        df=_dataset, total_units=len(grid_combos), n_jobs=n_jobs, memory_limit=memory_limit)

    grid_results = Parallel(n_jobs=true_n_jobs)(delayed(_calc_ice_lines_inter)(
        grid_combo, data=_dataset, model=model, model_features=model_features, n_classes=n_classes,
        feature_list=feature_list, predict_kwds=predict_kwds, data_transformer=data_transformer)
                                                for grid_combo in grid_combos)

    ice_lines = pd.concat(grid_results, axis=0).reset_index(drop=True)
    pdp = ice_lines.groupby(feature_list, as_index=False).mean()

    # combine the final results
    pdp_interact_params = {'n_classes': n_classes, 'features': features, 'feature_types': feature_types,
                           'feature_grids': feature_grids}
    if n_classes > 2:
        pdp_interact_out = []
        for n_class in range(n_classes):
            _pdp = pdp[feature_list + ['class_%d_preds' % n_class]].rename(
                columns={'class_%d_preds' % n_class: 'preds'})
            pdp_interact_out.append(
                PDPInteract(which_class=n_class,
                            pdp_isolate_outs=[pdp_isolate_outs[0][n_class], pdp_isolate_outs[1][n_class]],
                            pdp=_pdp, **pdp_interact_params))
    else:
        pdp_interact_out = PDPInteract(
            which_class=None, pdp_isolate_outs=pdp_isolate_outs, pdp=pdp, **pdp_interact_params)

    return pdp_interact_out


def pdp_interact_plot(pdp_interact_out, feature_names, plot_type='contour', x_quantile=False, plot_pdp=False,
                      which_classes=None, figsize=None, ncols=2, plot_params=None):
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
        _check_classes(classes_list=which_classes, n_classes=pdp_interact_plot_data[0].n_classes)

    # multi-class problem
    if len(pdp_interact_plot_data) > 1 and which_classes is not None:
        pdp_interact_plot_data = []
        for n_class in which_classes:
            pdp_interact_plot_data.append(pdp_interact_out[n_class])
    num_charts = len(pdp_interact_plot_data)

    inner_hspace = inner_wspace = 0
    if plot_type == 'grid' or plot_pdp:
        x_quantile = True
        if plot_type == 'grid':
            inner_hspace = inner_wspace = 0.1

    # calculate figure size
    title_height = 2
    if plot_pdp:
        unit_figsize = (10.5, 10.5)
    else:
        unit_figsize = (7.5, 7.5)

    width, height, nrows, ncols = _calc_figsize(
        num_charts=num_charts, ncols=ncols, title_height=title_height, unit_figsize=unit_figsize)
    if figsize is not None:
        width, height = figsize

    if plot_params is None:
        plot_params = dict()

    fig = plt.figure(figsize=(width, height))
    outer_grid = GridSpec(2, 1, wspace=0.0, hspace=0.1, height_ratios=[title_height, height - title_height])
    title_ax = plt.subplot(outer_grid[0])
    fig.add_subplot(title_ax)

    n_grids = [len(pdp_interact_plot_data[0].feature_grids[0]), len(pdp_interact_plot_data[0].feature_grids[1])]
    title = plot_params.get('title', 'PDP interact for "%s" and "%s"' % (feature_names[0], feature_names[1]))
    subtitle = plot_params.get('subtitle', "Number of unique grid points: (%s: %d, %s: %d)"
                               % (feature_names[0], n_grids[0], feature_names[1], n_grids[1]))

    _plot_title(title=title, subtitle=subtitle, title_ax=title_ax, plot_params=plot_params)

    inter_params = {'plot_type': plot_type, 'x_quantile': x_quantile, 'plot_params': plot_params}
    if num_charts == 1:
        feature_names_adj = feature_names
        if pdp_interact_plot_data[0].which_class is not None:
            feature_names_adj = ['%s (class %d)' % (
                feature_names[0], pdp_interact_plot_data[0].which_class), feature_names[1]]
        if plot_pdp:
            inner_grid = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_grid[1], height_ratios=[0.5, 7],
                                                 width_ratios=[0.5, 7], hspace=inner_hspace, wspace=inner_wspace)
            inter_ax = _pdp_inter_three(pdp_interact_out=pdp_interact_plot_data[0], chart_grids=inner_grid,
                                        fig=fig, feature_names=feature_names_adj, **inter_params)
        else:
            inter_ax = plt.subplot(outer_grid[1])
            fig.add_subplot(inter_ax)
            _pdp_inter_one(pdp_interact_out=pdp_interact_plot_data[0], inter_ax=inter_ax, norm=None,
                           feature_names=feature_names_adj, **inter_params)
    else:
        wspace = 0.3
        if plot_pdp and plot_type == 'grid':
            wspace = 0.35
        inner_grid = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=outer_grid[1], wspace=wspace, hspace=0.2)
        inter_ax = []
        for inner_idx in range(num_charts):
            feature_names_adj = ['%s (class %d)' % (
                feature_names[0], pdp_interact_plot_data[inner_idx].which_class), feature_names[1]]
            if plot_pdp:
                inner_inner_grid = GridSpecFromSubplotSpec(2, 2, subplot_spec=inner_grid[inner_idx],
                                                           height_ratios=[0.5, 7], width_ratios=[0.5, 7],
                                                           hspace=inner_hspace, wspace=inner_wspace)
                inner_inter_ax = _pdp_inter_three(
                    pdp_interact_out=pdp_interact_plot_data[inner_idx], chart_grids=inner_inner_grid, fig=fig,
                    feature_names=feature_names_adj, **inter_params)
            else:
                inner_inter_ax = plt.subplot(inner_grid[inner_idx])
                fig.add_subplot(inner_inter_ax)
                _pdp_inter_one(pdp_interact_out=pdp_interact_plot_data[inner_idx], inter_ax=inner_inter_ax,
                               norm=None, feature_names=feature_names_adj, **inter_params)
            inter_ax.append(inner_inter_ax)

    axes = {'title_ax': title_ax, 'pdp_inter_ax': inter_ax}

    return fig, axes

