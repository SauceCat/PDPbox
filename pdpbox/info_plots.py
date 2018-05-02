
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from info_plot_utils import _target_plot, _target_plot_interact, _prepare_data_x, _actual_plot, _actual_plot_title
from other_utils import _check_feature, _check_percentile_range, _check_target, _make_list, _expand_default


def actual_plot(pdp_isolate_out, feature_name, figsize=None, plot_params=None,
                multi_flag=False, which_class=None, ncols=None):
    """
    Plot actual prediction distribution through feature grids

    :param pdp_isolate_out: instance of pdp_isolate_obj
        a calculated pdp_isolate_obj instance
    :param feature_name: tring
        name of the feature, not necessary the same as the column name
    :param figsize: (width, height), default=None
        figure size
    :param plot_params: dict, default=None
        values of plot parameters
    :param multi_flag: boolean, default=False
        whether it is a subplot of a multiclass plot
    :param which_class: integer, default=None
        which class to plot
    :param ncols: integer, default=None
        used under multi-class mode
    """

    # check which_class
    if multi_flag and which_class >= len(pdp_isolate_out.keys()):
        raise ValueError('which_class: class does not exist')

    if figsize is None:
        figwidth = 16
    else:
        figwidth = figsize[0]

    # draw graph title
    plt.figure(figsize=(figwidth, figwidth / 6.7))
    ax1 = plt.subplot(111)

    if type(pdp_isolate_out) == dict and not multi_flag:
        n_classes = len(pdp_isolate_out.keys())
        _actual_plot_title(feature_name=feature_name, ax=ax1, figsize=figsize,
                                           multi_flag=multi_flag, which_class=which_class, plot_params=plot_params)

        if ncols is None:
            ncols = 2
        nrows = int(np.ceil(float(n_classes) / ncols))

        plt.figure(figsize=(figwidth, (figwidth / ncols) * nrows))
        outer = GridSpec(nrows, ncols, wspace=0.2, hspace=0.2)

        for n_class in range(n_classes):
            _actual_plot(pdp_isolate_out=pdp_isolate_out['class_%d' % n_class],
                                         feature_name=feature_name + ' class_%d' % n_class,
                                         figwidth=figwidth, plot_params=plot_params, outer=outer[n_class])
    else:
        if multi_flag:
            _pdp_isolate_out = pdp_isolate_out['class_%d' % which_class]
        else:
            _pdp_isolate_out = pdp_isolate_out

        _actual_plot_title(feature_name=feature_name, ax=ax1,
                                           figsize=figsize, multi_flag=multi_flag, which_class=which_class, plot_params=plot_params)

        _actual_plot(pdp_isolate_out=_pdp_isolate_out, feature_name=feature_name,
                                     figwidth=figwidth, plot_params=plot_params, outer=None)


def target_plot(df, feature, feature_name, target, num_grid_points=10, grid_type='percentile',
                percentile_range=None, grid_range=None, cust_grid_points=None, show_percentile=False,
                show_outliers=False, figsize=None, plot_params=None):
    """Plot average target value across different feature values (feature grids)

    Parameters:
    -----------

    :param df: pandas DataFrame
        data set to investigate on, should contain at least
        the feature to investigate as well as the target
    :param feature: string or list
        feature or feature list to investigate
        for one-hot encoding features, feature list is required
    :param feature_name: string
        name of the feature, not necessary a column name
    :param target: string or list
        column name or column name list for target value
        for multi-class problem, a list of one-hot encoding target column
    :param num_grid_points: integer, optional, default=10
        number of grid points for numeric feature
    :param grid_type: string, optional, default='percentile'
        'percentile' or 'equal'
        type of grid points for numeric feature
    :param percentile_range: tuple or None, optional, default=None
        percentile range to investigate
        for numeric feature when grid_type='percentile'
    :param grid_range: tuple or None, optional, default=None
        value range to investigate
        for numeric feature when grid_type='equal'
    :param cust_grid_points: Series, 1d-array, list or None, optional, default=None
        customized list of grid points
        for numeric feature
    :param show_percentile: bool, optional, default=False
        whether to display the percentile buckets
        for numeric feature when grid_type='percentile'
    :param show_outliers: bool, optional, default=False
        whether to display the out of range buckets
        for numeric feature when percentile_range or grid_range is not None
    :param figsize: tuple or None, optional, default=None
        size of the figure, (width, height)
    :param plot_params: dict or None, optional, default=None
        parameters for the plot

    Return:
    -------

    :return axes: matplotlib Axes
        Returns the Axes object with the plot for further tweaking
    """

    # check feature
    feature_type = _check_feature(feature=feature, df=df)

    # check percentile_range
    _check_percentile_range(percentile_range=percentile_range)

    # check show_outliers
    if (percentile_range is None) and (grid_range is None) and (cust_grid_points is None):
        show_outliers = False

    # create feature grids and bar counts
    target = _make_list(target)
    useful_features = [] + _make_list(feature) + target

    # prepare data for bar plot
    data = df[useful_features].copy()
    data_x, display_columns, percentile_columns = _prepare_data_x(
        feature=feature, feature_type=feature_type, data=data, num_grid_points=num_grid_points, grid_type=grid_type,
        percentile_range=percentile_range, grid_range=grid_range, cust_grid_points=cust_grid_points,
        show_percentile=show_percentile, show_outliers=show_outliers)

    data_x['fake_count'] = 1
    bar_data = data_x.groupby('x', as_index=False).agg({'fake_count': 'count'}).sort_values('x', ascending=True)
    summary_df = bar_data.rename(columns={'fake_count': 'count'})

    # prepare data for target lines
    target_lines = []
    for target_idx in range(len(target)):
        target_line = data_x.groupby('x', as_index=False).agg(
            {target[target_idx]: 'mean'}).sort_values('x', ascending=True)
        target_lines.append(target_line)
        summary_df = summary_df.merge(target_line, on='x', how='outer').fillna(0)

    summary_df['display_column'] = summary_df['x'].apply(lambda x: display_columns[x])
    info_cols = ['x', 'display_column']
    if len(percentile_columns) != 0:
        summary_df['percentile_column'] = summary_df['x'].apply(lambda x: percentile_columns[x])
        info_cols.append('percentile_column')
    summary_df = summary_df[info_cols + ['count'] + target]

    axes = _target_plot(
        feature_name=feature_name, display_columns=display_columns, percentile_columns=percentile_columns,
        target=target, bar_data=bar_data, target_lines=target_lines, figsize=figsize, plot_params=plot_params)
    return axes, summary_df


def target_plot_interact(df, features, feature_names, target, num_grid_points=None, grid_types=None,
                         percentile_ranges=None, grid_ranges=None, cust_grid_points=None, show_percentile=False,
                         show_outliers=False, figsize=None, ncols=2, plot_params=None):

    # check show_outliers
    num_grid_points = _expand_default(num_grid_points, 10)
    grid_types = _expand_default(grid_types, 'percentile')
    percentile_ranges = _expand_default(percentile_ranges, None)
    grid_ranges = _expand_default(grid_ranges, None)
    cust_grid_points = _expand_default(cust_grid_points, None)

    if not show_outliers:
        show_outliers = [False, False]
    else:
        show_outliers = [True, True]
        for i in range(2):
            if (percentile_ranges[i] is None) and (grid_ranges[i] is None) and (cust_grid_points[i] is None):
                show_outliers[i] = False

    # check features
    feature_types = [_check_feature(feature=features[0], df=df), _check_feature(feature=features[1], df=df)]

    # check percentile_range
    _check_percentile_range(percentile_range=percentile_ranges[0])
    _check_percentile_range(percentile_range=percentile_ranges[1])

    # create feature grids and bar counts
    target = _make_list(target)
    useful_features = [] + _make_list(features[0]) + _make_list(features[1]) + target

    # prepare data for bar plot
    data = df[useful_features].copy()

    # (data_x, display_columns, percentile_columns)
    results = []
    for i in range(2):
        result = _prepare_data_x(
            feature=features[i], feature_type=feature_types[i], data=data[_make_list(features[i])],
            num_grid_points=num_grid_points[i], grid_type=grid_types[i], percentile_range=percentile_ranges[i],
            grid_range=grid_ranges[i], cust_grid_points=cust_grid_points[i],
            show_percentile=show_percentile, show_outliers=show_outliers[i])
        results.append(result)

    data_x = pd.concat([results[0][0].rename(columns={'x': 'x1'}),
                        results[1][0].rename(columns={'x': 'x2'}),
                        data[target]], axis=1).reset_index(drop=True)

    data_x['fake_count'] = 1
    agg_dict = {}
    for t in target:
        agg_dict[t] = 'mean'
    agg_dict['fake_count'] = 'count'
    target_count_data = data_x.groupby(['x1', 'x2'], as_index=False).agg(agg_dict)
    summary_df = target_count_data.rename(columns={'fake_count': 'count'})

    info_cols = ['x1', 'x2', 'display_column_1', 'display_column_2']
    for i in range(2):
        summary_df['display_column_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(lambda x: results[i][1][x])
        if len(results[i][2]) != 0:
            summary_df['percentile_column_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(lambda x: results[i][2][x])
            info_cols.append('percentile_column_%d' % (i + 1))
    summary_df = summary_df[info_cols + ['count'] + target]

    axes = _target_plot_interact(
        feature_names=feature_names, display_columns=[results[0][1], results[1][1]],
        percentile_columns=[results[0][2], results[1][2]], target=target, target_count_data=target_count_data,
        figsize=figsize, ncols=ncols, plot_params=plot_params)
    return axes, summary_df



