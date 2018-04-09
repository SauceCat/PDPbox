import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import info_plot_utils, pdp_calc_utils


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
        info_plot_utils._actual_plot_title(feature_name=feature_name, ax=ax1, figsize=figsize,
                                           multi_flag=multi_flag, which_class=which_class, plot_params=plot_params)

        if ncols is None:
            ncols = 2
        nrows = int(np.ceil(float(n_classes) / ncols))

        plt.figure(figsize=(figwidth, (figwidth / ncols) * nrows))
        outer = GridSpec(nrows, ncols, wspace=0.2, hspace=0.2)

        for n_class in range(n_classes):
            info_plot_utils._actual_plot(pdp_isolate_out=pdp_isolate_out['class_%d' % n_class],
                                         feature_name=feature_name + ' class_%d' % n_class,
                                         figwidth=figwidth, plot_params=plot_params, outer=outer[n_class])
    else:
        if multi_flag:
            _pdp_isolate_out = pdp_isolate_out['class_%d' % which_class]
        else:
            _pdp_isolate_out = pdp_isolate_out

        info_plot_utils._actual_plot_title(feature_name=feature_name, ax=ax1,
                                           figsize=figsize, multi_flag=multi_flag, which_class=which_class, plot_params=plot_params)

        info_plot_utils._actual_plot(pdp_isolate_out=_pdp_isolate_out, feature_name=feature_name,
                                     figwidth=figwidth, plot_params=plot_params, outer=None)


def target_plot(df, feature, feature_name, target, num_grid_points=10, percentile_range=None,
                cust_grid_points=None, figsize=None, plot_params=None):
    """
    Plot target distribution through feature grids

    :param df: pandas DataFrame
        the whole dataset to investigate, including at least the feature to investigate as well as the target values
    :param feature: string or list
        column to investigate (for one-hot encoding features, a list of columns should be provided)
    :param feature_name: string
        name of the feature, not necessary the same as the column name
    :param target: string or list
        the column name of the target value
        for multi-class problem, a list of one-hot encoding target values could be provided
    :param num_grid_points: integer, default=10
        number of grid points for numeric features
    :param percentile_range: (low, high), default=None
        percentile range to consider for numeric features
    :param cust_grid_points: list, default=None
        customized grid points
    :param figsize: (width, height), default=None
        figure size
    :param plot_params: dict, default=None
        values of plot parameters
    """

    # check input dataset
    if type(df) != pd.core.frame.DataFrame:
        raise ValueError('df: only accept pandas DataFrame')

    # check feature
    if type(feature) == str:
        if feature not in df.columns.values:
            raise ValueError('feature does not exist: %s' % (feature))
        if sorted(list(np.unique(df[feature]))) == [0, 1]:
            feature_type = 'binary'
        else:
            feature_type = 'numeric'
    elif type(feature) == list:
        if len(feature) < 2:
            raise ValueError('one-hot encoding feature should contain more than 1 element')
        if not set(feature) < set(df.columns.values):
            raise ValueError('feature does not exist: %s' % (str(feature)))
        feature_type = 'onehot'
    else:
        raise ValueError('feature: please pass a string or a list (for onehot encoding feature)')

    # check percentile_range
    if percentile_range is not None:
        if len(percentile_range) != 2:
            raise ValueError('percentile_range: should contain 2 elements')
        if np.max(percentile_range) > 100 or np.min(percentile_range) < 0:
            raise ValueError('percentile_range: should be between 0 and 100')

    # check cust_grid_points
    if (feature_type != 'numeric') and (cust_grid_points is not None):
        raise ValueError('only numeric feature can accept cust_grid_points')

    # check target values and calculate target rate through feature grids
    if type(target) == str:
        if target not in df.columns.values:
            raise ValueError('target does not exist: %s' % (target))
        if sorted(list(np.unique(df[target]))) == [0, 1]:
            target_type = 'binary'
        else:
            target_type = 'regression'
    elif type(target) == list:
        if len(target) < 2:
            raise ValueError('multi-class target should contain more than 1 element')
        if not set(target) < set(df.columns.values):
            raise ValueError('target does not exist: %s' % (str(target)))
        for target_idx in range(len(target)):
            if sorted(list(np.unique(df[target[target_idx]]))) != [0, 1]:
                raise ValueError('multi-class targets should be one-hot encoded: %s' % (str(target[target_idx])))
        target_type = 'multi-class'
    else:
        raise ValueError('target: please pass a string or a list (for multi-class targets)')

    # create feature grids and bar counts
    useful_features = []
    if type(feature) == list:
        useful_features += feature
    else:
        useful_features.append(feature)
    if type(target) == list:
        useful_features += target
    else:
        useful_features.append(target)

    # prepare data for bar plot
    bar_counts = df[useful_features].copy()
    if feature_type == 'binary':
        feature_grids = np.array([0, 1])
        bar_counts['x'] = bar_counts[feature]
    if feature_type == 'numeric':
        if cust_grid_points is None:
            feature_grids = pdp_calc_utils._get_grids(df[feature], num_grid_points, percentile_range)
        else:
            feature_grids = np.array(sorted(cust_grid_points))
        bar_counts = bar_counts[(bar_counts[feature] >= feature_grids[0])
                                & (bar_counts[feature] <= feature_grids[-1])].reset_index(drop=True)
        bar_counts['x'] = bar_counts[feature].apply(lambda x: pdp_calc_utils._find_closest(x, feature_grids))
    if feature_type == 'onehot':
        feature_grids = np.array(feature)
        bar_counts['x'] = bar_counts[feature].apply(lambda x: pdp_calc_utils._find_onehot_actual(x), axis=1)
        bar_counts = bar_counts[bar_counts['x'].isnull() == False].reset_index(drop=True)

    bar_counts['fake_count'] = 1
    bar_counts_gp = bar_counts.groupby('x', as_index=False).agg({'fake_count': 'count'})

    # prepare data for target lines
    target_lines = []
    if target_type in ['binary', 'regression']:
        target_line = bar_counts.groupby('x', as_index=False).agg({target: 'mean'})
        target_lines.append(target_line)
    else:
        for target_idx in range(len(target)):
            target_line = bar_counts.groupby('x', as_index=False).agg({target[target_idx]: 'mean'})
            target_lines.append(target_line)

    info_plot_utils._target_plot(feature_name=feature_name, feature_grids=feature_grids, target=target,
                                 bar_counts_gp=bar_counts_gp, target_lines=target_lines, figsize=figsize, plot_params=plot_params)






